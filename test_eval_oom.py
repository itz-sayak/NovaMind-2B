"""
Quick DDP test to verify that evaluation does NOT OOM on 2× L40S.

Simulates what happens at step 500: model wrapped in DDP, forward with
targets in eval mode using REAL tokens from train.bin.

Run with:
  torchrun --nproc_per_node=2 test_eval_oom.py
"""
import os
import numpy as np
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.model_config import NovaMind2BConfig
from model.transformer import NovaMind2B

DATA_DIR = "/iitgn/home/sayak.dutta/GPT/datasets"


def load_real_batch(data_dir, seq_len, batch_size, rank, device):
    """Load batch_size real sequences from train.bin (offset by rank)."""
    bin_file = os.path.join(data_dir, "train.bin")
    data = np.memmap(bin_file, dtype=np.uint32, mode='r')
    xs, ys = [], []
    for b in range(batch_size):
        start = (rank * batch_size + b) * (seq_len + 1)
        chunk = torch.from_numpy(data[start : start + seq_len + 1].astype(np.int64))
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.stack(xs).to(device)  # (batch_size, seq_len)
    y = torch.stack(ys).to(device)
    return x, y

def main():
    # ---- DDP init ----
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dtype = torch.bfloat16
    seq_len = 65536
    batch_size = 2
    grad_accum = 4  # matches gradient_accumulation_steps = 4

    if rank == 0:
        print("=" * 60)
        print(f"OOM Fix Test — DDP eval with targets  (world_size={world_size})")
        print("=" * 60)
        for i in range(world_size):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # ---- Model ----
    if rank == 0:
        print("\nInitializing model...")
    config = NovaMind2BConfig()
    model = NovaMind2B(config).to(device=device, dtype=dtype)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,} ({total_params/1e9:.3f}B)")

    # ---- Dummy batch (same shape as training) ----
    if rank == 0:
        print(f"\nLoading real tokens from {DATA_DIR}/train.bin ...")
    x, y = load_real_batch(DATA_DIR, seq_len, batch_size, rank, device)
    if rank == 0:
        print(f"  Batch shape: x={tuple(x.shape)}, y={tuple(y.shape)}")

    # ---- Simulate grad_accum training steps (to fill cache like real training) ----
    # train.py calls raw model (bypassing DDP) with no_sync() for accumulation,
    # then syncs only on the last micro-step.
    if rank == 0:
        print(f"Running {grad_accum} gradient accumulation steps (forward+backward each)...")
    model.train()
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    raw = model.module
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    for acc_step in range(grad_accum):
        sync_ctx = nullcontext() if acc_step == grad_accum - 1 else model.no_sync()
        with sync_ctx:
            with ctx:
                result = raw(x, targets=y)
            (result["loss"] / grad_accum).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if rank == 0:
        train_peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Peak after training step: {train_peak:.2f} GB")

    # ---- Now test evaluation (the path that OOM'd) ----
    torch.cuda.empty_cache()  # same as the fix in train.py
    model.module.eval()
    dist.barrier()

    if rank == 0:
        alloc_before = torch.cuda.memory_allocated(device) / 1e9
        print(f"\nGPU memory before eval: {alloc_before:.2f} GB")
        print("Running eval forward pass (this is where OOM used to happen)...")

    try:
        with torch.no_grad():
            with ctx:
                result = model.module(x, targets=y)
        loss = result["loss"].item()

        # Gather loss from all ranks
        loss_tensor = torch.tensor(loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        if rank == 0:
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"\n*** PASS *** — No OOM!")
            print(f"  Avg loss: {loss_tensor.item():.4f}")
            print(f"  Peak GPU memory (rank 0): {peak:.2f} GB")
            if result["logits"] is None:
                print("  Logits: None (correct — not wasting 12 GB)")
            else:
                print(f"  WARNING: logits materialized — shape {result['logits'].shape}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[rank {rank}] *** FAIL *** — Still OOM: {e}")
        dist.destroy_process_group()
        return 1

    model.module.train()
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("\nTest complete — safe to restart training.")
    return 0

if __name__ == "__main__":
    exit(main())
