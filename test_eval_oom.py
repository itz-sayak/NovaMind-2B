"""
Quick DDP test to verify that evaluation does NOT OOM on 2× L40S.

Simulates what happens at step 500: model wrapped in DDP, forward with
targets in eval mode. If the fix works, this completes without CUDA OOM.

Run with:
  torchrun --nproc_per_node=2 test_eval_oom.py
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.model_config import NovaMind2BConfig
from model.transformer import NovaMind2B

def main():
    # ---- DDP init ----
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dtype = torch.bfloat16
    seq_len = 65536
    batch_size = 1

    if rank == 0:
        print("=" * 60)
        print(f"OOM Fix Test — DDP eval with targets  (world_size={world_size})")
        print("=" * 60)
        for i in range(world_size):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
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
        print(f"\nCreating dummy batch: batch_size={batch_size}, seq_len={seq_len}")
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # ---- Simulate a training step first (to fill cache like real training) ----
    if rank == 0:
        print("Running one training forward+backward (to fill GPU cache)...")
    model.train()
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    with ctx:
        result = model(x, targets=y)
    result["loss"].backward()
    model.zero_grad(set_to_none=True)

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
