"""
Training configuration for NovaMind-2B (2× NVIDIA L40S).
"""
from dataclasses import dataclass, field


@dataclass
class PretrainConfig:
    """Stage 1: Pretraining on code + math + general text (native 64K context)."""
    # Data
    data_dir: str = "/mnt/zone/A/datasets/pretrain"

    # Training
    # 2 GPUs × batch=1 × grad_accum=1 × seq=65536 = 131,072 tokens/step
    # 1.5M steps × 131K ≈ 196B tokens  (Chinchilla-optimal for 3B is ~60B;
    # ~3× Chinchilla intentionally overtrains for inference efficiency)
    batch_size: int = 1             # micro batch per GPU (tight at 65K, needs all VRAM)
    gradient_accumulation_steps: int = 1
    max_steps: int = 1_500_000
    warmup_steps: int = 2000
    
    # Optimizer (Muon + AdamW)
    # 3B scale: keep 2.2e-4 (LLaMA-3.2-3B uses 3e-4; 2.2e-4 is conservative/stable)
    learning_rate: float = 2.2e-4
    min_lr: float = 0.0             # WSD decays to 0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Muon-specific
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.01
    
    # Schedule (WSD = Warmup-Stable-Decay)
    lr_schedule: str = "wsd"
    decay_fraction: float = 0.15    # final 15% of training is cosine decay to 0
    
    # IO
    eval_interval: int = 500
    log_interval: int = 10
    save_interval: int = 5_000
    output_dir: str = "/mnt/zone/A/checkpoints/novamind-2b/pretrain"
    
    # Batch-size warmup (grad_accum ramps from initial -> full)
    # Both set to 1 → no warmup; ramp only applies when initial < target.
    grad_accum_initial: int = 1
    grad_accum_warmup_steps: int = 5000

    # EMA (stored on CPU)
    ema_enabled: bool = True
    ema_decay: float = 0.9995

    # Multi-stage data phases
    data_phases: list = field(default_factory=list)

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 4          # DataLoader worker processes (0 = main process only)
    shuffle_buffer: int = 2048    # StreamingPretrainDataset shuffle ring buffer size
    
    # Multi-GPU — set ddp=True and launch with torchrun --nproc_per_node=2
    ddp: bool = True
    
    # MTP
    use_mtp: bool = True
    mtp_loss_weight: float = 0.3
    mtp_loss_weight_final: float = 0.1


@dataclass
class SFTConfig:
    """Stage 2: Supervised Fine-Tuning on instruction data."""
    data_dir: str = "/mnt/zone/A/datasets/sft"
    # Context extension: SFT runs at 128K (16× pretrain length).
    # rope_scale_factor divides token positions by 16, keeping them inside
    # the pretrain RoPE distribution (position interpolation).
    max_seq_len: int = 131072
    rope_scale_factor: float = 16.0
    
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_epochs: int = 2
    max_steps: int = -1
    warmup_steps: int = 100
    
    learning_rate: float = 5e-6
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    lr_schedule: str = "cosine"
    
    eval_interval: int = 200
    log_interval: int = 10
    save_interval: int = 500
    output_dir: str = "/mnt/zone/A/checkpoints/novamind-2b/sft"
    
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    gradient_checkpointing: bool = True
    
    pretrained_ckpt: str = "/mnt/zone/A/checkpoints/novamind-2b/pretrain/latest.pt"


@dataclass
class DPOConfig:
    """Stage 3: Direct Preference Optimization."""
    data_dir: str = "/mnt/zone/A/datasets/dpo"
    # Mirrors SFTConfig: 128K context with position interpolation.
    max_seq_len: int = 131072
    rope_scale_factor: float = 16.0
    
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_steps: int = 5000
    warmup_steps: int = 100
    
    learning_rate: float = 5e-7
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    beta: float = 0.1  # KL penalty coefficient
    
    lr_schedule: str = "cosine"
    
    eval_interval: int = 200
    log_interval: int = 10
    save_interval: int = 500
    output_dir: str = "/mnt/zone/A/checkpoints/novamind-2b/dpo"
    
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    gradient_checkpointing: bool = True
    
    sft_ckpt: str = "/mnt/zone/A/checkpoints/novamind-2b/sft/latest.pt"


@dataclass
class LongContextConfig:
    """Context extension: continued pretraining from the 64K checkpoint to 100K.

    Strategy:
      - NTK-aware RoPE base scaling: ~6.5e6 for 100K
      - Lower LR (~10% of pretrain peak) with a fresh WSD schedule
      - Short run (~5K steps ≈ 1B tokens)
      - Load weights only from the 64K checkpoint (discard stale optimizer state)

    Launch command (100K example):
        torchrun --nproc_per_node=2 train.py \\
            --resume <64k_output_dir>/latest.pt \\
            --weights-only --reset-step \\
            --seq-len 102400 \\
            --rope-base 6500000 \\
            --max-steps 5000 \\
            --grad-accum 1 \\
            --data-dir <data_dir> \\
            --output-dir <output_dir>/longctx_100k \\
            --wandb --wandb-run-name longctx-100k
    """
    # Data — same tokenized bin as pretrain
    data_dir: str = "/mnt/zone/A/datasets/pretrain"

    # --- Context / RoPE (100K extension from 64K base) ---
    max_seq_len: int = 102400         # 100K
    rope_base: float = 6_500_000.0    # NTK-aware for 100K

    # --- Training (fresh schedule, 10% of pretrain LR) ---
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    # 5000 steps × 2GPU × 1 × 1 × 102400 ≈ 1B tokens
    max_steps: int = 5_000
    warmup_steps: int = 200
    decay_fraction: float = 0.20      # final 20% cosine decay

    learning_rate: float = 2.2e-5     # 10% of pretrain peak (2.2e-4)
    min_lr: float = 0.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    muon_lr: float = 2.0e-3           # 10% of pretrain muon_lr
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.01

    lr_schedule: str = "wsd"

    eval_interval: int = 200
    log_interval: int = 10
    save_interval: int = 1_000
    output_dir: str = "/mnt/zone/A/checkpoints/novamind-2b/longctx_100k"

    # Batch-size warmup: no ramp (grad_accum=1 throughout)
    grad_accum_initial: int = 1
    grad_accum_warmup_steps: int = 500

    ema_enabled: bool = True
    ema_decay: float = 0.9995

    data_phases: list = field(default_factory=list)

    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 4
    shuffle_buffer: int = 512         # smaller buffer: 64K seqs are 128× heavier in RAM

    ddp: bool = True
    use_mtp: bool = True
    mtp_loss_weight: float = 0.3
    mtp_loss_weight_final: float = 0.1
