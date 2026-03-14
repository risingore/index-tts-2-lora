"""IndexTTS-2 LoRA: Fine-tuning script for voice cloning.

Applies LoRA to GPT-2 attention + MLP layers in UnifiedVoice.
Uses precomputed features from extract_codec.py.

Loss: text_weight * CE(text) + (1 - text_weight) * CE(mel)
Optimizer: LoRA+ (B-matrix gets higher LR)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning for IndexTTS-2 GPT model",
    )
    # Paths
    p.add_argument("--indextts-dir", type=str, required=True,
                    help="Path to IndexTTS-2 installation")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Path to model checkpoints (default: {indextts-dir}/checkpoints)")
    p.add_argument("--data-dir", type=str, required=True,
                    help="Directory with extracted features (output of extract_codec.py)")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for LoRA checkpoints (default: {indextts-dir}/results/lora)")
    p.add_argument("--device", type=str, default=None,
                    help="Device (default: cuda if available)")

    # LoRA config
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    p.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")

    # Training config
    p.add_argument("--lr", type=float, default=5e-5, help="Base learning rate (default: 5e-5)")
    p.add_argument("--lora-plus-ratio", type=float, default=8.0,
                    help="LoRA+ B-matrix LR multiplier (default: 8.0)")
    p.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    p.add_argument("--warmup-ratio", type=float, default=0.1,
                    help="Warmup ratio of total steps (default: 0.1)")
    p.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    p.add_argument("--text-weight", type=float, default=0.1,
                    help="Text loss weight in dual CE loss (default: 0.1)")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="Max gradient norm for clipping (default: 1.0)")
    p.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader workers (default: 2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


class FinetuneDataset(Dataset):
    """Load precomputed features from .npz files."""

    def __init__(self, metadata_path: Path) -> None:
        self.entries: list[dict[str, Any]] = []
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.entries[idx]
        data = np.load(entry["npz"])
        return {
            "codes": torch.from_numpy(data["codes"].astype(np.int64)),
            "condition": torch.from_numpy(data["condition"]).float(),
            "emo_vec": torch.from_numpy(data["emo_vec"]).float(),
            "text_tokens": torch.from_numpy(data["text_tokens"].astype(np.int64)),
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences."""
    codes_list = [b["codes"] for b in batch]
    text_list = [b["text_tokens"] for b in batch]

    codes_padded = nn.utils.rnn.pad_sequence(codes_list, batch_first=True, padding_value=0)
    text_padded = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)

    return {
        "codes": codes_padded,
        "codes_lengths": torch.tensor([len(c) for c in codes_list], dtype=torch.long),
        "text_tokens": text_padded,
        "text_lengths": torch.tensor([len(t) for t in text_list], dtype=torch.long),
        "condition": torch.stack([b["condition"] for b in batch]),
        "emo_vec": torch.stack([b["emo_vec"] for b in batch]),
    }


def training_forward(
    model: Any,
    speaker_cond: nn.Parameter,
    batch: dict[str, torch.Tensor],
    text_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute dual CE loss (text + mel) for training."""
    text_inputs = batch["text_tokens"]
    text_lengths = batch["text_lengths"]
    mel_codes = batch["codes"]
    mel_codes_lengths = batch["codes_lengths"]
    emo_vec = batch["emo_vec"]
    B = text_inputs.shape[0]

    speech_cond = speaker_cond.unsqueeze(0).expand(B, -1, -1)

    text_inputs = model.set_text_padding(text_inputs.clone(), text_lengths)
    text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)

    mel_codes = model.set_mel_padding(mel_codes.clone(), mel_codes_lengths)
    mel_codes = F.pad(mel_codes, (0, 1), value=model.stop_mel_token)

    use_speed = torch.zeros(B, dtype=torch.long, device=text_inputs.device)
    duration_emb = model.speed_emb(use_speed)
    duration_emb_half = model.speed_emb(torch.ones_like(use_speed))

    conds = torch.cat(
        (speech_cond + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)),
        dim=1,
    )

    text_inputs_aligned, text_targets = model.build_aligned_inputs_and_targets(
        text_inputs, model.start_text_token, model.stop_text_token
    )
    text_emb = model.text_embedding(text_inputs_aligned) + model.text_pos_embedding(text_inputs_aligned)

    mel_codes_aligned, mel_targets = model.build_aligned_inputs_and_targets(
        mel_codes, model.start_mel_token, model.stop_mel_token
    )
    mel_emb = model.mel_embedding(mel_codes_aligned) + model.mel_pos_embedding(mel_codes_aligned)

    text_logits, mel_logits = model.get_logits(
        conds, text_emb, model.text_head, mel_emb, model.mel_head,
        get_attns=False, return_latent=False,
    )

    text_loss = F.cross_entropy(text_logits, text_targets, ignore_index=model.stop_text_token)
    mel_loss = F.cross_entropy(mel_logits, mel_targets, ignore_index=model.stop_mel_token)
    loss = text_weight * text_loss + (1.0 - text_weight) * mel_loss

    return loss, {
        "loss": loss.item(),
        "text_loss": text_loss.item(),
        "mel_loss": mel_loss.item(),
    }


def create_loraplus_optimizer(
    model: nn.Module,
    speaker_cond: nn.Parameter,
    lr: float,
    loraplus_lr_ratio: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """LoRA+ optimizer: B-matrix gets higher LR."""
    lora_a_params: list[nn.Parameter] = []
    lora_b_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            lora_a_params.append(param)
        elif "lora_B" in name:
            lora_b_params.append(param)
        else:
            other_params.append(param)

    param_groups: list[dict] = [
        {"params": lora_a_params, "lr": lr, "weight_decay": weight_decay},
        {"params": lora_b_params, "lr": lr * loraplus_lr_ratio, "weight_decay": weight_decay},
    ]
    if other_params:
        param_groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
    param_groups.append({"params": [speaker_cond], "lr": lr, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups)


def main() -> None:
    args = parse_args()

    indextts_dir = Path(args.indextts_dir).resolve()
    sys.path.insert(0, str(indextts_dir))

    from omegaconf import OmegaConf
    from peft import LoraConfig, get_peft_model
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else indextts_dir / "checkpoints"
    cfg_path = ckpt_dir / "config.yaml"
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else indextts_dir / "results" / "lora"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(">> Loading config & model...")
    cfg = OmegaConf.load(str(cfg_path))
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, str(ckpt_dir / cfg.gpt_checkpoint))
    gpt = gpt.to(device)
    gpt.train()

    for param in gpt.parameters():
        param.requires_grad = False

    # Monkey-patch: IndexTTS deletes gpt.wte but PEFT needs it
    gpt.gpt.wte = nn.Embedding(1, 1)
    gpt.gpt.config.gradient_checkpointing = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    )
    gpt.gpt = get_peft_model(gpt.gpt, lora_config)

    if hasattr(gpt.gpt.base_model.model, "wte"):
        gpt.gpt.base_model.model.wte.weight.requires_grad = False

    gpt.gpt.print_trainable_parameters()

    # Learnable speaker condition
    medoid_condition = np.load(str(data_dir / "medoid_condition.npy"))
    speaker_cond = nn.Parameter(
        torch.from_numpy(medoid_condition).float().to(device),
        requires_grad=True,
    )

    n_lora = sum(p.numel() for _, p in gpt.named_parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in gpt.parameters())
    n_trainable = n_lora + speaker_cond.numel()
    print(f">> Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    # Data
    train_ds = FinetuneDataset(data_dir / "metadata_train.jsonl")
    val_ds = FinetuneDataset(data_dir / "metadata_val.jsonl")
    print(f">> Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Optimizer & scheduler
    optimizer = create_loraplus_optimizer(
        gpt, speaker_cond, args.lr, args.lora_plus_ratio, args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_loss = float("inf")
    all_params = list(gpt.parameters()) + [speaker_cond]
    log_file = open(output_dir / "train.log", "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    try:
        log(f">> Config: r={args.lora_r}, alpha={args.lora_alpha}, lr={args.lr}, "
            f"epochs={args.epochs}, batch={args.batch_size}")
        log(f">> Total steps: {total_steps}, Warmup: {warmup_steps}")

        for epoch in range(1, args.epochs + 1):
            gpt.train()
            epoch_losses: list[float] = []
            epoch_mel_losses: list[float] = []
            t_start = time.time()

            for step, batch in enumerate(train_loader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, metrics = training_forward(gpt, speaker_cond, batch, args.text_weight)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                epoch_losses.append(metrics["loss"])
                epoch_mel_losses.append(metrics["mel_loss"])

                if step % 5 == 0 or step == len(train_loader):
                    lr_current = scheduler.get_last_lr()[0]
                    log(
                        f"  Epoch {epoch}/{args.epochs} Step {step}/{len(train_loader)} | "
                        f"loss={metrics['loss']:.4f} mel={metrics['mel_loss']:.4f} "
                        f"text={metrics['text_loss']:.4f} lr={lr_current:.2e}"
                    )

            train_loss = np.mean(epoch_losses)
            train_mel = np.mean(epoch_mel_losses)
            elapsed = time.time() - t_start

            # Validate
            gpt.eval()
            val_losses: list[float] = []
            val_mel_losses: list[float] = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    _, metrics = training_forward(gpt, speaker_cond, batch, args.text_weight)
                    val_losses.append(metrics["loss"])
                    val_mel_losses.append(metrics["mel_loss"])

            val_loss = np.mean(val_losses) if val_losses else float("inf")
            val_mel = np.mean(val_mel_losses) if val_mel_losses else float("inf")

            log(
                f">> Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_mel={train_mel:.4f} | "
                f"val_loss={val_loss:.4f} val_mel={val_mel:.4f} | "
                f"time={elapsed:.1f}s"
            )

            ckpt = {
                "epoch": epoch,
                "lora_state_dict": {
                    k: v.cpu() for k, v in gpt.gpt.state_dict().items()
                    if "lora_" in k
                },
                "speaker_cond": speaker_cond.detach().cpu(),
                "val_loss": val_loss,
                "val_mel_loss": val_mel,
            }
            torch.save(ckpt, output_dir / f"lora_epoch_{epoch:02d}.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, output_dir / "lora_best.pt")
                log(f"  ** New best model (val_loss={val_loss:.4f})")

        log(f"\n>> Training complete. Best val_loss: {best_val_loss:.4f}")
        log(f">> Checkpoints saved to: {output_dir}")
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
