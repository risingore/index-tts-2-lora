"""IndexTTS-2 LoRA: Merge LoRA weights into base model and generate test audio.

Usage:
  python merge.py --indextts-dir /path/to/index-tts --lora-dir /path/to/lora
  python merge.py --indextts-dir /path/to/index-tts --lora-dir /path/to/lora --epoch 5
  python merge.py --indextts-dir /path/to/index-tts --checkpoint /path/to/lora_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge LoRA weights into IndexTTS-2 base model",
    )
    # Paths
    p.add_argument("--indextts-dir", type=str, required=True,
                    help="Path to IndexTTS-2 installation")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Path to model checkpoints (default: {indextts-dir}/checkpoints)")
    p.add_argument("--lora-dir", type=str, default=None,
                    help="Directory containing LoRA checkpoints (output of train.py)")
    p.add_argument("--output", type=str, default=None,
                    help="Output path for merged model (default: {checkpoint-dir}/model_merged_lora.pth)")
    p.add_argument("--device", type=str, default=None,
                    help="Device (default: cuda if available)")

    # Checkpoint selection
    p.add_argument("--epoch", type=int, default=None,
                    help="Use checkpoint from specific epoch")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to specific checkpoint file")

    # LoRA config (must match training)
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    p.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")

    # Test audio generation
    p.add_argument("--no-generate", action="store_true",
                    help="Skip test audio generation")
    p.add_argument("--ref-audio", type=str, default=None,
                    help="Reference audio for test generation")
    p.add_argument("--test-output", type=str, default=None,
                    help="Output path for test audio (default: {output-dir}/test_output.wav)")
    p.add_argument("--test-text", type=str, default=None,
                    help="Text for test audio generation")
    return p.parse_args()


def merge_lora_weights(
    gpt: nn.Module,
    lora_state_dict: dict[str, torch.Tensor],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> nn.Module:
    """Apply LoRA config, load trained weights, merge, and return clean model."""
    from peft import LoraConfig, get_peft_model

    # Monkey-patch: IndexTTS deletes gpt.wte but PEFT needs it
    gpt.gpt.wte = nn.Embedding(1, 1)
    gpt.gpt.config.gradient_checkpointing = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    )
    gpt.gpt = get_peft_model(gpt.gpt, lora_config)

    missing, unexpected = gpt.gpt.load_state_dict(lora_state_dict, strict=False)
    lora_loaded = sum(1 for k in lora_state_dict if "lora_" in k)
    print(f">> Loaded {lora_loaded} LoRA parameters")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}...")

    gpt.gpt = gpt.gpt.merge_and_unload()
    print(">> LoRA merged into base model")
    return gpt


def generate_test_audio(
    indextts_dir: Path,
    ckpt_dir: Path,
    cfg_path: Path,
    merged_model_path: Path,
    ref_audio: str,
    test_text: str,
    output_path: str,
    device: str,
) -> None:
    """Generate test audio using the merged model via IndexTTS2 inference."""
    from omegaconf import OmegaConf
    from indextts.infer_v2 import IndexTTS2

    cfg = OmegaConf.load(str(cfg_path))
    cfg.gpt_checkpoint = merged_model_path.name
    tmp_cfg = ckpt_dir / "config_ft_temp.yaml"
    OmegaConf.save(cfg, str(tmp_cfg))

    print(">> Loading IndexTTS2 with merged model for inference...")
    tts = IndexTTS2(
        cfg_path=str(tmp_cfg),
        model_dir=str(ckpt_dir),
        use_fp16=False,
        device=device,
    )

    print(f">> Reference audio: {ref_audio}")
    print(f">> Test text: {test_text}")
    print(f">> Output: {output_path}")

    try:
        tts.infer(
            spk_audio_prompt=ref_audio,
            text=test_text,
            output_path=output_path,
        )
        print(f">> Test audio saved: {output_path}")
    finally:
        if tmp_cfg.exists():
            tmp_cfg.unlink()


def main() -> None:
    args = parse_args()

    indextts_dir = Path(args.indextts_dir).resolve()
    sys.path.insert(0, str(indextts_dir))

    from omegaconf import OmegaConf
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else indextts_dir / "checkpoints"
    cfg_path = ckpt_dir / "config.yaml"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Determine which checkpoint to use
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.lora_dir and args.epoch:
        ckpt_path = Path(args.lora_dir) / f"lora_epoch_{args.epoch:02d}.pt"
    elif args.lora_dir:
        ckpt_path = Path(args.lora_dir) / "lora_best.pt"
    else:
        print(">> ERROR: Specify --lora-dir or --checkpoint")
        sys.exit(1)

    if not ckpt_path.exists():
        print(f">> ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f">> Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Val loss: {ckpt.get('val_loss', '?'):.4f}")

    lora_state_dict = ckpt["lora_state_dict"]

    # Load base model
    print(">> Loading base GPT model...")
    cfg = OmegaConf.load(str(cfg_path))
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, str(ckpt_dir / cfg.gpt_checkpoint))
    gpt = gpt.to(device)

    # Merge LoRA
    gpt = merge_lora_weights(
        gpt, lora_state_dict, args.lora_r, args.lora_alpha, args.lora_dropout,
    )

    # Save merged model and verify against original
    output_path = Path(args.output) if args.output else ckpt_dir / "model_merged_lora.pth"
    original_state = torch.load(str(ckpt_dir / cfg.gpt_checkpoint), map_location="cpu", weights_only=False)
    merged_state = {k: v for k, v in gpt.state_dict().items() if "wte" not in k}
    diff_count = sum(
        1 for key in original_state
        if key in merged_state and not torch.equal(original_state[key], merged_state[key].cpu())
    )
    torch.save(merged_state, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f">> Saved merged model: {output_path} ({size_mb:.1f} MB)")
    print(f">> Verification: {diff_count} parameters differ from original model")

    if diff_count == 0:
        print(">> WARNING: No parameter differences found! LoRA may not have been applied correctly.")
    else:
        print(">> Merge verified successfully.")

    # Generate test audio
    if not args.no_generate:
        if not args.ref_audio:
            print(">> Skipping test audio: --ref-audio not specified")
        else:
            test_text = args.test_text or "Hello, this is a test of the fine-tuned voice model."
            test_output = args.test_output or str(output_path.parent / "test_output.wav")
            generate_test_audio(
                indextts_dir, ckpt_dir, cfg_path,
                output_path, args.ref_audio, test_text, test_output, device,
            )

    print(">> Done.")


if __name__ == "__main__":
    main()
