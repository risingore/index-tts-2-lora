"""IndexTTS-2 LoRA: Extract precomputed features for fine-tuning.

For each WAV clip:
  1. w2v-bert features -> semantic_codec.quantize() -> mel codes
  2. gpt.get_conditioning() -> speaker conditioning latent (32, 1280)
  3. gpt.get_emo_conditioning() + linear -> emotion vector (1280,)
  4. BPE tokenization -> text token IDs

Outputs:
  - Per-clip .npz files (codes, condition, emo_vec, text_tokens)
  - medoid_condition.npy (representative speaker vector)
  - metadata_train.jsonl / metadata_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """Load TSV manifest -> {filename_stem: transcription}."""
    mapping: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                stem = Path(parts[0]).stem
                mapping[stem] = parts[1]
    return mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract precomputed features for IndexTTS-2 LoRA fine-tuning",
    )
    p.add_argument("--indextts-dir", type=str, required=True,
                    help="Path to IndexTTS-2 installation (contains indextts/ package)")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Path to model checkpoints (default: {indextts-dir}/checkpoints)")
    p.add_argument("--audio-dir", type=str, required=True,
                    help="Directory containing WAV files for fine-tuning")
    p.add_argument("--manifest", type=str, required=True,
                    help="TSV manifest file: filename<TAB>transcription")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for .npz features (default: {indextts-dir}/data/ft_data)")
    p.add_argument("--val-ratio", type=float, default=0.1,
                    help="Fraction of data for validation (default: 0.1)")
    p.add_argument("--min-duration", type=float, default=1.0,
                    help="Minimum clip duration in seconds (default: 1.0)")
    p.add_argument("--max-duration", type=float, default=20.0,
                    help="Maximum clip duration in seconds (default: 20.0)")
    p.add_argument("--device", type=str, default=None,
                    help="Device to use (default: cuda if available, else cpu)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    indextts_dir = Path(args.indextts_dir).resolve()
    sys.path.insert(0, str(indextts_dir))

    # Lazy imports after sys.path setup
    from omegaconf import OmegaConf
    from transformers import SeamlessM4TFeatureExtractor
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
    from indextts.utils.checkpoint import load_checkpoint
    from indextts.utils.front import TextNormalizer, TextTokenizer
    import safetensors.torch

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else indextts_dir / "checkpoints"
    cfg_path = ckpt_dir / "config.yaml"
    audio_dir = Path(args.audio_dir)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else indextts_dir / "data" / "ft_data"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f">> Config: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(">> Loading semantic model (w2v-bert-2.0)...")
    extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(
        str(ckpt_dir / cfg.w2v_stat)
    )
    semantic_model = semantic_model.to(device)
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)

    print(">> Loading semantic codec (MaskGCT RepCodec)...")
    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    from huggingface_hub import hf_hub_download
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device)
    semantic_codec.eval()

    print(">> Loading GPT model...")
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, str(ckpt_dir / cfg.gpt_checkpoint))
    gpt = gpt.to(device)
    gpt.eval()

    print(">> Loading BPE tokenizer...")
    bpe_path = str(ckpt_dir / cfg.dataset["bpe_model"])
    normalizer = TextNormalizer(enable_glossary=True)
    normalizer.load()
    tokenizer = TextTokenizer(bpe_path, normalizer)

    # Load manifest & collect WAV files
    manifest = load_manifest(manifest_path)
    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f">> Found {len(wav_files)} WAV files in {audio_dir}")

    # Process each clip
    all_conditions: list[np.ndarray] = []
    all_entries: list[dict] = []
    skipped = 0
    resamplers: dict[int, torchaudio.transforms.Resample] = {}

    for wav_path in tqdm(wav_files, desc="Extracting"):
        stem = wav_path.stem
        if stem not in manifest:
            print(f"  SKIP (no transcript): {stem}")
            skipped += 1
            continue

        text = manifest[stem]
        audio, sr = torchaudio.load(str(wav_path))
        duration = audio.shape[1] / sr

        if duration < args.min_duration or duration > args.max_duration:
            print(f"  SKIP (duration {duration:.1f}s): {stem}")
            skipped += 1
            continue

        if sr != 16000:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
            audio_16k = resamplers[sr](audio)
        else:
            audio_16k = audio

        with torch.no_grad():
            inputs = extract_features(
                audio_16k.squeeze(0).numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            )
            input_features = inputs["input_features"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            vq_emb = semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = vq_emb.hidden_states[17]
            feat = (feat - semantic_mean) / semantic_std

            codes, _ = semantic_codec.quantize(feat)
            codes_np = codes.squeeze(0).cpu().numpy().astype(np.int16)

            feat_dt = feat.transpose(1, 2)
            cond_lengths = torch.tensor([feat_dt.shape[-1]], device=device)
            condition = gpt.get_conditioning(feat_dt, cond_lengths)
            condition_np = condition.squeeze(0).cpu().float().numpy()

            emo_cond = gpt.get_emo_conditioning(feat_dt, cond_lengths)
            emo_vec = gpt.emovec_layer(emo_cond)
            emo_vec = gpt.emo_layer(emo_vec)
            emo_vec_np = emo_vec.squeeze(0).cpu().float().numpy()

        text_tokens_list = tokenizer.tokenize(text)
        text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens_list)
        text_tokens_np = np.array(text_token_ids, dtype=np.int32)

        out_path = output_dir / f"{stem}.npz"
        np.savez(
            str(out_path),
            codes=codes_np,
            condition=condition_np,
            emo_vec=emo_vec_np,
            text_tokens=text_tokens_np,
        )

        all_conditions.append(condition_np)
        all_entries.append({
            "stem": stem,
            "audio": str(wav_path),
            "text": text,
            "npz": str(out_path),
            "duration": round(duration, 4),
            "n_codes": len(codes_np),
            "n_text_tokens": len(text_tokens_np),
        })

    print(f"\n>> Processed: {len(all_entries)}, Skipped: {skipped}")

    if not all_entries:
        print(">> ERROR: No clips processed. Check manifest and audio directory.")
        sys.exit(1)

    # Compute medoid condition
    print(">> Computing medoid condition...")
    cond_stack = np.stack(all_conditions)
    cond_flat = cond_stack.reshape(len(cond_stack), -1)
    # Compute pairwise distances row-by-row to avoid O(N^2) memory spike
    dist_sums = np.array([np.linalg.norm(cond_flat - c, axis=1).sum() for c in cond_flat])
    medoid_idx = int(np.argmin(dist_sums))
    medoid_condition = cond_stack[medoid_idx]
    np.save(str(output_dir / "medoid_condition.npy"), medoid_condition)
    print(f"  Medoid clip: {all_entries[medoid_idx]['stem']} (index {medoid_idx})")

    # Train/val split
    n_val = max(1, int(len(all_entries) * args.val_ratio))
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_entries))
    val_indices = set(indices[:n_val])

    train_entries = [e for i, e in enumerate(all_entries) if i not in val_indices]
    val_entries = [e for i, e in enumerate(all_entries) if i in val_indices]

    with open(output_dir / "metadata_train.jsonl", "w", encoding="utf-8") as f:
        for e in train_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(output_dir / "metadata_val.jsonl", "w", encoding="utf-8") as f:
        for e in val_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f">> Train: {len(train_entries)}, Val: {len(val_entries)}")
    print(f">> Output: {output_dir}")
    print(">> Done.")


if __name__ == "__main__":
    main()
