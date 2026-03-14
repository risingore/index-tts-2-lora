# IndexTTS-2 LoRA Fine-Tuning

LoRA fine-tuning pipeline for [IndexTTS-2](https://github.com/index-tts/index-tts) voice cloning. Adapt the GPT model to a target speaker's voice with as few as ~50 audio clips.

**This is the world's first LoRA fine-tuning implementation for IndexTTS-2.**

## How It Works

IndexTTS-2 uses a GPT-2 backbone (`UnifiedVoice`) to predict semantic mel codes from text tokens, conditioned on speaker embeddings. This pipeline applies LoRA to all attention and MLP projections in the transformer:

- **LoRA targets**: `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj` (4 modules × 24 layers = 96 adapted weight matrices)
- **LoRA+**: B-matrix gets 8× the base learning rate for faster convergence
- **Dual CE loss**: `0.1 × CE(text) + 0.9 × CE(mel)` — mel prediction is the primary objective
- **Learnable speaker condition**: Initialized from the medoid of all training clips' speaker embeddings

## Requirements

- Python 3.10+
- CUDA GPU (8GB+ VRAM recommended)
- [IndexTTS-2](https://github.com/index-tts/index-tts) installed with checkpoints downloaded

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Prepare a manifest

Create a TSV file mapping audio filenames to transcriptions:

```
clip_001.wav	Hello, how are you today?
clip_002.wav	The weather is nice outside.
```

### Step 2: Extract features

```bash
python extract_codec.py \
  --indextts-dir /path/to/index-tts \
  --audio-dir /path/to/wav/clips \
  --manifest /path/to/manifest.tsv \
  --output-dir /path/to/features
```

This extracts semantic codes, speaker conditioning latents, emotion vectors, and text tokens for each clip.

### Step 3: Train LoRA

```bash
python train.py \
  --indextts-dir /path/to/index-tts \
  --data-dir /path/to/features \
  --output-dir /path/to/lora_output \
  --epochs 15 \
  --batch-size 4
```

Training saves checkpoints every epoch and tracks the best model by validation loss.

### Step 4: Merge & test

```bash
python merge.py \
  --indextts-dir /path/to/index-tts \
  --lora-dir /path/to/lora_output \
  --ref-audio /path/to/reference.wav \
  --test-text "Hello, this is a test."
```

The merged model replaces the base GPT checkpoint for inference.

## Hyperparameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA scaling factor |
| `--lora-dropout` | 0.1 | LoRA dropout |
| `--lr` | 5e-5 | Base learning rate |
| `--lora-plus-ratio` | 8.0 | B-matrix LR multiplier |
| `--epochs` | 15 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--warmup-ratio` | 0.1 | Warmup fraction |
| `--text-weight` | 0.1 | Text loss weight (mel = 1 - text_weight) |
| `--val-ratio` | 0.1 | Validation split ratio |
| `--min-duration` | 1.0s | Minimum clip duration |
| `--max-duration` | 20.0s | Maximum clip duration |

## Security Note

This tool uses `torch.load(weights_only=False)` to load checkpoints. Only use checkpoint files (`.pt`, `.pth`) from sources you trust, as malicious files could execute arbitrary code during loading.

## Tips

- **Data quality matters more than quantity.** 50-100 clean clips (1-20s each) with accurate transcriptions work well.
- **Early stopping**: Monitor `val_loss` — the best checkpoint is often around epoch 3-5 for small datasets.
- **VRAM**: With batch_size=4 and fp32, expect ~6-7GB VRAM usage on a single GPU.
- **Works with community checkpoints** like [IndexTTS-2-Japanese](https://huggingface.co/Jmica/IndexTTS-2-Japanese).

## Architecture Details

```
IndexTTS-2 GPT (UnifiedVoice)
├── text_embedding + text_pos_embedding
├── mel_embedding + mel_pos_embedding
├── gpt (GPT2Model, 24 layers, 1280 dim, 20 heads)  ← LoRA applied here
│   └── Each layer:
│       ├── attn.c_attn  (3840 → 1280, QKV projection)  ← LoRA
│       ├── attn.c_proj  (1280 → 1280)                   ← LoRA
│       ├── mlp.c_fc     (1280 → 5120)                   ← LoRA
│       └── mlp.c_proj   (5120 → 1280)                   ← LoRA
├── text_head (logits for text tokens)
├── mel_head (logits for mel codes)
├── conditioning_encoder (ConformerEncoder + PerceiverResampler)
└── emo_conditioning_encoder
```

## License

MIT

## 日本語

IndexTTS-2 の GPT モデルに LoRA ファインチューニングを適用し、少量の音声データ（50-100クリップ程度）でターゲット話者の声質を学習させるパイプラインです。

### 使い方

1. **特徴量抽出**: `extract_codec.py` — WAVファイルからセマンティックコード・話者条件ベクトル・感情ベクトル・テキストトークンを事前計算
2. **LoRA学習**: `train.py` — GPT-2 の全 Attention/MLP 層に LoRA を適用し、デュアル CE ロスで学習
3. **マージ**: `merge.py` — LoRA 重みをベースモデルに統合し、推論用モデルを生成

詳細な引数は各スクリプトの `--help` を参照してください。
