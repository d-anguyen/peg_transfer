# PegTransfer Video Classification

Training ANN and SNN models for surgical video classification.

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Install [uv](https://github.com/astral-sh/uv) if you don't have it.
3. Install dependencies using uv:
   ```bash
   uv sync
   ```

_**Note:** When using `pyproject.toml`, `uv sync` will handle all dependencies. If you don't want to use uv, you can put them into a `requirements.txt` file and install them with `pip install -r requirements.txt`._

## Data Preparation

Before training, you need to extract frames from the video files. Use the `preprocess_videos.py` script to do so and save them to a directory. The training script expects pre-extracted frames rather than raw video files to avoid video decoding overhead. 

```bash
uv run src/preprocess_videos.py \
  --fps 2.0 \ # Frames per second to extract
  --quality 95 \ # JPEG quality
  --output_dir <path-to-output-directory> \
```

## Training

All training commands use `uv run src/train.py` with configuration flags.

**Common flags:**
- `--annotations` - Path to PegTransfer.csv
- `--frames_dir` - Directory with pre-extracted frames
- `--output_dir` - Output directory for models/results
- `--num_frames` - Number of frames to sample (default: 16)
- `--image_size` - Image size (default: 224)
- `--batch_size` - Batch size (default: 4)
- `--epochs` - Training epochs (default: 30)
- `--seed` - Random seed (default: 42)

**Note:** Remove `--use_wandb` flag to disable Weights & Biases logging.

### ANN with Top-K Sampling
```bash
uv run src/train.py \
  --model_type ann \
  --sampling_method topk \
  --topk_step 2 \
  --topk_metric l1 \
  --cache_motion \
  --output_dir outputs/ann_topk \
  --use_wandb
```

### SNN with Top-K Sampling
```bash
uv run src/train.py \
  --model_type snn \
  --sampling_method topk \
  --topk_step 2 \
  --topk_metric l1 \
  --cache_motion \
  --output_dir outputs/snn_topk \
  --use_wandb
```

### ANN with Uniform Sampling (No Random Offset)
```bash
uv run src/train.py \
  --model_type ann \
  --sampling_method uniform \
  --no_random_offset \
  --output_dir outputs/ann_no_offset \
  --use_wandb
```

### SNN with Uniform Sampling (No Random Offset)
```bash
uv run src/train.py \
  --model_type snn \
  --sampling_method uniform \
  --no_random_offset \
  --output_dir outputs/snn_no_offset \
  --use_wandb
```

### ANN with Uniform Sampling (Random Offset)
```bash
uv run src/train.py \
  --model_type ann \
  --sampling_method uniform \
  --output_dir outputs/ann_offset \
  --use_wandb
```

### SNN with Uniform Sampling (Random Offset)
```bash
uv run src/train.py \
  --model_type snn \
  --sampling_method uniform \
  --output_dir outputs/snn_offset \
  --use_wandb
```