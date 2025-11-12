# PegTransfer Video Classification

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
uv run preprocess_videos.py \
  --video_dir <path-to-video-directory> \
  --output_dir <path-to-frames-directory>
```

## Training

All training commands use `uv run src/run_train.py` with configuration flags or fall back to the default values in `src/config.py` if not specified.

**X3D**:
```bash
uv run -m src/run_train \
  --backbone x3d \
  --pooling max \
  --epochs 100 \
  --lr 5e-5 \
  --num_frames 16 \
  --sampling_rate 5 \
  --num_workers 8 \
  --seed 42 \
  --annotations <annotations-csv> \
  --frames_dir <frames-directory> \
  --output_dir <output-directory>
```

**MetaSpikeFormer**:
```bash
uv run -m src.run_train \
  --backbone spikeformer \
  --pooling max \
  --epochs 100 \
  --lr 5e-5 \
  --num_frames 16 \
  --sampling_rate 5 \
  --seed 42 \
  --annotations <annotations-csv> \
  --frames_dir <frames-directory> \
  --output_dir <output-directory>
```
- same as X3D, but with `--backbone spikeformer` instead of `--backbone x3d`.

## Available Checkpoints
| Checkpoint | Backbone | Pooling | Normalized MCC | Accuracy |
|------------|----------|----------|----------------|----------|
| [x3d-peg-transfer-error-detection.pth.tar](checkpoints/x3d-peg-transfer-error-detection.pth.tar) | x3d | max | 0.9836478481397317 | 0.9687494307761135 |

An example for how to load a checkpoint is provided in `load_checkpoint.py`.


## Note on VRAM requirements
The current training setup processes one video per batch, fitting as many clips from that video into GPU memory (VRAM) as possible. Each batch contains all possible clips from a single video to achieve the desired video coverage, which requires significant VRAM - currently, training runs on an A100 80GB GPU. Longer clips will require even more memory.

To support training on GPUs with less VRAM, I am now porting Isabel's approach: multi-clip training with gradient accumulation. In this approach, gradients are accumulated over several smaller mini-batches of clips from a single video, which reduces per-batch memory requirements and enables training with smaller GPUs. 