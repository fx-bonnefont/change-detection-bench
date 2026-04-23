# cd-bench

Semantic Change Detection benchmark on HI-UCD satellite imagery. Frozen encoder (DINOv3, ConvNeXt) + trainable decoder head. All experiments tracked in MLflow.

## Quick start

```bash
# 1. Clone and install
git clone <repo-url> && cd cd-bench
uv sync

# 2. Configure paths
cp configs/settings.toml.example configs/settings.toml
# Edit configs/settings.toml: set raw_data_dir and features_dir to your HI-UCD location

# 3. Start MLflow
docker compose up -d
# UI available at http://localhost:5001

# 4. Extract features (one-time, ~15 min)
cdbench extract -e dinov3-small

# 5. Tune hyperparameters (optional but recommended, ~1h for 30 trials)
cdbench tune -e dinov3-small -d baseline-conv

# 6. Train
cdbench train -e dinov3-small -d baseline-conv

# 7. Visualize predictions
cdbench show -r <run_id> --random 10
```

## Commands

| Command | Description |
|---------|-------------|
| `cdbench extract` | Extract encoder features to disk (memmap). Run once per encoder. |
| `cdbench tune` | Hyperparameter search via Optuna (loss weights, LR). Persistent SQLite DB, resumable. |
| `cdbench train` | Train a decoder head. Loads tuned hparams from store. Evaluates on test at the end. |
| `cdbench eval` | Evaluate a checkpoint on val or test without training. |
| `cdbench show` | Visualize SCD predictions (semantic maps + change contours) with HI-UCD palette. |
| `cdbench eda` | Exploratory stats on HI-UCD masks (class distributions). |
| `cdbench bench encoder` | Benchmark encoder forward pass speed. |
| `cdbench bench decoder` | Benchmark decoder forward+backward speed. |
| `cdbench search` | Search HuggingFace model hub for encoder IDs. |
| `cdbench mlflow-reset` | Purge MLflow DB, artifacts, and logs. |

Run any command with `--help` for full options.

## Typical workflow

```
extract ──> tune ──> train ──> show
                       │
                       └──> eval (on test, with a specific checkpoint)
```

1. **Extract** features once per encoder. Stored as memory-mapped files for fast loading.
2. **Tune** loss hyperparameters with Optuna. Results saved to `configs/loss_hparams.json`. Ctrl+C safe (trials persisted in SQLite).
3. **Train** with tuned hyperparameters (loaded automatically). Checkpoints + metrics logged to MLflow. Test set evaluated at the end of training.
4. **Show** predictions on test images. Use `--run-id` to load a checkpoint directly from MLflow.

## Configuration

All paths are centralized in `configs/settings.toml`:

```toml
[data]
raw_data_dir = "/path/to/HI-UCD"          # Contains train/, val/, test/
features_dir = "/path/to/extracted-features"

[mlflow]
tracking_uri = "http://localhost:5001"
db_path = "mlflow-db/mlflow.db"
artifacts_dir = "mlflow-artifacts"
```

Tuned loss hyperparameters are stored in `configs/loss_hparams.json` (auto-updated by `cdbench tune`).

## Architecture

**Pipeline**: frozen encoder extracts features offline, trainable decoder predicts 2K channels (K semantic classes per date). Change = where argmax(T1) differs from argmax(T2).

**Loss**: `CE_t1 + CE_t2 + lambda_bcd * Focal_change`
- CE for semantic segmentation at each date
- Focal auxiliary term on differentiable change probability to prevent the "same map" shortcut

**Encoders**: DINOv3 (small, base), ConvNeXt (base). Registered in `src/cd_bench/models/encoders/`.

**Decoders**: BaselineConv (conv + upsampling), QueryDecoder (transformer cross-attention). Registered in `src/cd_bench/models/decoders/`.

**Dataset**: HI-UCD 10 raw classes remapped to 7 (0=unlabeled/ignored + 6 useful classes). Classes <1% representation merged into unlabeled.

## Hardware

CUDA, MPS (Apple Silicon), and CPU are auto-detected. No configuration needed.

```
CUDA  -> torch.device("cuda")   # Linux/Windows with NVIDIA GPU
MPS   -> torch.device("mps")    # macOS with Apple Silicon
CPU   -> torch.device("cpu")    # Fallback
```

## Project structure

```
cd-bench/
├── configs/
│   ├── settings.toml.example   # Template, copy to settings.toml
│   ├── settings.toml           # Local config (gitignored)
│   └── loss_hparams.json       # Tuned hyperparameters (auto-managed)
├── src/cd_bench/
│   ├── cli/                    # Typer commands (extract, train, tune, eval, show, ...)
│   ├── data/                   # Datasets, splits, mask remapping
│   ├── models/
│   │   ├── encoders/           # Frozen feature extractors
│   │   └── decoders/           # Trainable decoder heads
│   ├── training/               # Trainer, losses (SCDLoss), hparams store
│   ├── inference/              # Prediction + visualization
│   └── utils/                  # Device detection, I/O
├── docker-compose.yml          # MLflow server
└── pyproject.toml
```

## Requirements

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv)
- Docker & Docker Compose (for MLflow)
