# change-detection-bench

A workbench for experimenting with state-of-the-art semantic change detection in remote sensing imagery.

The repository is structured to facilitate rapid experimentation and to keep a traceable history of every run via an MLflow instance (Docker).

## Key features

- **Feature extraction** : A feature extractor computes image embeddings through a frozen DINOv3 backbone and persists the resulting tensors to disk. This avoids redundant forward passes across experiments and dramatically speeds up training.
- **Feature dataset** : A PyTorch dataset that loads the pre-computed embeddings produced by the feature extractor, ready for downstream model training.
- **Baseline model** : A reference model that computes cosine similarity between either the CLS tokens or the per-patch embeddings of image pairs. On the small, small-plus and base DINOv3 ViT variants, the patch-level approach yields ~10 % higher F1-score than the CLS-only variant.
- **Plug-and-play model architecture** : Adding a new model is as simple as dropping a `your_model.py` file into `src/models/` and calling it or multiple other models implemented in the run.py file of the scripts/ folder.
- **Quick set-up** : Just drop your personnal paths in the config.py file at the root of src.

## Requirements

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (recommended)
- Docker & Docker Compose (for MLflow)

## Getting started

```bash
# Clone and install
git clone <repo-url> && cd cd-bench
uv sync

# Start the MLflow tracking server
docker compose up -d

# Run an experiment
uv run python scripts/run.py
```

MLflow UI is then available at `http://localhost:5001`.

## Project structure

```
cd-bench/
├── scripts/            # Entry-point scripts (extraction, training, EDA)
├── src/
│   ├── data/           # Datasets and data paths
│   ├── models/         # Model definitions (baseline + yours)
│   ├── training/       # Trainer and metrics
│   └── utils/          # Device helpers, I/O
├── docker-compose.yml  # MLflow service
└── pyproject.toml
```
