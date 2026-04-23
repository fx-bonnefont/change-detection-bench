"""Sous-commande ``cdbench show`` : visualisation des prédictions SCD sur le split test HI-UCD.

Charge un modèle (encoder + decoder) depuis un checkpoint, inférence sur des
paires d'images brutes du split test officiel HI-UCD (celui sans masques),
et produit une visualisation : images brutes + cartes sémantiques prédites.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import typer
from PIL import Image

from cd_bench.config import MLFLOW_TRACKING_URI, RAW_DATA_DIR
from cd_bench.data.paths import get_paths
from cd_bench.inference.predict import load_model, predict_pair
from cd_bench.inference.visualize import make_figure, save_figure


def _resolve_checkpoint(checkpoint: str | None, run_id: str | None, decoder: str = "baseline-conv") -> Path:
    if checkpoint is not None:
        p = Path(checkpoint)
        if not p.exists():
            raise typer.BadParameter(f"Checkpoint introuvable : {p}")
        return p

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    if run_id is not None:
        local_path = client.download_artifacts(run_id, "checkpoints/best.pt")
        return Path(local_path)

    # Cherche le run le plus récent avec un checkpoint dans l'expérience TRAIN
    exp = client.get_experiment_by_name(f"CD-BENCH_{decoder}")
    if exp is None:
        raise typer.BadParameter(f"Aucune expérience MLflow 'CD-BENCH_{decoder}'. Lance `cdbench train` d'abord.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise typer.BadParameter("Aucun run trouvé. Lance `cdbench train` d'abord.")
    typer.echo(f"Run MLflow le plus récent : {runs[0].info.run_id}")
    local_path = client.download_artifacts(runs[0].info.run_id, "checkpoints/best.pt")
    return Path(local_path)


def show(
    encoder: str = typer.Option("dinov3-small", "--encoder", "-e"),
    decoder: str = typer.Option("baseline-conv", "--decoder", "-d"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Chemin local vers best.pt."),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="MLflow run_id (télécharge checkpoints/best.pt)."),
    n: int = typer.Option(10, "--n", "-n", help="Nombre de paires à visualiser."),
    seed: int = typer.Option(24, "--seed", help="Seed pour le tirage aléatoire."),
    min_change: float = typer.Option(
        0.1, "--min-change", "-m",
        help="Taux minimum de pixels prédits 'changed' pour garder une paire (ex: 0.01 = 1%%).",
    ),
    save_dir: Optional[str] = typer.Option(None, "--save-dir", "-o", help="Si fourni, sauvegarde les figures au lieu de les afficher."),
    img_size: int = typer.Option(512, "--img-size", help="Résolution des images HI-UCD."),
):
    """Visualise les prédictions SCD sur le split test HI-UCD (sans masques)."""
    paths = get_paths(RAW_DATA_DIR, limit=None)
    test_paths = paths.get("test")
    if test_paths is None or not test_paths.get("2018"):
        raise typer.BadParameter("Pas de split test trouvé dans HI-UCD.")

    paths_2018 = test_paths["2018"]
    paths_2019 = test_paths["2019"]
    n_total = len(paths_2018)
    typer.echo(f"Split test HI-UCD : {n_total} paires disponibles.")

    ckpt_path = _resolve_checkpoint(checkpoint, run_id, decoder)
    typer.echo(f"Chargement du modèle {encoder} + {decoder} depuis {ckpt_path}...")
    model = load_model(encoder, decoder, ckpt_path, img_size=img_size)
    typer.echo("Modèle chargé.")

    rng = random.Random(seed)
    candidates = list(range(n_total))
    rng.shuffle(candidates)

    collected: list[int] = []
    scanned = 0

    for i in candidates:
        tile_id = paths_2018[i].stem
        img_2018 = Image.open(paths_2018[i]).convert("RGB")
        img_2019 = Image.open(paths_2019[i]).convert("RGB")
        sem_t1, sem_t2, change_mask = predict_pair(model, img_2018, img_2019)
        sem_t1_np = sem_t1.numpy()
        sem_t2_np = sem_t2.numpy()
        change_np = change_mask.numpy()
        change_pct = float(change_np.mean())
        scanned += 1

        if change_pct < min_change:
            typer.echo(f"  [{i}] {tile_id} — {change_pct * 100:.1f}% change (< {min_change * 100:.1f}%), skipped")
            continue

        typer.echo(f"  [{i}] {tile_id} — {change_pct * 100:.1f}% change")
        fig = make_figure(img_2018, img_2019, sem_t1_np, sem_t2_np, change_np, tile_id)
        _output_figure(fig, tile_id, save_dir)
        collected.append(i)

        if len(collected) >= n:
            break

    typer.echo(f"Scanned {scanned}/{n_total} tiles, kept {len(collected)}/{n}.")
    if len(collected) < n:
        typer.echo(
            f"Pas assez de tiles avec >= {min_change * 100:.1f}% de changement prédit. "
            f"Essaie --min-change plus bas."
        )
    typer.echo("Done.")


def _output_figure(fig, tile_id: str, save_dir: str | None) -> None:
    if save_dir is not None:
        out_path = Path(save_dir) / f"{tile_id}.png"
        save_figure(fig, out_path)
        typer.echo(f"  -> {out_path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
