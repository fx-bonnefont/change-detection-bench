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


def _resolve_checkpoint(checkpoint: str | None, run_id: str | None) -> Path:
    if checkpoint is not None:
        p = Path(checkpoint)
        if not p.exists():
            raise typer.BadParameter(f"Checkpoint introuvable : {p}")
        return p
    if run_id is not None:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, "checkpoints/best.pt")
        return Path(local_path)
    raise typer.BadParameter("Il faut fournir --checkpoint ou --run-id.")


def show(
    encoder: str = typer.Option("dinov3-small", "--encoder", "-e"),
    decoder: str = typer.Option("baseline-conv", "--decoder", "-d"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Chemin local vers best.pt."),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="MLflow run_id (télécharge checkpoints/best.pt)."),
    start: Optional[int] = typer.Option(None, "--start", "-s", help="Index de début dans le split test sorted."),
    end: Optional[int] = typer.Option(None, "--end", help="Index de fin (exclu) dans le split test sorted."),
    random_n: Optional[int] = typer.Option(None, "--random", "-n", help="Nombre de paires aléatoires à visualiser."),
    count: int = typer.Option(5, "--count", "-k", help="Nombre max de paires à afficher (avec --start/--end)."),
    min_pred_change: float = typer.Option(
        0.0, "--min-pred-change", "-m",
        help="Taux minimum de pixels prédits 'changed' pour afficher une paire (ex: 0.01 = 1%%).",
    ),
    save_dir: Optional[str] = typer.Option(None, "--save-dir", "-o", help="Si fourni, sauvegarde les figures au lieu de les afficher."),
    img_size: int = typer.Option(512, "--img-size", help="Résolution des images HI-UCD."),
):
    """Visualise les prédictions SCD sur le split test HI-UCD (sans masques)."""
    has_range = start is not None or end is not None
    if not has_range and random_n is None:
        raise typer.BadParameter("Il faut fournir --start/--end ou --random.")
    if has_range and random_n is not None:
        raise typer.BadParameter("--start/--end et --random sont mutuellement exclusifs.")

    paths = get_paths(RAW_DATA_DIR, limit=None)
    test_paths = paths.get("test")
    if test_paths is None or not test_paths.get("2018"):
        raise typer.BadParameter("Pas de split test trouvé dans HI-UCD.")

    paths_2018 = test_paths["2018"]
    paths_2019 = test_paths["2019"]
    n_total = len(paths_2018)
    typer.echo(f"Split test HI-UCD : {n_total} paires disponibles.")

    ckpt_path = _resolve_checkpoint(checkpoint, run_id)
    typer.echo(f"Chargement du modèle {encoder} + {decoder} depuis {ckpt_path}...")
    model = load_model(encoder, decoder, ckpt_path, img_size=img_size)
    typer.echo("Modèle chargé.")

    if has_range:
        s = start if start is not None else 0
        e = end if end is not None else n_total
        if s < 0 or e > n_total or s >= e:
            raise typer.BadParameter(f"Bornes invalides : [{s}, {e}) sur {n_total} tiles.")
        candidates = list(range(s, e))
        target_n = count
    else:
        assert random_n is not None
        candidates = list(range(n_total))
        random.shuffle(candidates)
        target_n = random_n

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

        if change_pct < min_pred_change:
            typer.echo(f"  [{i}] {tile_id} — {change_pct * 100:.1f}% change (< {min_pred_change * 100:.1f}%), skipped")
            continue

        typer.echo(f"  [{i}] {tile_id} — {change_pct * 100:.1f}% change")
        fig = make_figure(img_2018, img_2019, sem_t1_np, sem_t2_np, change_np, tile_id)
        _output_figure(fig, tile_id, save_dir)
        collected.append(i)

        if len(collected) >= target_n:
            break

    typer.echo(f"Scanned {scanned}/{len(candidates)} tiles, kept {len(collected)}/{target_n}.")
    if len(collected) < target_n:
        typer.echo(
            f"Pas assez de tiles avec >= {min_pred_change * 100:.1f}% de changement prédit. "
            f"Essaie --min-pred-change plus bas ou une plage plus large."
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
