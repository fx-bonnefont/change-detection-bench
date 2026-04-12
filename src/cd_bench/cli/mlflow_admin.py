"""Sous-commande ``cdbench mlflow-reset`` : purge complète du tracking MLflow.

Détruit la SQLite backend store + tous les artefacts + les logs locaux,
puis redémarre le container.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from cd_bench.config import MLFLOW_ARTIFACTS_DIR, MLFLOW_DB_PATH

LOGS_DIR = Path("logs")


def mlflow_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip la confirmation interactive."),
):
    typer.echo("Cette action va DÉTRUIRE :")
    typer.echo(f"  - {MLFLOW_DB_PATH}  (toute la base SQLite)")
    typer.echo(f"  - {MLFLOW_ARTIFACTS_DIR}/*  (tous les checkpoints/artefacts)")
    typer.echo(f"  - {LOGS_DIR}/*  (logs d'entraînement locaux)")
    if not yes and not typer.confirm("Continuer ?"):
        typer.echo("Annulé.")
        raise typer.Exit(code=1)

    typer.echo("→ docker compose down")
    subprocess.run(["docker", "compose", "down"], check=True)

    if MLFLOW_DB_PATH.exists():
        MLFLOW_DB_PATH.unlink()
        typer.echo(f"→ supprimé {MLFLOW_DB_PATH}")

    for d in (MLFLOW_ARTIFACTS_DIR, LOGS_DIR):
        if d.exists():
            subprocess.run(f"rm -rf {d}/* {d}/.[!.]*", shell=True, check=False)
            typer.echo(f"→ vidé {d}")

    typer.echo("→ docker compose up -d")
    subprocess.run(["docker", "compose", "up", "-d"], check=True)
    typer.echo("MLflow réinitialisé.")
