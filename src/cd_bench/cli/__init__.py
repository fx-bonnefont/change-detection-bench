"""CLI unifiée du projet cd-bench.

Point d'entrée Typer exposé par ``pyproject.toml`` :

    [project.scripts]
    cdbench = "cd_bench.cli:app"

Sous-commandes :
    cdbench extract  -> extraction des features (offline, par encoder)
    cdbench train    -> entraînement (mode gelé ou dégelé)
    cdbench eda      -> visualisations exploratoires
    cdbench search   -> recherche d'IDs HuggingFace
"""
from __future__ import annotations

import typer

from cd_bench.cli.bench import bench_app
from cd_bench.cli.eda import eda
from cd_bench.cli.extract import extract
from cd_bench.cli.mlflow_admin import mlflow_reset
from cd_bench.cli.search import search
from cd_bench.cli.eval import eval
from cd_bench.cli.show import show
from cd_bench.cli.train import train
from cd_bench.cli.tune import tune

app = typer.Typer(
    name="cdbench",
    help="Change-detection benchmark — pipeline d'extraction et d'entraînement.",
    no_args_is_help=True,
    add_completion=True,
)

app.add_typer(bench_app, name="bench")
app.command(name="extract", help="Extrait les features d'un encoder vers un memmap.")(extract)
app.command(name="train", help="Entraîne un décodeur (mode gelé ou dégelé).")(train)
app.command(name="tune", help="Recherche d'hyperparams de loss via Optuna.")(tune)
app.command(name="eval", help="Évalue un checkpoint sur val ou test (sans entraînement).")(eval)
app.command(name="show", help="Visualise les prédictions sur le split test HI-UCD.")(show)
app.command(name="eda", help="Visualise les statistiques exploratoires des masques.")(eda)
app.command(name="search", help="Recherche des IDs de modèles sur HuggingFace.")(search)
app.command(name="mlflow-reset", help="Purge la DB et les artefacts MLflow puis redémarre le container.")(mlflow_reset)


__all__ = ["app"]
