"""Sous-commande ``cdbench search`` : recherche d'IDs HuggingFace.

Pratique pour trouver l'``hf_id`` exact à coller dans le registry des
encoders (``cd_bench/models/encoders/__init__.py``) sans quitter le
terminal.
"""
from __future__ import annotations

import requests
import typer

HF_API = "https://huggingface.co/api/models"


def search(
    query: str = typer.Argument(..., help="Sous-chaîne à matcher dans les IDs HuggingFace (ex: 'dinov3-vith')."),
    limit: int = typer.Option(10, "--limit", "-n", help="Nombre max de résultats."),
    library: str = typer.Option("transformers", "--library", "-l", help="Filtre par lib (transformers, timm, …)."),
    sort: str = typer.Option("downloads", "--sort", "-s", help="downloads | likes | lastModified | trendingScore."),
):
    r = requests.get(
        HF_API,
        params={"search": query, "library": library, "sort": sort, "limit": limit},
        timeout=10,
    )
    r.raise_for_status()
    results = r.json()
    if not results:
        typer.echo(f"Aucun modèle ne matche '{query}' (library={library}).")
        raise typer.Exit(code=1)

    for m in results:
        downloads = m.get("downloads", 0)
        typer.echo(f"{m['id']:<70}  ↓{downloads:>10}")
