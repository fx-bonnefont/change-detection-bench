"""Sous-commande ``cdbench bench`` : micro-benchmarks encoder et decoder.

Mesure les temps de forward (et forward+backward pour le decoder) sur
le device courant, sans data loading. Utile pour diagnostiquer les
bottlenecks et projeter le coût d'un entraînement.
"""
from __future__ import annotations

import time

import torch
import typer

from cd_bench.config import metadata_path
from cd_bench.data.mask_mapping import N_CLASSES
from cd_bench.models.decoders import DECODERS, get_decoder
from cd_bench.models.encoders import ENCODERS, get_encoder
from cd_bench.training.losses import SCDLoss
from cd_bench.utils.device import get_device
from cd_bench.utils.io import load_metadata

bench_app = typer.Typer(name="bench", help="Micro-benchmarks encoder/decoder.", no_args_is_help=True)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@bench_app.command(name="encoder")
def bench_encoder(
    encoder: str = typer.Option("dinov3-small", "--encoder", "-e"),
    img_size: int = typer.Option(512, "--img-size"),
    batch_size: int = typer.Option(1, "--batch-size", "-b"),
    warmup: int = typer.Option(3, "--warmup"),
    repeats: int = typer.Option(10, "--repeats"),
):
    """Benchmark du forward encodeur (inference only)."""
    device = get_device()
    typer.echo(f"Device: {device}")
    typer.echo(f"Encoder: {encoder}")
    typer.echo(f"Image size: {img_size}x{img_size}, batch_size: {batch_size}")

    enc = get_encoder(encoder)
    enc.load(img_size)
    enc.model.to(device)
    enc.model.eval()

    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    typer.echo(f"Warmup ({warmup} iters)...")
    with torch.inference_mode():
        for _ in range(warmup):
            enc.forward(dummy)
            _sync(device)

    times = []
    with torch.inference_mode():
        for i in range(repeats):
            t0 = time.perf_counter()
            out = enc.forward(dummy)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)
            if i == 0:
                typer.echo(f"Output shape: {out.shape}")

    avg = sum(times) / len(times)
    per_image = avg / batch_size
    typer.echo(f"\nResults ({repeats} repeats):")
    typer.echo(f"  Batch ({batch_size} images): {avg:.1f} ms")
    typer.echo(f"  Per image:                   {per_image:.1f} ms")
    typer.echo(f"  Min: {min(times):.1f} ms, Max: {max(times):.1f} ms")

    n_train = 10000
    fwd_per_epoch = 2 * n_train * per_image / 1000
    typer.echo(f"\n  Projected encoder forward per epoch ({n_train} pairs): {fwd_per_epoch:.0f}s ({fwd_per_epoch/60:.1f} min)")


@bench_app.command(name="decoder")
def bench_decoder(
    encoder: str = typer.Option("dinov3-small", "--encoder", "-e"),
    decoder: str = typer.Option("baseline-conv", "--decoder", "-d"),
    batch_size: int = typer.Option(32, "--batch-size", "-b"),
    iters: int = typer.Option(20, "--iters"),
    warmup: int = typer.Option(3, "--warmup"),
    limit: int | None = typer.Option(None, "--limit"),
):
    """Benchmark du decoder : forward only + full training step (forward+backward+optim)."""
    device = get_device()
    typer.echo(f"device = {device}")

    meta = load_metadata(metadata_path(encoder, limit))
    d_model = meta["dim"]
    n_tokens = meta["n_tokens"]
    out_size = meta["img_size"]
    typer.echo(f"encoder={encoder}  d_model={d_model}  n_tokens={n_tokens}  out_size={out_size}")

    dec = get_decoder(decoder)(d_model=d_model, out_size=out_size).to(device)
    n_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    typer.echo(f"decoder={decoder}  trainable_params={n_params}")

    K = N_CLASSES + 1
    t1 = torch.randn(batch_size, n_tokens, d_model, device=device)
    t2 = torch.randn(batch_size, n_tokens, d_model, device=device)
    sem_t1 = torch.randint(0, K, (batch_size, out_size, out_size), device=device)
    sem_t2 = torch.randint(0, K, (batch_size, out_size, out_size), device=device)
    valid = torch.ones(batch_size, out_size, out_size, device=device)

    loss_fn = SCDLoss(n_classes=K)
    optimizer = torch.optim.AdamW(dec.parameters(), lr=1e-4)

    # Forward only
    dec.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            dec(t1, t2)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            y = dec(t1, t2)
        _sync(device)
        fwd_ms = (time.perf_counter() - t0) / iters * 1000

    # Full training step
    dec.train()
    for _ in range(warmup):
        optimizer.zero_grad()
        y = dec(t1, t2)
        loss = loss_fn(y, sem_t1, sem_t2, valid)
        loss.backward()
        optimizer.step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad()
        y = dec(t1, t2)
        loss = loss_fn(y, sem_t1, sem_t2, valid)
        loss.backward()
        optimizer.step()
    _sync(device)
    train_ms = (time.perf_counter() - t0) / iters * 1000

    typer.echo(f"\noutput shape           : {tuple(y.shape)}")
    typer.echo(f"forward only           : {fwd_ms:.1f} ms / batch")
    typer.echo(f"forward + backward + step: {train_ms:.1f} ms / batch")
    typer.echo(f"backward overhead      : {train_ms - fwd_ms:.1f} ms ({train_ms/fwd_ms:.2f}x forward)")

    n_batches = 420
    proj_fwd = fwd_ms / 1000 * n_batches
    proj_train = train_ms / 1000 * n_batches
    typer.echo(f"\nprojection epoch ({n_batches} batches) :")
    typer.echo(f"  forward only        : {proj_fwd:.1f}s (~{proj_fwd / 60:.1f} min)")
    typer.echo(f"  full train step     : {proj_train:.1f}s (~{proj_train / 60:.1f} min)")
