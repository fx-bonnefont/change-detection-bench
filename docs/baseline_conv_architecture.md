# Baseline Conv Decoder — Architecture

Avec D=384 (DINOv3-small) et hidden=128.

```
Résolution   Couche                              Channels    Params

             ── FUSION ──
32×32        [p1, p2, |p1-p2|]  concat            3×384
32×32        Linear(1152→384) + GELU + LayerNorm  384         443k

             ── PROJECTION ──
32×32        Conv2d 3×3 + GELU                    384→128     442k

             ── UPBLOCK 1 ──
32→64        ConvTranspose2d 4×4 stride 2 + GELU  128→128     262k

             ── UPBLOCK 2 ──
64→128       ConvTranspose2d 4×4 stride 2 + GELU  128→64      131k

             ── HEAD ──
128×128      Conv2d 1×1                           64→14        0.9k
128→512      nearest ×4 (non apprenable)          14            —

                                                  Total     ~1.3M
```
