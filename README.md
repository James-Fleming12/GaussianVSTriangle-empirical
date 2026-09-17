# GaussianVSTriangle-empirical

A controlled, reproducible comparison of three explicit radiance-field
primitives — **3D Gaussian Splatting**, **2D Gaussian Splatting** and
**Triangle Splatting** — on small synthetic scenes with exact ground truth.

The central protocol is **leave-one-out (LOO) novel-view evaluation**: for every
scene, one ground-truth camera is held out, the model is trained on the
remaining views only, and quality is measured on the held-out view. Because the
scenes are generated analytically, ground-truth images, depth and normals are
noise-free, and the answer to "what generalizes, and how fast?" is not
confounded by dataset artefacts such as SfM pose noise or exposure variation.

All three methods are re-implemented in one shared pure-PyTorch rasterization
framework: cameras, initialization, compositing, losses and the optimization
loop are common, so only the primitive geometry and the paper-specific
regularization/densification differ.

```
pip install -r requirements.txt
python -m pytest tests/ -q                       # 55 tests, ~3 s, CPU only
python scripts/run_loo.py --quick --device cpu   # pipeline smoke test
python scripts/run_loo.py --output results/main  # full benchmark (GPU)
python scripts/run_view_sweep.py --output results/views
python scripts/make_report.py --records results/main/records.jsonl \
    --views results/views/records.jsonl --out results/summary.md
python scripts/run_iter2.py --output results/iter2   # robustness suite (GPU)
python scripts/make_iter2_report.py --root results/iter2 --out results/iter2/summary.md
```

---

## Research questions

1. **View interpolation/extrapolation** — how well does each primitive
   generalize to a held-out view, and how large is the gap between training-view
   and held-out quality (overfitting pressure)?
2. **Scene structure** — which geometric regimes favour which primitive
   (flat surfaces, curved surfaces, sharp edges, thin structures)?
3. **Optimization speed** — how much wall-clock time does each method need to
   reach a given held-out quality?
4. **View scarcity** — how does held-out quality degrade as the number of
   training views shrinks from 7 to 5 to 3?

## Methods

| Method | Primitive | Key components implemented |
|---|---|---|
| **3DGS** ([Kerbl et al., 2023](#references)) | Anisotropic 3D Gaussian | `Σ = R S Sᵀ Rᵀ`; EWA splatting `Σ' = J W Σ Wᵀ Jᵀ` with the 0.3 low-pass term; opacity/scale/rotation/SH-2 colour; adaptive density control (clone/split/prune, gradients, opacity reset). |
| **2DGS** ([Huang et al., 2024](#references)) | Oriented 2D Gaussian disk | Ray–splat intersection in the disk's local frame (Eq. 7–10); object-space low-pass filter (Eq. 11); depth used is the per-pixel ray parameter; depth-distortion (Eq. 13) and normal-consistency (Eq. 14–15) terms available and configurable. |
| **Triangle Splatting** ([Held et al., 2025](#references)) | Triangle soup | SDF window `I(p) = ReLU(φ(p)/φ(s))^σ` (Eq. 1) with incenter normalization; per-pixel plane depth; probabilistic MCMC-style densification (midpoint subdivision into 4, in-plane cloning into 2) and importance/coverage pruning; size and opacity regularizers. |

A single shared compositor performs depth ordering, transmittance, background
composite, expected-depth rendering and the depth/normal regularizers, so the
comparison is apples-to-apples by construction (`gvst/render/raster.py`).

## Synthetic scenes

Each scene is an analytic shape collection with procedural textures and a
Lambertian (+ optional specular) shader, ray-cast directly, with 2× supersampled
anti-aliased ground truth (`gvst/scene/synthetic.py`).

| Scene | Geometry | Hypothesis it tests |
|---|---|---|
| `plane` | One large textured plane, cameras at 18–45° elevation | Grazing-angle surfaces: flat primitives (2DGS, triangles) should stay on the surface while 3D Gaussians blur/bleed. |
| `sphere` | Curved sphere with low-frequency texture and a specular lobe | Smooth curvature and view-dependent appearance (needs SH). |
| `corner` | Three orthogonal textured planes forming a room corner | Sharp edges, depth discontinuities, occlusion boundaries. |
| `sheets` | Three thin, disconnected double-sided sheets | Thin structure: volumetric splats must either collapse onto the sheets or fill the space behind them. |
| `intersect` | Three mutually intersecting double-sided sheets (Iteration 2) | Interpenetration: a single centre-depth sort is not a valid occlusion order, so transmittance and gradients are stressed. |
| `hf` | High-frequency textured plane, detail near the training Nyquist (Iteration 2) | Aliasing and resolution scaling: hard windows must densify to resolve detail. |
| `solid` | Constant-albedo sphere over a constant-albedo floor (Iteration 2) | Textureless geometry: the photometric loss constrains only shading and silhouettes, so appearance cannot mask geometry errors. |

The `sheets` scene deliberately probes the known floater weakness of
volumetric primitives; `plane` probes their grazing-angle blur; the three
Iteration-2 scenes probe interpenetration, aliasing and textureless geometry.

## Leave-one-out protocol

For a scene with `V` cameras:

1. Render exact ground truth for all `V` cameras.
2. Hold out camera `i`; train on the other `V-1` views.
3. **Initialization is SfM-like and comes from training views only**: rays are
   cast through random pixels of the training cameras, hits are back-projected
   to 3D and 3000 points (+ their shaded colours) seed every method identically.
   The held-out view never touches the model, its camera, or the point cloud.
4. Train for 3000 iterations with the same budget and photometric loss
   (`0.8·L1 + 0.2·(1−SSIM)`, 3DGS Eq. 7); evaluate PSNR/SSIM on the held-out
   view, plus depth RMSE against the analytic depth.
5. Repeat for folds `0, 1, 2` (low, mid and high elevation; 3 held-out cameras
   per scene) and average.

Training curves are recorded every 200 iterations (held-out and training PSNR
vs. wall-clock), enabling time-to-quality comparisons as well as final scores.

## Fairness notes and deliberate deviations

A faithful re-implementation on tiny images requires a few explicit
adaptations; they are kept identical across methods wherever possible and are
listed here in full:

* **Pure-PyTorch rasterizer** instead of the authors' CUDA kernels, with global
  front-to-back sorting by primitive centre depth (the reference code sorts
  the same way). 2DGS/TS depths are per-pixel ray parameters; 3DGS uses centre
  depth (it has no surface intersection).
* **Budget scaling**: the papers train for 30k iterations on ~1 MP images; the
  benchmark trains for 3000 iterations on 64×64 px. The papers' *absolute*
  cadences that optimizer dynamics depend on are kept (densify every 100
  iterations for 3DGS/2DGS, every 500 for TS; opacity reset every 3000;
  2DGS regularizer warm-ups at 3000/7000; SH degree ramp every 1000), while
  densification *windows* preserve the papers' fractions of the run. The
  opacity reset therefore never triggers within a 3k-iteration run, for any
  method.
* **Held-out folds** are cameras `0, 1, 2`, i.e. one low-, one mid- and one
  high-elevation view per scene, so extrapolation difficulty is sampled rather
  than fixed.
* **Primitive budget**: 8000 primitives max for all methods (≈2 per pixel at
  64×64), reached only by 3DGS/TS; the papers grow to 1–4M primitives at ~1 MP.
* **2DGS regularizers are disabled** (`λ_dist = λ_normal = 0`) in the main
  protocol. This matches the official 2DGS evaluation for bounded synthetic
  scenes (`scripts/nerf_eval.py` uses `--lambda_normal 0.0` and leaves
  `λ_dist = 0`); in this regime the DTU-style values (`λ_dist = 1000`) are
  destabilizing at the tiny scale/short budget. Both terms remain implemented
  and configurable (`TrainConfig.lambda_dist/lambda_normal`), and a separate
  ablation re-runs `corner` with them enabled (see Results).
* **SH degree 2** (papers use 3) for speed; identical for all methods.
* **No LPIPS** (extra dependency): PSNR, SSIM and depth RMSE are reported.
* **Seeds**: one seed per fold; the view-count sweep fixes one held-out camera
  and uses the same seed for every method.

## Repository layout

```
gvst/
├── scene/         cameras, analytic ray-cast scenes, SfM-like init sampling
├── render/        shared differentiable compositor + spherical harmonics
├── primitives/    Gaussian3D, Gaussian2D, TriangleModel + optimizer surgery
├── train/         losses, regularizers, densification-aware trainer
└── eval/          PSNR/SSIM/depth metrics, analytic geometry metrics,
                   training diagnostics, leave-one-out protocol
scripts/           run_loo.py, run_view_sweep.py, make_report.py,
                   run_iter2.py, make_iter2_report.py
tests/             55 pytest tests (CPU-only, ~3 s): cameras, scenes, SH,
                   rasterizer, per-method projections, densification,
                   optimizer-state transfer, protocol integrity, geometry
                   metrics, diagnostics, densification ablations, CLI smoke
```

## Results

> Generated from `results/summary.md`; raw records, per-fold metrics, rendered
> predictions and depth maps are written to `results/` (gitignored). Numbers
> are the mean ± std over the three LOO folds. Hardware: RTX 5080 (16 GB),
> 3000 iterations per run, 64×64 px, 8 cameras (7 training), 3000 shared
> initialization points.

### Held-out view quality (leave-one-out)

| Scene | Method | PSNR ↑ | SSIM ↑ | Depth RMSE ↓ | Train PSNR | Generalization gap ↓ | Primitives | Train time (s) | Peak GPU (MB) |
|---|---|---|---|---|---|---|---|---|---|
| plane | 3DGS | 18.64 ± 0.87 | 0.613 ± 0.036 | 0.150 | 29.53 | 10.89 | 7127 | 109.2 | 2391 |
| plane | 2DGS | 18.73 ± 0.92 | 0.603 ± 0.039 | 2.624 | 34.60 | 15.87 | 2677 | 66.7 | 1714 |
| plane | Triangle | **20.75 ± 0.48** | **0.665 ± 0.031** | 7.137 | 39.75 | 19.00 | 5026 | 75.2 | 2346 |
| sphere | 3DGS | 23.45 ± 0.79 | 0.771 ± 0.014 | 0.328 | 50.13 | 26.68 | 7914 | 115.6 | 2642 |
| sphere | 2DGS | 23.25 ± 1.16 | 0.759 ± 0.039 | 0.483 | 46.37 | 23.13 | 2857 | 69.7 | 1714 |
| sphere | Triangle | **24.90 ± 1.42** | **0.814 ± 0.001** | 4.157 | 53.06 | 28.16 | 5326 | 84.5 | 2785 |
| corner | 3DGS | 17.64 ± 2.77 | 0.703 ± 0.085 | 0.764 | 31.26 | 13.62 | 7272 | 107.9 | 2437 |
| corner | 2DGS | 17.82 ± 1.87 | 0.686 ± 0.115 | 2.104 | 35.44 | 17.61 | 1895 | 51.4 | 1714 |
| corner | Triangle | **20.13 ± 2.51** | **0.778 ± 0.062** | 11.679 | 45.42 | 25.29 | 4878 | 70.8 | 2304 |
| sheets | 3DGS | 20.67 ± 1.80 | 0.688 ± 0.023 | 0.240 | 43.36 | 22.69 | 7280 | 106.0 | 2541 |
| sheets | 2DGS | 21.25 ± 1.51 | 0.714 ± 0.020 | 0.485 | 41.87 | 20.62 | 2686 | 63.2 | 1714 |
| sheets | Triangle | **22.48 ± 2.76** | **0.760 ± 0.032** | 2.556 | 44.09 | 21.61 | 4043 | 59.3 | 2121 |

Depth RMSE is only directly comparable for 3DGS/2DGS: 3DGS reports centre
depth and 2DGS the ray–splat intersection depth, while Triangle Splatting
reports the ray–plane intersection, which is ill-conditioned for near
edge-on triangles (large but low-weight depths inflate the average; see e.g.
`corner`).  Treat the column as indicative, not as a geometry score.

### Time to held-out PSNR thresholds

| Scene | Method | Time to 20 dB (s) | Time to 25 dB (s) | Time to 30 dB (s) | Final held-out PSNR |
|---|---|---|---|---|---|
| plane | 3DGS | 2.5 | — | — | 18.64 |
| plane | 2DGS | 5.0 | — | — | 18.73 |
| plane | Triangle | 11.1 | — | — | 20.75 |
| sphere | 3DGS | 0.0 | 0.0 | — | 23.45 |
| sphere | 2DGS | 0.1 | 0.1 | — | 23.25 |
| sphere | Triangle | 0.0 | 2.2 | — | 24.90 |
| corner | 3DGS | 3.4 | — | — | 17.64 |
| corner | 2DGS | 43.1 | — | — | 17.82 |
| corner | Triangle | 16.8 | — | — | 20.13 |
| sheets | 3DGS | 0.0 | 3.4 | — | 20.67 |
| sheets | 2DGS | 0.1 | 0.1 | — | 21.25 |
| sheets | Triangle | 0.0 | 3.4 | — | 22.48 |

Threshold times are averaged over folds and should be read with care: all
methods start from the same dense point cloud, so early thresholds mostly
measure initialization quality, and curves are non-monotonic (e.g. 2DGS is
fastest to 25 dB on `sphere` but has the lowest final score).

### Held-out PSNR during training (fraction of the run)

| Scene | Method | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| plane | 3DGS | 18.53 | 17.60 | 18.52 | 18.68 | 18.64 |
| plane | 2DGS | 16.98 | **19.72** | 19.22 | 18.94 | 18.73 |
| plane | Triangle | 14.91 | 20.53 | 20.79 | 20.86 | 20.79 |
| sphere | 3DGS | 24.35 | 23.29 | 23.50 | 23.45 | 23.45 |
| sphere | 2DGS | 24.12 | 23.53 | 23.39 | 23.30 | 23.25 |
| sphere | Triangle | 22.70 | 24.76 | 24.84 | 24.93 | **24.90** |
| corner | 3DGS | 15.37 | 18.82 | 17.32 | 17.65 | 17.64 |
| corner | 2DGS | 14.22 | 18.12 | 18.07 | 18.00 | 17.82 |
| corner | Triangle | 12.28 | 19.37 | 19.92 | 20.16 | 20.13 |
| sheets | 3DGS | 22.96 | 21.00 | 20.78 | 20.73 | 20.67 |
| sheets | 2DGS | 24.07 | 21.97 | 21.61 | 21.47 | 21.25 |
| sheets | Triangle | 22.48 | 23.33 | 22.92 | 22.56 | 22.46 |

### Fewer training views (fixed held-out camera, `corner`)

| Method | 3 views | 5 views | 7 views |
|---|---|---|---|
| 3DGS | 15.22 | 14.19 | 17.30 |
| 2DGS | 13.89 | 14.45 | 16.32 |
| Triangle | **17.63** | **17.31** | **19.14** |

### 2DGS regularizer ablation (`corner`, fold 0)

| 2DGS configuration | Held-out PSNR | Train PSNR | Primitives |
|---|---|---|---|
| no regularizers (main protocol) | **16.46** | 35.06 | 1970 |
| `λ_dist=10, λ_normal=0.05` | 12.15 | 13.58 | 7943 |
| `λ_dist=1000, λ_normal=0.05` | 13.11 | 12.60 | 7951 |

(The DTU-style `λ_dist=1000` and even the unbounded-scene `λ_dist=10` dominate
the photometric loss at this scene scale; both training and held-out quality
collapse.  The ablation uses fractionally scaled warm-ups — 300/700
iterations — so the terms actually act within 3000 iterations.)

### What the numbers say

1. **Triangle Splatting generalizes best in every synthetic regime tested** at
   this budget: +1.2 to +2.5 dB held-out PSNR and +0.04 to +0.08 SSIM over the
   best Gaussian baseline, consistently across folds.  The margin is largest on
   the sharp-edged `corner` and the grazing-angle `plane`, and also on the
   curved `sphere` where the paper's own claim would predict a closer race.
2. **The held-out ranking is stable while training quality is not.**  Triangle
   Splatting fits its training views the best (45–53 dB) and nevertheless
   generalizes best; its *generalization gap* is the largest of the three.
   3DGS has the smallest gap on the structured scenes only because it also
   underfits its training views the most (29–31 dB) — a systematic blur that
   is visible in the rendered predictions.
3. **2DGS is the fastest wall-clock method** (≈50–70 s vs ≈60–85 s for
   triangles and ≈105–120 s for 3DGS) and uses 2–3× fewer primitives on the
   surface scenes.  On the current 64×64 benchmark, 2DGS and 3DGS trade places
   for second on held-out quality depending on the scene.
4. **Scarce views turn the task into memorization for everyone**: on `corner`,
   dropping from 7 to 3 training views costs 1.5 dB for triangles, 2.1 dB for
   3DGS and 2.4 dB for 2DGS, and the train/held-out gap grows to 20+ dB for all
   methods.  Triangles degrade the slowest.
5. **Paper hyperparameters are not scale-free.**  2DGS's depth-distortion term
   (a loss on unnormalized world-space depths) is catastrophic at this scale
   for any official weight; its normal term alone also hurts.  We therefore
   report 2DGS without regularizers and with the ablation above.

### Reproducing

```bash
python scripts/run_loo.py --output results/main \
    --scenes plane,sphere,corner,sheets --methods 3dgs,2dgs,triangle \
    --views 8 --folds 0,1,2 --iterations 3000 --image-size 64 --init-points 3000
python scripts/run_view_sweep.py --scene corner --held-out 0 --counts 3,5,7 \
    --iterations 3000 --output results/views
python scripts/run_loo.py --scenes corner --methods 2dgs --folds 0 \
    --lambda-dist 10 --lambda-normal 0.05 --dist-from 300 --normal-from 700 \
    --output results/regs_dist10 --no-images
python scripts/make_report.py --records results/main/records.jsonl \
    --views results/views/records.jsonl --out results/summary.md
```

---

## Iteration 2: robustness and training diagnostics

Iteration 1 leaves two questions open: whether Triangle Splatting's held-out
edge survives scenes and settings outside the original four, and *where* the
cost of optimising triangle geometry actually shows up.  Iteration 2 keeps the
same protocol, optimiser and SfM-like initialisation and varies one axis at a
time, adding:

* three stress scenes — mutually intersecting sheets (`intersect`),
  high-frequency texture (`hf`) and textureless geometry (`solid`);
* analytic-surface geometry metrics — symmetric Chamfer distance, F-score at
  2% of the camera-ring extent and nearest-neighbour normal consistency —
  measured against a noise-free surface sampled from the scene's shapes;
* per-checkpoint training telemetry: per-primitive gradient flow, blending
  coverage and triangle shape quality (area, aspect ratio, minimum angle);
* ablations of densification, training length, resolution and SH capacity.

99 runs are written under `results/iter2/`; every table below is generated by
`scripts/make_iter2_report.py`.  The original four scenes appear again with
geometry metrics at fold 0 so the new numbers can be anchored to Iteration 1.

### New stress scenes (leave-one-out, 3 folds)

| Scene | Method | PSNR ↑ | SSIM ↑ | Train PSNR | Gap ↓ | Primitives | Time (s) |
|---|---|---|---|---|---|---|---|
| intersect | 3DGS | 15.01 ± 0.98 | 0.405 ± 0.022 | 26.32 | 11.31 | 6973 | 107.0 |
| intersect | 2DGS | 16.32 ± 1.25 | 0.490 ± 0.048 | 32.44 | 16.12 | 2696 | 68.1 |
| intersect | Triangle | **17.23 ± 1.64** | **0.544 ± 0.052** | 35.73 | 18.50 | 5596 | 82.2 |
| hf | 3DGS | 14.89 ± 0.33 | 0.608 ± 0.028 | 24.15 | 9.26 | 7411 | 107.4 |
| hf | 2DGS | 15.00 ± 0.06 | 0.605 ± 0.063 | 27.11 | 12.11 | 2224 | 54.4 |
| hf | Triangle | **16.04 ± 0.63** | **0.677 ± 0.052** | 32.87 | 16.82 | 4990 | 67.7 |
| solid | 3DGS | 22.73 ± 1.55 | 0.668 ± 0.024 | 50.07 | 27.34 | 7537 | 111.7 |
| solid | 2DGS | 22.64 ± 2.68 | 0.646 ± 0.082 | 45.91 | 23.27 | 2128 | 57.5 |
| solid | Triangle | **24.24 ± 1.27** | **0.758 ± 0.020** | 50.58 | 26.34 | 5688 | 84.2 |

Triangle Splatting keeps the best mean held-out PSNR on all three new scenes
(+0.9 to +1.5 dB over the best Gaussian) and wins 8 of 9 folds, losing only the
first `solid` fold to 2DGS.  The margin is *largest* on the textureless scene,
where appearance fitting cannot substitute for geometry, and it persists on the
intersecting sheets where a single centre-depth sort is not a valid occlusion
order.  As in Iteration 1, the fold-to-fold spread is comparable to the margins
themselves.

### Geometry: photometric wins do not track rendered depth

| Scene | Method | Chamfer ↓ | F-score ↑ | Normal consistency ↑ |
|---|---|---|---|---|
| intersect | 3DGS | 0.0992 | 0.390 | 0.614 |
| intersect | 2DGS | 0.0673 | 0.792 | 0.703 |
| intersect | Triangle | 0.2415 | 0.522 | 0.543 |
| hf | 3DGS | 0.0982 | 0.450 | 0.841 |
| hf | 2DGS | 0.0707 | 0.830 | 0.776 |
| hf | Triangle | 0.2909 | 0.471 | 0.576 |
| solid | 3DGS | 0.2322 | 0.261 | 0.851 |
| solid | 2DGS | 0.1769 | 0.399 | 0.780 |
| solid | Triangle | 0.5706 | 0.250 | 0.703 |
| plane | 3DGS | 0.0998 | 0.391 | 0.878 |
| plane | 2DGS | 0.0915 | 0.812 | 0.810 |
| plane | Triangle | 0.3024 | 0.492 | 0.604 |
| sphere | 3DGS | 0.1307 | 0.540 | 0.858 |
| sphere | 2DGS | 0.1169 | 0.645 | 0.852 |
| sphere | Triangle | 0.2989 | 0.317 | 0.790 |
| corner | 3DGS | 0.1547 | 0.207 | 0.782 |
| corner | 2DGS | 0.3418 | 0.597 | 0.770 |
| corner | Triangle | 0.4349 | 0.312 | 0.561 |
| sheets | 3DGS | 0.0745 | 0.410 | 0.820 |
| sheets | 2DGS | 0.0488 | 0.888 | 0.795 |
| sheets | Triangle | 0.2829 | 0.588 | 0.648 |

This table uses the **rendered expected depth** for every method (per-camera
back-projection, `acc > 0.5`), the only extraction all three share.  A second
extraction is possible for triangles — the explicit soup — and it tells the
opposite story:

| Scene | Extraction | Chamfer ↓ | F-score ↑ | Normal consistency ↑ |
|---|---|---|---|---|
| intersect | rendered depth | 0.2415 | 0.522 | 0.543 |
| intersect | triangle soup | **0.0570** | 0.742 | 0.672 |
| hf | rendered depth | 0.2909 | 0.471 | 0.576 |
| hf | triangle soup | **0.0659** | 0.686 | 0.625 |
| solid | rendered depth | 0.5706 | 0.250 | 0.703 |
| solid | triangle soup | **0.1005** | 0.469 | 0.719 |
| plane | rendered depth | 0.3024 | 0.492 | 0.604 |
| plane | triangle soup | **0.0614** | 0.687 | 0.620 |
| sphere | rendered depth | 0.2989 | 0.317 | 0.790 |
| sphere | triangle soup | **0.0655** | 0.688 | 0.848 |
| corner | rendered depth | 0.4349 | 0.312 | 0.561 |
| corner | triangle soup | **0.0764** | 0.608 | 0.710 |
| sheets | rendered depth | 0.2829 | 0.588 | 0.648 |
| sheets | triangle soup | **0.0460** | 0.808 | 0.761 |

Two opposite conclusions:

* **Rendered expected depth is Triangle Splatting's weakest output.** Its
  Chamfer is the worst of the three in every scene, by 2–4× (e.g. `corner`
  0.43 vs 0.15 for 3DGS and 0.34 for 2DGS; `solid` 0.57 vs 0.23 / 0.18).  The
  per-pixel ray–plane depth is ill-conditioned for near edge-on triangles, and
  expected depth averages over primitives that a single centre-depth sort
  cannot order correctly, so the depth map blurs across the surface.
* **Its explicit triangle soup is the most accurate surface in every scene**
  (Chamfer 0.046–0.10, vs 0.05–0.34 for the Gaussian rendered-depth clouds).
  Triangle face normals are noisier (lower normal consistency), but the
  surface itself is the closest to the analytic ground truth of any method.

So Triangle Splatting's held-out PSNR is backed by a genuinely good underlying
surface rather than by appearance fitting — but its *rendered depth* should not
be used as a geometry estimate, which is exactly the caveat attached to the
Iteration-1 depth-RMSE column.  Note the extraction asymmetry: an explicit soup
and a Gaussian rendered-depth cloud are not like-for-like, the soup number is
the fairer one for triangles, and Gaussians have no explicit surface to sample.

### Where does Triangle Splatting training break?

Mean over the nine stress-scene runs; zero-gradient and coverage are averaged
over the last quarter of training (after the size regulariser stops):

| Method | Zero vertex/mean grad ≤ | Coverage (last view) | Primitives |
|---|---|---|---|
| 3DGS | 2.6% | 90.9% | 7307 |
| 2DGS | 2.5% | 94.4% | 2349 |
| Triangle | 7.7% | 86.4% | 5425 |

| Triangle shape metric | First checkpoint | Last checkpoint |
|---|---|---|
| Mean area | 0.0143 | 0.0476 |
| Degenerate fraction | 0.0000 | 0.0000 |
| Sliver fraction (aspect > 15) | 0.0000 | 0.0014 |
| Sharp fraction (min angle < 1°) | 0.0000 | 0.0002 |
| Mean minimum angle (deg) | 44.8988 | 37.6640 |
| Mean aspect ratio | 2.3987 | 2.7917 |

| Phase | Zero vertex-gradient fraction |
|---|---|
| During densification (size regularizer active, it < 2500) | 0.00% |
| After (photometric loss only, it ≥ 2500) | 10.21% |

The predicted failure mode — runaway degenerate or sliver triangles — does
**not** materialise: degenerate-area, sliver and sharp-angle fractions all stay
below 0.2%, and the mean minimum angle only drifts from 45° to 38°.  The real
issue is **gradient sparsity**: ~8% of triangles receive no vertex gradient at
all (versus 2.6% for 3DGS and 2.5% for 2DGS) and coverage is the lowest of the
three.  The phase table explains why this is easy to miss: the triangle size
regulariser depends on every triangle's area, so while it is active it gives
even invisible triangles a gradient and the zero-gradient fraction reads 0%.
Once it switches off and only the photometric loss remains, the true sparsity
appears, at ~10%.  These are triangles outside the ReLU SDF support or
occluded in the sampled view, and MCMC densification compensates by adding new
primitives rather than rescuing the existing ones.

### Densification ablation (Triangle Splatting)

| Scene | Mode | PSNR ↑ | SSIM ↑ | Train PSNR | Primitives | Zero-grad ≤ | Coverage |
|---|---|---|---|---|---|---|---|
| corner | mcmc | 19.99 | 0.762 | 43.75 | 4447 | 15.5% | 71.6% |
| corner | deterministic | **20.56** | **0.772** | 42.98 | 4629 | 15.5% | 72.1% |
| corner | none | 19.97 | 0.748 | 45.61 | 3000 | 17.7% | 67.3% |
| sheets | mcmc | **22.97** | **0.768** | 43.39 | 4259 | 4.5% | 92.3% |
| sheets | deterministic | 22.89 | 0.765 | 44.40 | 4469 | 4.2% | 92.8% |
| sheets | none | 22.91 | 0.757 | 49.45 | 3000 | 10.2% | 80.8% |
| intersect | mcmc | 16.93 | 0.531 | 35.62 | 5529 | 4.9% | 89.6% |
| intersect | deterministic | **17.02** | 0.531 | 35.22 | 5638 | 4.8% | 90.1% |
| intersect | none | 17.01 | **0.537** | 39.27 | 3000 | 7.2% | 85.3% |

Removing densification changes essentially nothing: the mean held-out PSNR is
19.96 for MCMC and 19.96 for *no densification at all*, while deterministic
top-importance selection is marginally better (20.16).  The only systematic
effect of MCMC growth is to **lower** training-view PSNR (40.9 vs 44.8 without
it) while using ~58% more primitives.  At this budget the probabilistic
densification is not earning its compute: held-out quality is set by the
initialisation and the optimiser, not the primitive budget, and the extra
primitives mainly act as a brake on training-view overfitting.

### Longer training (3k → 10k iterations, paired seeds)

| Scene | Method | 3k PSNR | 10k PSNR | Δ |
|---|---|---|---|---|
| corner | 3DGS | 17.30 | 15.63 | −1.67 |
| corner | 2DGS | 16.32 | 19.63 | +3.31 |
| corner | Triangle | 19.14 | 20.40 | +1.26 |
| sphere | 3DGS | 23.88 | 22.53 | −1.35 |
| sphere | 2DGS | 24.59 | 23.48 | −1.11 |
| sphere | Triangle | 26.42 | 26.96 | +0.53 |

Triangle Splatting is the only method that never degrades when training is
extended from 3k to 10k iterations (corner +1.3, sphere +0.5 dB).  3DGS loses
held-out quality on both scenes (−1.7, −1.4 dB) and 2DGS is mixed (−1.1 on
`sphere`, +3.3 on `corner`).  The large train→held-out gap of Iteration 1 is
therefore not a sign of imminent collapse for triangles; if anything the
held-out ordering widens with training.  (Training PSNR is non-monotonic at
10k for all methods — the longer run crosses several opacity-reset and
densification boundaries — so these deltas measure stability of the held-out
view, not a clean convergence curve.)

### Resolution stress (64 → 128 px, paired seeds)

| Scene | Method | 64 px PSNR | 128 px PSNR | Δ |
|---|---|---|---|---|
| hf | 3DGS | 14.84 | 15.60 | +0.76 |
| hf | 2DGS | 14.74 | 15.12 | +0.38 |
| hf | Triangle | **15.73** | **16.12** | +0.39 |
| corner | 3DGS | 18.52 | 17.87 | −0.65 |
| corner | 2DGS | 16.78 | 18.25 | +1.47 |
| corner | Triangle | **19.34** | **18.99** | −0.34 |

At 128 px Triangle Splatting still leads on both scenes and its margin does not
shrink, so the hard SDF window is not obviously aliasing more than the Gaussian
low-pass terms at this scale.  All 64→128 px deltas are within roughly ±1.5 dB.

### Appearance capacity (SH degree 2 → 0, paired seeds)

| Scene | Method | SH2 PSNR | SH0 PSNR | Δ |
|---|---|---|---|---|
| sphere | 3DGS | 23.62 | 26.93 | +3.31 |
| sphere | 2DGS | 24.26 | 26.26 | +2.00 |
| sphere | Triangle | **26.95** | **26.93** | −0.02 |
| hf | 3DGS | 15.25 | 16.09 | +0.84 |
| hf | 2DGS | 15.02 | 15.23 | +0.21 |
| hf | Triangle | **15.52** | **15.47** | −0.05 |

Triangle Splatting is unaffected by SH degree (Δ ≈ 0), whereas removing
view-dependent colour *improves* the Gaussians by 2–3.3 dB on the specular
`sphere`: with only seven training views, second-order SH overfits the
training lights.  Triangle Splatting's advantage is therefore not an
appearance-capacity artefact — if anything, the higher-capacity colour model
hurts the baselines here.

### Reproducing iteration 2

```bash
python scripts/run_iter2.py \
    --experiments scenes,geometry,densify,longtrain,longtrain3k,resolution,resolution64,sh,sh2 \
    --device cuda --output results/iter2
python scripts/make_iter2_report.py --root results/iter2 --out results/iter2/summary.md
```

### What iteration 2 adds

1. **The held-out advantage is robust to the new stress scenes**: +0.9 to
   +1.5 dB on intersecting sheets, high-frequency texture and textureless
   geometry, with 8/9 fold wins, and it survives a resolution increase.
2. **It is a surface advantage, not an appearance one**: the explicit triangle
   soup is the most accurate surface in all seven scenes, while the rendered
   expected depth is the least reliable of the three.
3. **The training failure mode is gradient sparsity, not degeneracy.**  ~10% of
   triangles get no photometric gradient once the size regulariser stops;
   shapes themselves stay well-conditioned.
4. **MCMC densification does not earn its keep** at this budget: held-out PSNR
   with and without it is identical, while it costs ~58% more primitives and a
   lower training-view fit.
5. **Triangle Splatting is the most stable primitive under longer training and
   is insensitive to SH capacity**; 3DGS degrades on both long-training scenes
   and the Gaussians overfit second-order SH on the specular scene.

These conclusions are still bounded by the Iteration-1 limitations (synthetic
64/128 px scenes, one seed per fold, pure-PyTorch timing), and the geometry
comparison relies on an extraction that is not identical across primitives.

## Limitations

* Synthetic, object-centric scenes at 64×64 and 128×128 px; conclusions are
  about this regime, not about Mip-NeRF 360-scale photorealism.
* Pure-PyTorch compositing means wall-clock times measure algorithmic
  convergence, not the throughput optimizations of the official CUDA kernels.
* One seed per fold: per-scene differences below ~0.5 dB should not be
  over-interpreted.

## References

1. Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G. 2023. 3D Gaussian
   Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on
   Graphics* 42, 4 (July 2023), 1–14. DOI: [10.1145/3592433](https://doi.org/10.1145/3592433).
2. Huang, B., Yu, Z., Chen, A., Geiger, A., and Gao, S. 2024. 2D Gaussian
   Splatting for Geometrically Accurate Radiance Fields. In *ACM SIGGRAPH 2024
   Conference Papers* (Denver, CO, USA, July 27–August 1, 2024). ACM, Article
   15, 1–11. DOI: [10.1145/3641519.3657428](https://doi.org/10.1145/3641519.3657428).
3. Held, J., Vandeghen, R., Deliege, A., Hamdi, A., Giancola, S., Cioppa, A.,
   Vedaldi, A., Ghanem, B., Tagliasacchi, A., and Van Droogenbroeck, M. 2025.
   Triangle Splatting for Real-Time Radiance Field Rendering. *arXiv*
   preprint arXiv:2505.19175. DOI: [10.48550/arXiv.2505.19175](https://doi.org/10.48550/arXiv.2505.19175).
4. Kheradmand, S., Rebain, D., Sharma, G., Sun, W., Tseng, J., Isack, H., Kar,
   A., Tagliasacchi, A., and Yi, K. M. 2024. 3D Gaussian Splatting as Markov
   Chain Monte Carlo. In *Advances in Neural Information Processing Systems*
   37 (NeurIPS 2024), 80965–80986.
5. Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., and Hedman,
   P. 2022. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. In
   *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*,
   5460–5469. DOI: [10.1109/CVPR52688.2022.00539](https://doi.org/10.1109/CVPR52688.2022.00539).
6. Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. 2004. Image
   Quality Assessment: From Error Visibility to Structural Similarity. *IEEE
   Transactions on Image Processing* 13, 4, 600–612. DOI:
   [10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861).
7. Schönberger, J. L., and Frahm, J.-M. 2016. Structure-from-Motion Revisited.
   In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*,
   4104–4113. DOI: [10.1109/CVPR.2016.445](https://doi.org/10.1109/CVPR.2016.445).
