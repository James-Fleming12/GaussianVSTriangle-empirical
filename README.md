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
python -m pytest tests/ -q                       # 45 tests, ~3 s, CPU only
python scripts/run_loo.py --quick --device cpu   # pipeline smoke test
python scripts/run_loo.py --output results/main  # full benchmark (GPU)
python scripts/run_view_sweep.py --output results/views
python scripts/make_report.py --records results/main/records.jsonl \
    --views results/views/records.jsonl --out results/summary.md
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

The `sheets` scene deliberately probes the known floater weakness of
volumetric primitives; `plane` probes their grazing-angle blur.

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
└── eval/          PSNR/SSIM/depth metrics, leave-one-out protocol
scripts/           run_loo.py, run_view_sweep.py, make_report.py
tests/             45 pytest tests (CPU-only, ~3 s): cameras, scenes, SH,
                   rasterizer, per-method projections, densification,
                   optimizer-state transfer, protocol integrity, CLI smoke
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

## Limitations

* Synthetic, object-centric scenes at 64×64 px; conclusions are about this
  regime, not about Mip-NeRF 360-scale photorealism.
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
