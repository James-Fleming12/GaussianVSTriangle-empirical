#!/usr/bin/env python3
"""Aggregate the Iteration-2 experiments into Markdown tables for the README.

Usage::

    python scripts/make_iter2_report.py --root results/iter2 --out results/iter2/summary.md
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics as st


def load(root: str, name: str) -> list:
    path = os.path.join(root, name, "records.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def mean_std(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return st.mean(values), st.stdev(values)


def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def group(rows, keys):
    d = collections.OrderedDict()
    for r in rows:
        k = tuple(r.get(k) for k in keys)
        d.setdefault(k, []).append(r)
    return d


def diag_series(record, key):
    out = []
    for h in record.get("history", []):
        d = h.get("diagnostics")
        if d and d.get(key) is not None:
            out.append(d[key])
    return out


def diag_run_mean(record, key):
    vals = diag_series(record, key)
    return st.mean(vals) if vals else None


def diag_run_tail(record, key, frac: float = 0.25):
    """Mean of a diagnostic over the last ``frac`` of the checkpoints."""
    vals = diag_series(record, key)
    if not vals:
        return None
    k = max(1, int(len(vals) * frac))
    return st.mean(vals[-k:])


METHOD_LABEL = {"3dgs": "3DGS", "2dgs": "2DGS", "triangle": "Triangle"}
ORDER = ("3dgs", "2dgs", "triangle")


def _sorted_methods(rows):
    present = {r["method"] for r in rows}
    return [m for m in ORDER if m in present]


def gradient_phase_table(rows):
    """Triangle zero-gradient fraction before/after the size regularizer stops.

    The size regularizer is active for ``iteration < tri_densify_until_iter``
    (0.833 of a 3k run), and it depends on every triangle's area, so it gives
    even invisible triangles a gradient.  Removing it exposes how sparse the
    photometric gradient really is.
    """
    tri = [r for r in rows if r["method"] == "triangle"]
    if not tri:
        return ""
    cutoff = 2500
    during, after = [], []
    for r in tri:
        vals = [(h["iteration"], h["diagnostics"]["zero_grad_frac"])
                for h in r.get("history", []) if h.get("diagnostics")]
        d = [v for it, v in vals if it < cutoff]
        a = [v for it, v in vals if it >= cutoff]
        if d:
            during.append(st.mean(d))
        if a:
            after.append(st.mean(a))
    dm, _ = mean_std(during)
    am, _ = mean_std(after)
    out = ["### Triangle Splatting gradient sparsity by phase\n"]
    out.append("| Phase | Zero vertex-gradient fraction |")
    out.append("|---|---|")
    out.append(f"| During densification (size regularizer active, it < 2500) | {fmt(dm*100 if dm is not None else None, 2)}% |")
    out.append(f"| After (photometric loss only, it ≥ 2500) | {fmt(am*100 if am is not None else None, 2)}% |")
    return "\n".join(out)


def scene_table(rows, title):
    out = [f"### {title}\n"]
    out.append("| Scene | Method | PSNR ↑ | SSIM ↑ | Train PSNR | Gap ↓ | Primitives | Time (s) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for (scene, method), group_rows in group(rows, ("scene", "method")).items():
        p_m, p_s = mean_std([r["psnr"] for r in group_rows])
        s_m, s_s = mean_std([r["ssim"] for r in group_rows])
        t_m, _ = mean_std([r["train_psnr"] for r in group_rows])
        gap, _ = mean_std([r["train_psnr"] - r["psnr"] for r in group_rows])
        pr, _ = mean_std([r["n_primitives"] for r in group_rows])
        sec, _ = mean_std([r["seconds"] for r in group_rows])
        out.append(
            f"| {scene} | {METHOD_LABEL.get(method, method)} | {fmt(p_m)} ± {fmt(p_s)} "
            f"| {fmt(s_m, 3)} ± {fmt(s_s, 3)} | {fmt(t_m)} | {fmt(gap)} "
            f"| {fmt(pr, 0)} | {fmt(sec, 1)} |"
        )
    return "\n".join(out)


def geometry_table(rows, title="Geometry against the analytic surface"):
    out = [f"### {title}\n"]
    out.append("| Scene | Method | Chamfer ↓ | F-score ↑ | Normal consistency ↑ |")
    out.append("|---|---|---|---|---|")
    for (scene, method), group_rows in group(rows, ("scene", "method")).items():
        c, _ = mean_std([r.get("geom_chamfer") for r in group_rows])
        f, _ = mean_std([r.get("geom_fscore") for r in group_rows])
        n, _ = mean_std([r.get("geom_normal_consistency") for r in group_rows])
        label = METHOD_LABEL.get(method, method)
        out.append(f"| {scene} | {label} | {fmt(c, 4)} | {fmt(f, 3)} | {fmt(n, 3)} |")
    out.append("")
    out.append(
        "Chamfer/F-score/normal consistency use the rendered-depth surface cloud for every "
        "method; see the triangle-soup comparison below.\n"
    )
    return "\n".join(out)


def soup_table(rows, title="Triangle Splatting: explicit soup vs rendered depth"):
    out = [f"### {title}\n"]
    out.append("| Scene | Extraction | Chamfer ↓ | F-score ↑ | Normal consistency ↑ |")
    out.append("|---|---|---|---|---|")
    for (scene,), scene_rows in group(rows, ("scene",)).items():
        r_rows = [r for r in scene_rows if r["method"] == "triangle"]
        if not r_rows:
            continue
        for label, prefix in (("rendered depth", "geom_"), ("triangle soup", "geom_soup_")):
            c, _ = mean_std([r.get(f"{prefix}chamfer") for r in r_rows])
            f, _ = mean_std([r.get(f"{prefix}fscore") for r in r_rows])
            n, _ = mean_std([r.get(f"{prefix}normal_consistency") for r in r_rows])
            out.append(f"| {scene} | {label} | {fmt(c, 4)} | {fmt(f, 3)} | {fmt(n, 3)} |")
    return "\n".join(out)


def diagnostics_table(rows, title="Training diagnostics"):
    out = [f"### {title}\n"]
    out.append(
        "| Method | Zero vertex/mean grad ≤ | Coverage (last view) | Primitives |"
    )
    out.append("|---|---|---|---|")
    for method in _sorted_methods(rows):
        m_rows = [r for r in rows if r["method"] == method]
        zg, _ = mean_std([diag_run_tail(r, "zero_grad_frac") for r in m_rows])
        cov, _ = mean_std([diag_run_tail(r, "coverage_frac") for r in m_rows])
        pr, _ = mean_std([r["n_primitives"] for r in m_rows])
        out.append(f"| {METHOD_LABEL.get(method, method)} | {fmt(zg*100 if zg is not None else None, 1)}% "
                   f"| {fmt(cov*100 if cov is not None else None, 1)}% | {fmt(pr, 0)} |")
    out.append("")
    # triangle shape-health sub-table
    tri_rows = [r for r in rows if r["method"] == "triangle"]
    if tri_rows:
        out.append(
            "| Triangle shape metric | First checkpoint | Last checkpoint |\n|---|---|---|"
        )
        metrics = [
            ("area_mean", "Mean area"),
            ("degenerate_frac", "Degenerate fraction"),
            ("sliver_frac", "Sliver fraction (aspect > 15)"),
            ("sharp_frac", "Sharp fraction (min angle < 1°)"),
            ("min_angle_mean_deg", "Mean minimum angle (deg)"),
            ("aspect_mean", "Mean aspect ratio"),
        ]
        for key, label in metrics:
            firsts, lasts = [], []
            for r in tri_rows:
                series = diag_series(r, key)
                if series:
                    firsts.append(series[0])
                    lasts.append(series[-1])
            fm, _ = mean_std(firsts)
            lm, _ = mean_std(lasts)
            out.append(f"| {label} | {fmt(fm, 4)} | {fmt(lm, 4)} |")
    return "\n".join(out)


def densify_table(rows):
    out = ["### Triangle Splatting densification ablation\n"]
    out.append("| Scene | Mode | PSNR ↑ | SSIM ↑ | Train PSNR | Primitives | Zero-grad ≤ | Coverage |")
    out.append("|---|---|---|---|---|---|---|---|")
    for (scene, mode), group_rows in group(rows, ("scene", "tri_densify_mode")).items():
        p, _ = mean_std([r["psnr"] for r in group_rows])
        s, _ = mean_std([r["ssim"] for r in group_rows])
        t, _ = mean_std([r["train_psnr"] for r in group_rows])
        pr, _ = mean_std([r["n_primitives"] for r in group_rows])
        zg, _ = mean_std([diag_run_tail(r, "zero_grad_frac") for r in group_rows])
        cov, _ = mean_std([diag_run_tail(r, "coverage_frac") for r in group_rows])
        out.append(
            f"| {scene} | {mode} | {fmt(p)} | {fmt(s, 3)} | {fmt(t)} | {fmt(pr, 0)} "
            f"| {fmt(zg*100 if zg is not None else None, 1)}% "
            f"| {fmt(cov*100 if cov is not None else None, 1)}% |"
        )
    return "\n".join(out)


def paired_table(base_rows, comp_rows, base_label, comp_label, key="psnr"):
    """Paired base-vs-comparison table keyed by (scene, method, fold)."""
    out = []
    base = {(r["scene"], r["method"], r["fold"]): r for r in base_rows}
    comp = {(r["scene"], r["method"], r["fold"]): r for r in comp_rows}
    keys = [k for k in base if k in comp]
    if not keys:
        return ""
    out.append(f"| Scene | Method | {base_label} | {comp_label} | Δ |")
    out.append("|---|---|---|---|---|")
    per = collections.OrderedDict()
    for k in keys:
        per.setdefault((k[0], k[1]), []).append(comp[k][key] - base[k][key])
    for (scene, method), deltas in per.items():
        b, _ = mean_std([base[k][key] for k in keys if k[0] == scene and k[1] == method])
        c, _ = mean_std([comp[k][key] for k in keys if k[0] == scene and k[1] == method])
        d, _ = mean_std(deltas)
        out.append(f"| {scene} | {METHOD_LABEL.get(method, method)} | {fmt(b)} | {fmt(c)} | {fmt(d, 2)} |")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results/iter2")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    sections = []
    scenes = load(args.root, "scenes")
    geom = load(args.root, "geometry")
    if scenes:
        sections.append(scene_table(scenes, "New stress scenes (leave-one-out, 3 folds)"))
    if geom:
        sections.append(scene_table(geom, "Original scenes with geometry metrics (fold 0)"))
    if scenes or geom:
        sections.append(geometry_table(scenes + geom))
        sections.append(soup_table(scenes + geom))
    if scenes:
        sections.append(diagnostics_table(scenes))
        sections.append(gradient_phase_table(scenes))

    dens = load(args.root, "densify")
    if dens:
        sections.append(densify_table(dens))

    lt, lt3 = load(args.root, "longtrain"), load(args.root, "longtrain3k")
    if lt and lt3:
        sections.append("### Longer training (3k → 10k iterations)\n")
        sections.append(paired_table(lt3, lt, "3k PSNR", "10k PSNR"))
        sections.append("")
        sections.append(paired_table(lt3, lt, "3k train PSNR", "10k train PSNR", key="train_psnr"))

    res, res64 = load(args.root, "resolution"), load(args.root, "resolution64")
    if res and res64:
        sections.append("### Resolution stress (64 → 128 px)\n")
        sections.append(paired_table(res64, res, "64 px PSNR", "128 px PSNR"))

    sh, sh2 = load(args.root, "sh"), load(args.root, "sh2")
    if sh and sh2:
        sections.append("### Appearance capacity (SH degree 2 → 0)\n")
        sections.append(paired_table(sh2, sh, "SH2 PSNR", "SH0 PSNR"))

    text = "\n\n".join(s for s in sections if s)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
