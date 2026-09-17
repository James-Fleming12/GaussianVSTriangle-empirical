#!/usr/bin/env python3
"""Aggregate benchmark records into Markdown tables for the README.

Usage::

    python scripts/make_report.py --records results/main/records.jsonl \
        --views results/views/records.jsonl --out results/summary.md
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics as st


def load(path):
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value, digits=2, unit=""):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}{unit}"


def mean_std(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return st.mean(values), st.stdev(values)


METHOD_ORDER = ("3dgs", "2dgs", "triangle")
METHOD_LABEL = {"3dgs": "3DGS", "2dgs": "2DGS", "triangle": "Triangle"}


def loo_tables(rows):
    by_scene = collections.OrderedDict()
    for row in rows:
        by_scene.setdefault(row["scene"], {}).setdefault(row["method"], []).append(row)

    out = []
    out.append("### Held-out view quality (leave-one-out)\n")
    out.append("| Scene | Method | PSNR ↑ | SSIM ↑ | Depth RMSE ↓ | Train PSNR | Generalization gap ↓ | Primitives | Train time (s) | Peak GPU (MB) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for scene, methods in by_scene.items():
        for method in METHOD_ORDER:
            if method not in methods:
                continue
            group = methods[method]
            psnr_m, psnr_s = mean_std([r["psnr"] for r in group])
            ssim_m, ssim_s = mean_std([r["ssim"] for r in group])
            depth_m, _ = mean_std([r.get("depth_rmse") for r in group])
            train_m, _ = mean_std([r["train_psnr"] for r in group])
            gap = None
            if psnr_m is not None and train_m is not None:
                gap, _ = mean_std([r["train_psnr"] - r["psnr"] for r in group])
            prim_m, _ = mean_std([r["n_primitives"] for r in group])
            sec_m, _ = mean_std([r["seconds"] for r in group])
            mem_m, _ = mean_std([r.get("peak_memory_mb", 0.0) for r in group])
            out.append(
                f"| {scene} | {METHOD_LABEL.get(method, method)} "
                f"| {fmt(psnr_m)} ± {fmt(psnr_s)} "
                f"| {fmt(ssim_m, 3)} ± {fmt(ssim_s, 3)} "
                f"| {fmt(depth_m, 3)} "
                f"| {fmt(train_m)} "
                f"| {fmt(gap)} "
                f"| {fmt(prim_m, 0)} "
                f"| {fmt(sec_m, 1)} "
                f"| {fmt(mem_m, 0)} |"
            )
        out.append("")

    out.append("### Time to held-out PSNR thresholds\n")
    out.append("| Scene | Method | Time to 20 dB (s) | Time to 25 dB (s) | Time to 30 dB (s) | Final held-out PSNR |")
    out.append("|---|---|---|---|---|---|")
    for scene, methods in by_scene.items():
        for method in METHOD_ORDER:
            if method not in methods:
                continue
            group = methods[method]

            def t2thr(threshold):
                times = []
                for r in group:
                    hit = None
                    for entry in r["history"]:
                        if entry.get("heldout_psnr", -1e9) >= threshold:
                            hit = entry["seconds"]
                            break
                    times.append(hit)
                m, _ = mean_std(times)
                return m

            psnr_m, _ = mean_std([r["psnr"] for r in group])
            out.append(
                f"| {scene} | {METHOD_LABEL.get(method, method)} "
                f"| {fmt(t2thr(20.0), 1)} | {fmt(t2thr(25.0), 1)} | {fmt(t2thr(30.0), 1)} | {fmt(psnr_m)} |"
            )
        out.append("")
    return "\n".join(out)


def convergence_table(rows):
    out = ["### Held-out PSNR during training (fraction of the run)\n"]
    out.append("| Scene | Method | 0% | 25% | 50% | 75% | 100% |")
    out.append("|---|---|---|---|---|---|---|")
    by_scene = collections.OrderedDict()
    for row in rows:
        by_scene.setdefault(row["scene"], {}).setdefault(row["method"], []).append(row)
    for scene, methods in by_scene.items():
        for method in METHOD_ORDER:
            if method not in methods:
                continue
            group = methods[method]
            fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
            cols = []
            for frac in fractions:
                vals = []
                for r in group:
                    hist = r["history"]
                    if not hist:
                        continue
                    idx = min(len(hist) - 1, round(frac * (len(hist) - 1)))
                    vals.append(hist[idx].get("heldout_psnr"))
                m, _ = mean_std(vals)
                cols.append(fmt(m))
            out.append(f"| {scene} | {METHOD_LABEL.get(method, method)} | " + " | ".join(cols) + " |")
        out.append("")
    return "\n".join(out)


def view_table(rows):
    if not rows:
        return ""
    by = collections.OrderedDict()
    for row in rows:
        by.setdefault(row["scene"], {}).setdefault(row["method"], {}).setdefault(row["n_train"], []).append(row)
    counts = sorted({r["n_train"] for r in rows})
    out = ["### Fewer training views (fixed held-out camera)\n"]
    out.append("| Scene | Method | " + " | ".join(f"{c} views" for c in counts) + " |")
    out.append("|---|---|" + "---|" * len(counts))
    for scene, methods in by.items():
        for method in METHOD_ORDER:
            if method not in methods:
                continue
            cells = []
            for count in counts:
                group = methods[method].get(count, [])
                m, _ = mean_std([r["psnr"] for r in group])
                cells.append(fmt(m))
            out.append(f"| {scene} | {METHOD_LABEL.get(method, method)} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", default="results/main/records.jsonl")
    p.add_argument("--views", default="results/views/records.jsonl")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    rows = load(args.records)
    if not rows:
        raise SystemExit(f"no records found at {args.records}")
    sections = [loo_tables(rows), convergence_table(rows)]
    vrows = load(args.views)
    if vrows:
        sections.append(view_table(vrows))
    text = "\n".join(sections)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
