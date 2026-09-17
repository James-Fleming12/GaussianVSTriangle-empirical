"""CLI / benchmark smoke test: the whole pipeline runs and writes records."""

import json
import os

import torch

from gvst.config import BenchmarkConfig, SceneSpec, TrainConfig, scaled_train_config
from gvst.eval.loo import run_benchmark


def test_benchmark_smoke(tmp_path):
    cfg = scaled_train_config(TrainConfig(iterations=12, sh_degree=1, seed=0))
    bench = BenchmarkConfig(
        scenes=[
            SceneSpec(name="plane", description="", views=4, width=12, height=12, init_points=80),
            SceneSpec(name="sphere", description="", views=4, width=12, height=12, init_points=80),
        ],
        methods=("3dgs", "2dgs", "triangle"),
        train_config=cfg,
        folds=[0],
        seeds=(0,),
        output_dir=str(tmp_path / "results"),
        device="cpu",
        eval_every=4,
        save_images=True,
    )
    records = run_benchmark(bench)
    assert len(records) == 2 * 3
    path = os.path.join(bench.output_dir, "records.jsonl")
    assert os.path.exists(path)
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 6
    for row in rows:
        assert row["scene"] in ("plane", "sphere")
        assert row["method"] in ("3dgs", "2dgs", "triangle")
        assert torch.isfinite(torch.tensor(row["psnr"]))
        assert row["n_train"] == 3
    # prediction images are saved per run
    preds = []
    for root, _, files in os.walk(bench.output_dir):
        preds += [f for f in files if f == "heldout_pred.png"]
    assert len(preds) == 6
