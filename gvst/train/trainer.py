"""Training loop shared by the three methods.

Differences between methods live in (a) the primitive model's ``screen_chunk``,
(b) the optional regularizers, and (c) the densification schedule -- all
selected here from :class:`gvst.config.TrainConfig`.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional

import torch
from torch import Tensor

from ..render.raster import depth_normal_map, distortion_loss, normal_consistency, render
from ..train.losses import photometric_loss, triangle_size_loss


class Trainer:
    def __init__(self, model, cfg, background: Tensor | None = None, device: str = "cpu", log: bool = False):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.background = (torch.zeros(3, device=self.device) if background is None else background.to(self.device))
        self.log = log

    # ------------------------------------------------------------------
    def _sh_degree(self, iteration: int) -> int:
        """3DGS raises the SH degree every 1000/30000 of training."""
        cfg = self.cfg
        step = max(1, cfg.iterations // 30)
        return min(cfg.sh_degree, iteration // step)

    def _loss(self, out, gt: Tensor, camera, iteration: int) -> Tensor:
        cfg = self.cfg
        model = self.model
        loss = photometric_loss(out.image, gt, cfg.lambda_dssim)

        if cfg.method == "2dgs":
            if iteration >= cfg.dist_from_iter and cfg.lambda_dist > 0:
                loss = loss + cfg.lambda_dist * distortion_loss(out.weights, out.depth_maps)
            if iteration >= cfg.normal_from_iter and cfg.lambda_normal > 0:
                n_map = depth_normal_map(camera, out.depth)
                loss = loss + cfg.lambda_normal * normal_consistency(out.weights, out.normals, n_map)

        elif cfg.method == "triangle":
            loss = loss + cfg.tri_lambda_opacity * model.opacity.abs().mean()
            if iteration < cfg.tri_densify_until_iter:
                loss = loss + cfg.tri_lambda_size * triangle_size_loss(model.triangle_area())
            if iteration >= cfg.tri_iteration_mesh:
                if cfg.tri_lambda_dist > 0:
                    loss = loss + cfg.tri_lambda_dist * distortion_loss(out.weights, out.depth_maps)
                if cfg.tri_lambda_normals > 0:
                    n_map = depth_normal_map(camera, out.depth)
                    loss = loss + cfg.tri_lambda_normals * normal_consistency(out.weights, out.normals, n_map)
        return loss

    def _need_weights(self) -> bool:
        return self.cfg.method in ("2dgs", "triangle")

    def _need_depth_maps(self) -> bool:
        if self.cfg.method == "2dgs":
            return self.cfg.lambda_dist > 0
        if self.cfg.method == "triangle":
            return self.cfg.tri_lambda_dist > 0
        return False

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, camera, need_weights: bool = True):
        was_training = self.model.training
        self.model.eval()
        out = render(
            self.model,
            camera,
            background=self.background,
            need_weights=need_weights,
            need_depth_maps=False,
        )
        if was_training:
            self.model.train()
        return out

    def fit(
        self,
        train_cameras: list,
        eval_camera=None,
        eval_target: Optional[Tensor] = None,
        eval_every: int = 200,
        train_targets: Optional[list] = None,
        progress: Optional[Callable] = None,
    ) -> dict:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        rng = random.Random(cfg.seed)

        self.model.to(self.device)
        self.model.train()
        self.model.setup_optimizer(cfg)

        if train_targets is None:
            train_targets = [None] * len(train_cameras)  # type: ignore[list-item]

        history = []
        t0 = time.time()
        last_loss = float("nan")
        for iteration in range(cfg.iterations + 1):
            self.model.set_sh_degree(self._sh_degree(iteration))
            self.model.update_learning_rate(iteration)

            if iteration == cfg.iterations:
                break

            idx = rng.randrange(len(train_cameras))
            camera = train_cameras[idx]
            gt = train_targets[idx]
            if gt is None:
                raise ValueError("train_targets must contain a ground-truth image per training camera")

            out = render(
                self.model,
                camera,
                background=self.background,
                need_weights=self._need_weights(),
                need_depth_maps=self._need_depth_maps(),
            )
            loss = self._loss(out, gt, camera, iteration)
            loss.backward()
            self.model.accumulate_densification_stats(out)

            if self.model.optimizer is not None:
                self.model.optimizer.step()
                self.model.optimizer.zero_grad(set_to_none=True)

            self.model.densify(iteration)
            self.model._view_ctx = {}
            last_loss = float(loss.detach())

            if eval_camera is not None and (iteration % eval_every == 0 or iteration == cfg.iterations - 1):
                entry = self._checkpoint(
                    iteration, train_cameras, train_targets, eval_camera, eval_target, rng, t0, last_loss
                )
                history.append(entry)
                if self.log:
                    print(
                        f"[{cfg.method}] iter {iteration:5d} loss {last_loss:.4f} "
                        f"heldout {entry['heldout_psnr']:.2f} dB train {entry['train_psnr']:.2f} dB "
                        f"n={entry['n_primitives']}"
                    )

        self.model.prune_final()
        self.model._view_ctx = {}
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        total_seconds = time.time() - t0

        result = {
            "method": cfg.method,
            "iterations": cfg.iterations,
            "seconds": total_seconds,
            "n_primitives": int(self.model.num_primitives),
            "history": history,
            "final_loss": last_loss,
        }
        if eval_camera is not None and eval_target is not None:
            out = self._evaluate(eval_camera, need_weights=False)
            result["heldout_image"] = out.image.detach().cpu()
            result["heldout_depth"] = out.depth.detach().cpu()
        return result

    def _checkpoint(self, iteration, train_cameras, train_targets, eval_camera, eval_target, rng, t0, loss) -> dict:
        from ..eval.metrics import evaluate

        # mean train PSNR over up to three fixed training views (stable estimate)
        n = len(train_cameras)
        idxs = sorted({0, n // 2, n - 1})
        t_psnr, t_ssim = [], []
        for i in idxs:
            t_out = self._evaluate(train_cameras[i], need_weights=False)
            m = evaluate(t_out.image, train_targets[i])
            t_psnr.append(m["psnr"])
            t_ssim.append(m["ssim"])
        entry = {
            "iteration": iteration,
            "loss": loss,
            "train_psnr": float(sum(t_psnr) / len(t_psnr)),
            "train_ssim": float(sum(t_ssim) / len(t_ssim)),
            "n_primitives": int(self.model.num_primitives),
        }
        if eval_camera is not None and eval_target is not None:
            e_out = self._evaluate(eval_camera, need_weights=False)
            e_metrics = evaluate(e_out.image, eval_target)
            entry["heldout_psnr"] = e_metrics["psnr"]
            entry["heldout_ssim"] = e_metrics["ssim"]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        entry["seconds"] = time.time() - t0
        return entry
