"""Train the multi-modal predictor, with checkpointing that survives Colab.

Free-tier runtimes are reclaimed without warning, so state is written every
epoch and training resumes from the last checkpoint rather than restarting.
Point --checkpoint-dir at Drive so it outlives the machine.

Progress is reported as minADE and minFDE over the six modes, the same
quantities the project's evaluator computes, so a run can be compared directly
against the kinematic baselines it is meant to beat.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    OBJECT_TYPE_INDEX, FeatureDataset, consolidate, object_type_codes,
    sampling_weights, split_indices,
)
from model import TrajectoryPredictor, multimodal_loss


def displacement_metrics(
    trajectories: torch.Tensor, future: torch.Tensor, mask: torch.Tensor
) -> tuple[float, float, int]:
    """minADE and minFDE over the modes, respecting partial futures."""

    error = torch.linalg.vector_norm(trajectories - future.unsqueeze(1), dim=-1)
    expanded = mask.unsqueeze(1)
    counts = expanded.sum(dim=-1).clamp(min=1.0)
    ade = (error * expanded).sum(dim=-1) / counts
    best = ade.argmin(dim=1)
    rows = torch.arange(trajectories.shape[0], device=trajectories.device)

    min_ade = ade[rows, best]
    # Final displacement uses each agent's last observed step, not step 16.
    last = torch.where(mask.sum(dim=1) > 0, mask.cumsum(dim=1).argmax(dim=1), 0)
    min_fde = error[rows, best, last]
    keep = mask.sum(dim=1) > 0
    return float(min_ade[keep].sum()), float(min_fde[keep].sum()), int(keep.sum())


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    """Overall and per-class displacement error on the monitor split.

    Per-class figures are reported because the aggregate hid a class-dependent
    failure: the classes least represented in training were the ones the model
    degraded most against the kinematic baseline.
    """

    model.eval()
    ade_sum = fde_sum = 0.0
    seen = 0
    per_class: dict[int, list[float]] = {}
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        trajectories, _ = model(
            batch["target_history"], batch["neighbors"], batch["map_points"],
            batch["object_type"],
        )
        ade, fde, n = displacement_metrics(
            trajectories, batch["future"], batch["future_mask"]
        )
        ade_sum += ade
        fde_sum += fde
        seen += n
        error = torch.linalg.vector_norm(
            trajectories - batch["future"].unsqueeze(1), dim=-1
        )
        mask = batch["future_mask"].unsqueeze(1)
        rowwise = ((error * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).amin(dim=1)
        for code, value in zip(batch["object_type"].tolist(), rowwise.tolist()):
            per_class.setdefault(int(code), []).append(value)

    seen = max(seen, 1)
    metrics = {"min_ade_m": ade_sum / seen, "min_fde_m": fde_sum / seen, "agents": seen}
    names = {index: name for name, index in OBJECT_TYPE_INDEX.items()}
    for code, values in per_class.items():
        metrics[f"min_ade_{names.get(code, 'unknown')}"] = sum(values) / len(values)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True,
                        help="directory of per-shard .npz archives")
    parser.add_argument("--flat", type=Path, required=True,
                        help="where consolidated .npy arrays live")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--balance-classes", action="store_true",
                        help="oversample under-represented object types")
    parser.add_argument("--balance-damping", type=float, default=0.5,
                        help="0 leaves the distribution alone, 1 equalises it")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not (args.flat / "meta.json").exists():
        count = consolidate(args.features, args.flat)
        print(f"consolidated {count} examples", flush=True)
    count = json.loads((args.flat / "meta.json").read_text())["count"]

    train_idx, monitor_idx = split_indices(count, args.holdout, seed=args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    train_set = FeatureDataset(args.flat, train_idx)
    if args.balance_classes:
        codes = object_type_codes(args.flat)[train_idx]
        weights = sampling_weights(codes, damping=args.balance_damping)
        sampler = WeightedRandomSampler(
            weights.tolist(), num_samples=len(train_idx), replacement=True
        )
        train_loader = DataLoader(train_set, sampler=sampler, drop_last=True, **common)
        names = {index: name for name, index in OBJECT_TYPE_INDEX.items()}
        share = {names.get(int(c), "unknown"): f"{(codes == c).mean():.1%}"
                 for c in np.unique(codes)}
        print(f"balancing classes (damping {args.balance_damping}); raw share {share}",
              flush=True)
    else:
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    monitor_loader = DataLoader(FeatureDataset(args.flat, monitor_idx), **common)
    print(f"train {len(train_idx)}  monitor {len(monitor_idx)}  device {device}", flush=True)

    model = TrajectoryPredictor(dim=args.dim, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_steps = max(args.epochs * len(train_loader), 1)

    def learning_rate(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = args.checkpoint_dir / "latest.pt"
    best_path = args.checkpoint_dir / "best.pt"
    start_epoch, best_ade, history = 0, float("inf"), []
    if latest.exists():
        state = torch.load(latest, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_ade = state["best_ade"]
        history = state["history"]
        print(f"resumed from epoch {state['epoch']} (best minADE {best_ade:.3f} m)", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        started, running, batches = time.time(), 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                trajectories, logits = model(
                    batch["target_history"], batch["neighbors"], batch["map_points"],
                    batch["object_type"],
                )
                loss, _, _ = multimodal_loss(
                    trajectories.float(), logits.float(),
                    batch["future"], batch["future_mask"],
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            running += float(loss.detach())
            batches += 1

        metrics = evaluate(model, monitor_loader, device)
        metrics.update(
            epoch=epoch, loss=running / max(batches, 1),
            seconds=round(time.time() - started, 1),
            lr=schedule.get_last_lr()[0],
        )
        history.append(metrics)
        classes = "  ".join(
            f"{name[:3]} {metrics[key]:5.3f}"
            for name in ("vehicle", "pedestrian", "cyclist")
            if (key := f"min_ade_{name}") in metrics
        )
        print(
            f"epoch {epoch:3d}  loss {metrics['loss']:7.3f}  "
            f"minADE {metrics['min_ade_m']:6.3f} m  minFDE {metrics['min_fde_m']:6.3f} m  "
            f"[{classes}]  {metrics['seconds']:.0f}s",
            flush=True,
        )

        # Update the record before writing, so a resumed run does not forget the
        # best score achieved and re-crown a worse epoch.
        improved = metrics["min_ade_m"] < best_ade
        if improved:
            best_ade = metrics["min_ade_m"]

        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_ade": best_ade, "history": history,
            "config": vars(args) | {"features": str(args.features),
                                    "flat": str(args.flat),
                                    "checkpoint_dir": str(args.checkpoint_dir)},
        }
        torch.save(state, latest)
        if improved:
            torch.save(state, best_path)
            print(f"           new best minADE {best_ade:.3f} m", flush=True)

    (args.checkpoint_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"finished. best monitor minADE {best_ade:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
