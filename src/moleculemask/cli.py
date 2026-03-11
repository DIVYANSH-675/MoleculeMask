import argparse
import json

import numpy as np
import torch

from . import __version__
from .io import seed_all, start_run, write_json, write_npz
from .model import MaskedMolecule, load_data, one_mask, synthetic


def score(model, arrays, mean, scale):
    positions, numbers, targets = map(torch.as_tensor, arrays)
    mask = one_mask(len(positions), positions.shape[1], random=False)
    model.eval()
    with torch.no_grad():
        prediction, properties = model(positions, numbers, mask)
        error = ((prediction - positions).square().sum(-1) * mask).sum() / mask.sum()
        mae = (properties * scale + mean - targets).abs().mean()
        visible = (1 - mask)[..., None]
        centroid = (positions * visible).sum(1, keepdim=True) / visible.sum(1, keepdim=True)
        baseline = (((positions - centroid).square().sum(-1) * mask).sum() / mask.sum()).sqrt()
    return dict(
        masked_position_rmse=float(error.sqrt()),
        property_mae=float(mae),
        visible_centroid_rmse=float(baseline),
        training_mean_property_mae=float((targets - mean).abs().mean()),
    )


def main():
    parser = argparse.ArgumentParser(description="MoleculeMask equivariant geometry prototype")
    parser.add_argument("--version", action="version", version="moleculemask " + __version__)
    parser.add_argument("command", choices=["generate", "train", "evaluate"])
    parser.add_argument("--data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--property-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--selection-metric",
        choices=["masked_position_rmse", "property_mae"],
        default="masked_position_rmse",
    )
    parser.add_argument("--epochs", type=int, default=12)
    a = parser.parse_args()
    if a.batch_size < 1:
        parser.error("batch-size must be positive")
    if not np.isfinite(a.learning_rate) or a.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if not np.isfinite(a.property_weight) or a.property_weight < 0:
        parser.error("property-weight must be finite and nonnegative")
    a.output = a.output or (
        "runs/report.json"
        if a.command == "evaluate"
        else "runs/data.npz"
        if a.command in ("generate", "bank")
        else "runs/train"
    )
    if a.count < 5 or a.epochs < 1:
        parser.error("count >= 5 and epochs >= 1 are required")
    seed_all(a.seed)
    arrays = load_data(a.data) if a.data else synthetic(a.count, seed=a.seed)
    if a.command == "generate":
        write_npz(a.output, positions=arrays[0], atomic_numbers=arrays[1], properties=arrays[2])
        return
    model = MaskedMolecule()
    if a.command == "evaluate":
        if not a.checkpoint:
            parser.error("evaluate requires --checkpoint")
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        write_json(a.output, score(model, arrays, checkpoint["mean"], checkpoint["scale"]))
        return
    output = start_run(a.output, vars(a))
    order = np.random.default_rng(a.seed).permutation(len(arrays[0]))
    cut = max(1, len(order) // 5)
    validation = tuple(x[order[:cut]] for x in arrays)
    training = tuple(torch.as_tensor(x[order[cut:]]) for x in arrays)
    positions, numbers, targets = training
    mean, scale = float(targets.mean()), max(float(targets.std(unbiased=False)), 1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    history = []
    best = float("inf")
    best_epoch = None
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        permutation = torch.randperm(len(positions))
        for ids in permutation.split(a.batch_size):
            mask = one_mask(len(ids), positions.shape[1])
            predicted, prop = model(positions[ids], numbers[ids], mask)
            geometry = ((predicted - positions[ids]).square().sum(-1) * mask).sum() / mask.sum()
            loss = (
                geometry
                + a.property_weight * (prop - (targets[ids] - mean) / scale).square().mean()
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        history.append(
            dict(epoch=epoch, loss=total / len(positions), **score(model, validation, mean, scale))
        )
        if history[-1][a.selection_metric] < best:
            best = history[-1][a.selection_metric]
            best_epoch = epoch
            torch.save(
                dict(model=model.state_dict(), mean=mean, scale=scale, config=vars(a), epoch=epoch),
                output / "best.pt",
            )
    write_json(
        output / "selection.json", dict(metric=a.selection_metric, value=best, epoch=best_epoch)
    )
    torch.save(
        dict(model=model.state_dict(), mean=mean, scale=scale, config=vars(a)), output / "model.pt"
    )
    write_json(output / "metrics.json", history)
    write_json(
        output / "split.json", dict(train=order[cut:].tolist(), validation=order[:cut].tolist())
    )
    print(json.dumps(history[-1]))
