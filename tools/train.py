"""Offline full-policy imitation, on one device or synchronized GPU workers."""

import argparse
import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from farm_policy import encoding as E
from farm_policy.coordinated import CoordinatedConfig
from farm_policy.model import supervised_loss
from farm_policy.quota_policy import QuotaPolicy, save
from tools.data import TARGETS, Days, DaySampler, read_json, write_json

HEADS = (*TARGETS, "value")
COEFFICIENTS = (1, 0.5, 1, 0.5, 0.2, 0.2, 0.1)


def initialize(styles, args, state=None):
    config = (
        state["config"].copy()
        if state
        else dict(
            width=args.width,
            hidden=args.hidden,
            memory=args.memory,
            blocks=args.blocks,
            attention_layers=args.attention_layers,
        )
    )
    vocabulary = list(config.get("styles", ()))
    vocabulary += sorted(set(styles) - set(vocabulary))
    config["styles"] = tuple(vocabulary)
    model = QuotaPolicy(CoordinatedConfig(**config))
    if state:
        if (
            state.get("architecture") != "quota-coordinated-v1"
            or state.get("schema") != E.SCHEMA
        ):
            raise ValueError("Expected a compatible quota-policy checkpoint")
        weights = dict(state["state_dict"])
        old = weights["style.weight"]
        expanded = model.style.weight.detach().clone()
        expanded[: len(old)] = old
        weights["style.weight"] = expanded
        model.load_state_dict(weights, strict=True)
    return model, vocabulary


def erase_history(batch, probability, rng):
    if not probability:
        return batch
    batch = dict(batch)
    workers = torch.from_numpy(
        rng.random(batch["previous_unit"].shape) < probability
    ).to(batch["unit"].device)
    market = torch.from_numpy(
        rng.random(batch["previous_market"].shape[:-1]) < probability
    ).to(batch["unit"].device)
    for key in ("previous_unit", "previous_quantity"):
        batch[key] = batch[key].masked_fill(workers, 0)
    for key in ("previous_market", "previous_market_quantity"):
        batch[key] = batch[key].masked_fill(market[..., None], 0)
    return batch


def loss_counts(output, flat):
    weights = output["value"].new_ones(len(E.UNIT))
    weights[0] = 0.5
    for i, command in enumerate(E.UNIT):
        if command[0] in (
            "PLANT",
            "BUILD_COOP",
            "BUILD_PASTURE",
            "PLACE",
            "PICKUP",
            "DIG",
        ):
            weights[i] = 2
    valid = flat["unit"] != -100
    return torch.stack(
        [
            weights[flat["unit"][valid].long()].sum(),
            *[(flat[key] != -100).sum().float() for key in TARGETS[1:]],
            output["value"].new_tensor(flat["return"].numel()),
        ]
    )


class SequenceLoss(nn.Module):
    def __init__(self, policy, world):
        super().__init__()
        self.policy, self.world = policy, world

    def forward(self, batch):
        output, flat = self.policy.sequence(batch)
        _, losses = supervised_loss(output, flat)
        counts = loss_counts(output, flat)
        total = counts.detach().clone()
        if self.world > 1:
            dist.all_reduce(total)
        # DDP averages gradients, but valid label counts differ across ranks.
        return sum(
            losses[key]
            * counts[i]
            / total[i].clamp_min(1)
            * self.world
            * COEFFICIENTS[i]
            for i, key in enumerate(HEADS)
        )


def precision(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.no_grad()
def evaluate(model, days, batch_size, rank, world, device):
    model.eval()
    sums = torch.zeros(26, dtype=torch.float64, device=device)
    for start in range(rank * batch_size, days.size, batch_size * world):
        batch = days.batch(np.arange(start, min(start + batch_size, days.size)), device)
        with precision(device):
            output, flat = model.sequence(batch)
            _, losses = supervised_loss(output, flat)
        counts = loss_counts(output, flat)
        for i, key in enumerate(HEADS):
            sums[2 * i] += losses[key].double() * counts[i]
            sums[2 * i + 1] += counts[i]
        for i, key in enumerate(TARGETS):
            valid = flat[key] != -100
            sums[14 + 2 * i] += (
                (output[key + "_logits"].argmax(-1) == flat[key]) & valid
            ).sum()
            sums[15 + 2 * i] += valid.sum()
    if world > 1:
        dist.all_reduce(sums)
    numbers = sums.cpu().tolist()
    return dict(
        loss=sum(
            numbers[2 * i] / max(1, numbers[2 * i + 1]) * COEFFICIENTS[i]
            for i in range(7)
        ),
        accuracy={
            key: numbers[14 + 2 * i] / max(1, numbers[15 + 2 * i])
            for i, key in enumerate(TARGETS)
        },
    )


def atomic_save(path, value):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parent = parser.add_mutually_exclusive_group()
    parent.add_argument(
        "--warmstart",
        type=Path,
        help="Continue weights with a new optimizer; new styles may be appended",
    )
    parent.add_argument(
        "--resume",
        type=Path,
        help="Resume a training.pt including Adam and random states",
    )
    parser.add_argument(
        "--style",
        help="Default own-style label for deployment; fixed for a complete game",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--steps",
        type=int,
        default=12000,
        help="Total optimizer updates, including resumed updates",
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--batch-days",
        type=int,
        default=8,
        help="Global batch across all ranks; each sequence has 24 frames",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Defaults to 1e-4 from scratch or 1e-5 for warm starts",
    )
    parser.add_argument("--history-probability", type=float, default=0.15)
    parser.add_argument("--fresh-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--width", type=int, default=192, help="Scratch model only")
    parser.add_argument("--hidden", type=int, default=768, help="Scratch model only")
    parser.add_argument("--memory", type=int, default=256, help="Scratch model only")
    parser.add_argument("--blocks", type=int, default=3, help="Scratch model only")
    parser.add_argument(
        "--attention-layers", type=int, default=2, help="Scratch model only"
    )
    args = parser.parse_args()
    rank, world, local = (
        int(os.environ.get(key, default))
        for key, default in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))
    )
    if (
        args.batch_days < world
        or args.batch_days % world
        or min(args.steps, args.save_every) < 1
    ):
        raise ValueError(
            "Positive steps and a batch divisible by the worker count are required"
        )
    if not 0 <= args.history_probability <= 1 or not 0 < args.fresh_fraction < 1:
        raise ValueError("Invalid history or rehearsal probability")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda", local) if use_cuda else torch.device("cpu")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "This GPU recipe requires bfloat16 support; use --device cpu otherwise"
            )
    if world > 1:
        dist.init_process_group("nccl" if use_cuda and os.name != "nt" else "gloo")
    try:
        run(args, rank, world, local, device)
    finally:
        if world > 1:
            dist.destroy_process_group()


def run(args, rank, world, local, device):
    index = read_json(args.cache / "index.json")
    declared = sorted(
        {r["style"] for p in index["partitions"].values() for r in p["entries"]}
    )
    source = args.resume or args.warmstart
    state = (
        torch.load(source, map_location="cpu", weights_only=True) if source else None
    )
    model, styles = initialize(declared, args, state)
    model.to(device)
    default_style = args.style or (state or {}).get("default_style") or styles[0]
    if default_style not in styles:
        raise ValueError("Unknown deployment style")
    train = Days(args.cache, "train", styles)
    validation = Days(args.cache, "validation", styles)
    sampler = DaySampler(train.entries, args.fresh_fraction)
    rng = np.random.default_rng(args.seed)
    history_rng = np.random.default_rng(args.seed + 10000 + rank)
    lr = args.learning_rate or (1e-5 if source else 1e-4)
    if lr <= 0:
        raise ValueError("Learning rate must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    settings = dict(
        batch_days=args.batch_days,
        world=world,
        seed=args.seed,
        history_probability=args.history_probability,
        fresh_fraction=args.fresh_fraction,
        cache_sha256=hashlib.sha256(
            (args.cache / "index.json").read_bytes()
        ).hexdigest(),
        styles=styles,
    )
    start = 0
    if args.resume:
        if state["settings"] != settings:
            raise ValueError(
                "Resume requires the same cache, styles, batch size, ranks and sampling recipe"
            )
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["sampler_rng"]
        saved_rng = state["rank_rng"][rank]
        history_rng.bit_generator.state = saved_rng["history"]
        torch.set_rng_state(saved_rng["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved_rng["cuda"], device)
        start = state["step"]
    if args.steps <= start:
        raise ValueError("Total steps must exceed the resumed update number")
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise FileExistsError(
                "Use a fresh output directory, or --resume its training.pt"
            )
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "config.json",
            dict(
                settings,
                learning_rate=optimizer.param_groups[0]["lr"],
                default_style=default_style,
                model=model.config.__dict__,
            ),
        )
    if world > 1:
        dist.barrier()
    learner = SequenceLoss(model, world)
    if world > 1:
        learner = DistributedDataParallel(
            learner,
            device_ids=[local] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    size, begun = args.batch_days // world, time.monotonic()
    for step in range(start + 1, args.steps + 1):
        model.train()
        # Identical global sampling stream; each rank consumes its own slice.
        selected = sampler.sample(rng, args.batch_days)[rank * size : (rank + 1) * size]
        batch = erase_history(
            train.batch(selected, device), args.history_probability, history_rng
        )
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            loss = learner(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        stop = torch.tensor(int((args.output / "STOP").exists()), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if step % 50 == 0 or step == start + 1:
            mean = loss.detach().clone()
            if world > 1:
                dist.all_reduce(mean)
                mean /= world
            if rank == 0:
                print(f"Update {step}/{args.steps}: loss {mean.item():.4f}", flush=True)
        if step % args.save_every == 0 or step == args.steps or stop.item():
            metrics = (
                {}
                if stop.item()
                else evaluate(model, validation, size, rank, world, device)
            )
            local_rng = dict(
                history=history_rng.bit_generator.state,
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
            )
            states = [local_rng]
            if world > 1:
                states = [None] * world
                dist.all_gather_object(states, local_rng)
            if rank == 0:
                metadata = dict(default_style=default_style)
                save(args.output / f"step-{step:06d}.pt", model, **metadata)
                atomic_save(
                    args.output / "training.pt",
                    dict(
                        architecture="quota-coordinated-v1",
                        schema=E.SCHEMA,
                        config=model.config.__dict__,
                        state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        step=step,
                        settings=settings,
                        sampler_rng=rng.bit_generator.state,
                        rank_rng=states,
                        **metadata,
                    ),
                )
                row = dict(
                    step=step,
                    seconds=time.monotonic() - begun,
                    validation=metrics,
                    stopped=bool(stop.item()),
                )
                with (args.output / "metrics.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if world > 1:
                dist.barrier()
        if stop.item():
            break


if __name__ == "__main__":
    main()
