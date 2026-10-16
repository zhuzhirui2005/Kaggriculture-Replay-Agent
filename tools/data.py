"""Offline preparation of compact, episode-disjoint replay training arrays."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from farm_policy import encoding as E

DECISIONS = 719
DAYS = 30
HOURS = 24
TARGETS = ("unit", "quantity", "market", "market_quantity", "goal_cell", "goal_command")


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def partition(episode):
    # Both seats ALWAYS use the same split. IDs must identify games, not players.
    bucket = (
        int(
            hashlib.sha256(f"kaggriculture-learn-v1:{episode}".encode()).hexdigest()[
                :8
            ],
            16,
        )
        % 100
    )
    return "test" if bucket < 10 else "validation" if bucket < 20 else "train"


def observation(replay, step, seat):
    row = replay["steps"][step]
    obs = dict(row[0]["observation"])
    obs.update(row[seat]["observation"])
    obs.update(player=seat, step=step)
    return obs


def encode_episode(task):
    row, root, output = task
    replay = read_json(Path(root) / row["file"])
    steps = replay["steps"]
    if len(steps) != 720 or [p["status"] for p in steps[-1]] != ["DONE", "DONE"]:
        raise ValueError(f"{row['id']}: expected a completed standard 720-state replay")
    configuration = replay["configuration"]
    if configuration.get("turnsPerDay", 24) != 24:
        raise ValueError("This sequence format uses 24-turn days")
    entries = []
    for seat_string, style in sorted(row["players"].items()):
        seat = int(seat_string)
        samples, past, previous = [], None, None
        final = steps[-1][seat]["reward"] - steps[-1][1 - seat]["reward"]
        for step in range(DECISIONS):
            obs = observation(replay, step, seat)
            # Replay actions live in the NEXT state, not the observed state.
            action = steps[step + 1][seat]["action"] or {}
            features = E.encode(obs, configuration, past, previous)
            labels = E.labels(obs, action)
            for actor in range(int(features["count"])):
                if not features["legal"][actor, labels["unit"][actor]]:
                    labels["unit"][actor] = labels["quantity"][actor] = -100
            margin = obs["farms"][seat]["money"] - obs["farms"][1 - seat]["money"]
            target = np.asarray(
                (final - margin) / 10000 + 2 * np.sign(final), np.float32
            )
            samples.append(dict(**features, **labels, **{"return": target}))
            past, previous = action, obs
        arrays = {key: np.stack([s[key] for s in samples]) for key in samples[0]}
        for key in ("board", "global_features", "actors"):
            arrays[key] = arrays[key].astype(np.float16)
        for key in (
            "positions",
            "count",
            "previous_unit",
            "previous_quantity",
            "previous_market",
            "previous_market_quantity",
        ):
            arrays[key] = arrays[key].astype(np.uint8)
        cells = np.full((DECISIONS, E.MAX_ACTORS), -100, np.int16)
        commands = cells.copy()
        for day in range(DAYS):
            pending = {}
            for step in range(
                min(DECISIONS - 1, day * HOURS + HOURS - 1), day * HOURS - 1, -1
            ):
                for actor in range(int(arrays["count"][step])):
                    unit = int(arrays["unit"][step, actor])
                    if unit >= 0 and E.UNIT[unit][0] not in (
                        "PASS",
                        "NORTH",
                        "SOUTH",
                        "EAST",
                        "WEST",
                    ):
                        pending[actor] = (step, arrays["positions"][step, actor], unit)
                    if actor in pending and pending[actor][0] - step <= 12:
                        _, cells[step, actor], commands[step, actor] = pending[actor]
        arrays.update(goal_cell=cells, goal_command=commands)
        stem = hashlib.sha256(str(row["id"]).encode()).hexdigest()[:24]
        target = Path(output) / f"{stem}-{seat}.npz"
        temporary = target.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(target)
        entries.append(
            dict(
                episode=str(row["id"]),
                seat=seat,
                style=style,
                group=row.get("group", "new"),
                split=partition(row["id"]),
                file=target.name,
            )
        )
    return entries


def prepare(manifest_path, output, workers):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    episodes = read_json(manifest_path)["episodes"]
    seen, files, tasks = set(), set(), []
    for row in episodes:
        identifier = str(row["id"])
        source = (manifest_path.parent / row["file"]).resolve(strict=True)
        if identifier in seen or source in files:
            raise ValueError(
                "List each replay once, with all selected seats in players"
            )
        if not row["players"] or not set(row["players"]) <= {"0", "1"}:
            raise ValueError('players maps seat "0" and/or "1" to a style label')
        if any(not isinstance(s, str) or not s for s in row["players"].values()):
            raise ValueError("Style labels must be nonempty strings")
        if row.get("group", "new") not in ("new", "rehearsal"):
            raise ValueError("group must be new or rehearsal")
        seen.add(identifier)
        files.add(source)
        tasks.append((row, str(manifest_path.parent), str(output)))
    if not tasks or workers < 1:
        raise ValueError("Provide replays and a positive worker count")
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for completed, result in enumerate(pool.map(encode_episode, tasks), 1):
            entries.extend(result)
            print(f"Encoded {completed}/{len(tasks)} episodes", flush=True)
    write_json(output / "index.json", dict(schema=E.SCHEMA, entries=entries))


def cache(source, output):
    """Expand compressed shards once; all training ranks share read-only mappings."""
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty cache directory")
    index = read_json(source / "index.json")
    if index["schema"] != E.SCHEMA:
        raise ValueError("Incompatible feature schema")
    output.mkdir(parents=True, exist_ok=True)
    partitions = {}
    for split in ("train", "validation"):
        entries = [r for r in index["entries"] if r["split"] == split]
        if not entries:
            raise ValueError(f"No {split} episodes; supply more independent games")
        folder = output / split
        folder.mkdir()
        with np.load(source / entries[0]["file"], allow_pickle=False) as first:
            arrays = {
                k: np.lib.format.open_memmap(
                    folder / f"{k}.npy",
                    mode="w+",
                    dtype=first[k].dtype,
                    shape=(len(entries) * DECISIONS, *first[k].shape[1:]),
                )
                for k in first.files
            }
        for i, row in enumerate(entries):
            with np.load(source / row["file"], allow_pickle=False) as shard:
                for key, target in arrays.items():
                    target[i * DECISIONS : (i + 1) * DECISIONS] = shard[key]
        for array in arrays.values():
            array.flush()
        partitions[split] = dict(entries=entries, arrays=list(arrays))
        del arrays
    # Test games stay in the compressed dataset, never in the training cache.
    write_json(output / "index.json", dict(schema=E.SCHEMA, partitions=partitions))


class Days:
    def __init__(self, path, split, styles):
        path = Path(path)
        index = read_json(path / "index.json")
        if index["schema"] != E.SCHEMA:
            raise ValueError("Incompatible feature schema")
        part = index["partitions"][split]
        self.entries = part["entries"]
        self.arrays = {
            key: np.load(path / split / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in part["arrays"]
        }
        self.style_indices = np.asarray(
            [styles.index(row["style"]) for row in self.entries], np.int64
        )
        self.size = len(self.entries) * DAYS

    def batch(self, indices, device):
        import torch

        episodes, days = np.asarray(indices) // DAYS, np.asarray(indices) % DAYS
        within = days[:, None] * HOURS + np.arange(HOURS)[None]
        frames = episodes[:, None] * DECISIONS + np.minimum(within, DECISIONS - 1)
        result = {
            key: torch.from_numpy(np.ascontiguousarray(value[frames])).to(device)
            for key, value in self.arrays.items()
        }
        valid = torch.from_numpy(within < DECISIONS).to(device)
        for key in TARGETS:
            result[key] = result[key].masked_fill(~valid[..., None], -100)
        # Retains the reference recipe: the last return target repeats once.
        result["style"] = torch.from_numpy(self.style_indices[episodes]).to(device)
        return result


class DaySampler:
    """Balance new demonstration styles; optionally retain rehearsal trajectories."""

    def __init__(self, entries, fresh_fraction):
        fresh, rehearsal = defaultdict(list), []
        for i, row in enumerate(entries):
            if row.get("group", "new") == "rehearsal":
                rehearsal.append(i)
            else:
                fresh[row["style"]].append(i)
        self.fresh = [np.asarray(fresh[s]) for s in sorted(fresh)]
        self.rehearsal = np.asarray(rehearsal)
        self.fraction = fresh_fraction

    def sample(self, rng, size):
        result = np.empty(size, np.int64)
        for i in range(size):
            use_fresh = self.fresh and (
                not len(self.rehearsal) or rng.random() < self.fraction
            )
            choices = (
                self.fresh[int(rng.integers(len(self.fresh)))]
                if use_fresh
                else self.rehearsal
            )
            result[i] = int(rng.choice(choices)) * DAYS + int(rng.integers(DAYS))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("cache")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output, args.workers)
    else:
        cache(args.source, args.output)


if __name__ == "__main__":
    main()
