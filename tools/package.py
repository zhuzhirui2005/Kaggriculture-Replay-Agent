"""Offline builder for a standalone CPU submission; never uploads anything."""

import argparse
import tarfile
from pathlib import Path

from farm_policy.quota_policy import load

RUNTIME = (
    "__init__.py",
    "agent.py",
    "encoding.py",
    "model.py",
    "recurrent.py",
    "coordinated.py",
    "quota_policy.py",
    "terminal.py",
    "worker_rules.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("submission.tar.gz"))
    args = parser.parse_args()
    model, state = load(args.weights)
    if state.get("default_style") not in model.config.styles:
        raise ValueError("The checkpoint needs a valid default_style for deployment")
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        archive.add(root / "main.py", arcname="main.py")
        archive.add(root / "LICENSE", arcname="LICENSE")
        for name in RUNTIME:
            archive.add(root / "farm_policy" / name, arcname="farm_policy/" + name)
        archive.add(args.weights, arcname="weights/model.pt")
    if args.output.stat().st_size > 100 * 2**20:
        raise ValueError("Archive exceeds 100 MiB")
    print(f"{args.output}: {args.output.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
