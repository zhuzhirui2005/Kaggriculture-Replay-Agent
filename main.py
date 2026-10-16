"""Kaggle entry point, including the official loader without __file__."""

import sys
from pathlib import Path

_AGENT = None


def _root():
    return Path(_root.__code__.co_filename).resolve().parent


def agent(observation, configuration=None):
    global _AGENT
    if _AGENT is None:
        root = _root()
        sys.path.insert(0, str(root))
        from farm_policy.agent import Agent

        _AGENT = Agent(root / "weights/model.pt")
    return _AGENT(observation, configuration)
