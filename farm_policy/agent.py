"""Stateful inference using the same features and action decoder as training."""

from copy import deepcopy

import numpy as np
import torch

from . import encoding as E
from .quota_policy import load
from .terminal import liquidate


class Agent:
    def __init__(self, checkpoint, style=None, device="cpu", terminal=True):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.model, self.metadata = load(checkpoint, self.device)
        self.model.eval()
        styles = list(self.model.config.styles)
        selected = (
            style
            if style is not None
            else self.metadata.get("default_style", styles[0])
        )
        if selected not in styles:
            raise ValueError("Style was not trained: " + str(selected))
        self.style = styles.index(selected)
        self.terminal = terminal
        self.history = {}

    @torch.inference_mode()
    def __call__(self, observation, configuration=None):
        seat = observation["player"]
        step = int(observation.get("step", 0))
        old = self.history.get(seat)
        if old and old["obs"] == observation and old["config"] == configuration:
            return deepcopy(old["action"])
        if step == 0 or old and old["obs"]["step"] >= step:
            old = None
        features = E.encode(
            observation,
            configuration,
            old["action"] if old else None,
            old["obs"] if old else None,
        )
        batch = {
            k: torch.from_numpy(np.asarray(v)[None]).to(self.device)
            for k, v in features.items()
        }
        batch["style"] = torch.tensor([self.style], device=self.device)
        if old and observation["day"] == old["obs"]["day"]:
            batch.update(old["memory"])
        output = self.model(batch, deterministic=True)
        action = E.decode(
            *(
                output[k][0].cpu().numpy()
                for k in ("unit", "quantity", "market", "market_quantity")
            ),
            int(features["count"]),
        )
        if self.terminal:
            action = liquidate(observation, configuration, action)
        self.history[seat] = dict(
            obs=deepcopy(observation),
            config=deepcopy(configuration),
            action=deepcopy(action),
            memory={k: output[k] for k in ("memory_global", "memory_actor")},
        )
        return action
