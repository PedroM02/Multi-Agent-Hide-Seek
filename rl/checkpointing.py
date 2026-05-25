"""Checkpoint save/load for shared-policy IPPO."""

from pathlib import Path

import torch

from rl.policy import ActorCritic
from rl.ppo import PPOConfig


def save_checkpoint(path, policy, optimizer, update, ppo_cfg, extra=None):
    payload = {
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "update": update,
        "ppo_config": ppo_cfg.__dict__,
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device, policy=None, optimizer=None):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    ppo_cfg = PPOConfig(**payload.get("ppo_config", {}))
    if policy is None:
        policy = ActorCritic()
    policy.load_state_dict(payload["policy_state"])
    policy.to(device)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return policy, ppo_cfg, payload
