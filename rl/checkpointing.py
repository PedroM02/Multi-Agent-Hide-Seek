"""Checkpoint save/load for IPPO and MAPPO policies."""

from pathlib import Path

import torch

from rl.algo import IPPO, MAPPO
from rl.policy import ActorCritic, MAPPOActorCritic
from rl.ppo import PPOConfig


def create_policy(algo, num_predators=3):
    if algo == MAPPO:
        return MAPPOActorCritic(num_predators=num_predators)
    return ActorCritic()


def save_checkpoint(path, policy, optimizer, update, ppo_cfg, extra=None):
    payload = {
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "update": update,
        "ppo_config": ppo_cfg.__dict__,
        "algo": MAPPO if isinstance(policy, MAPPOActorCritic) else IPPO,
    }
    if isinstance(policy, MAPPOActorCritic):
        payload["num_predators"] = policy.num_predators
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device, policy=None, optimizer=None, algo=None, num_predators=3):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)

    ppo_cfg = PPOConfig(**payload.get("ppo_config", {}))
    resolved_algo = payload.get("algo", algo or IPPO)
    resolved_predators = int(payload.get("num_predators", num_predators))

    if policy is None:
        policy = create_policy(resolved_algo, num_predators=resolved_predators)
    policy.load_state_dict(payload["policy_state"])
    policy.to(device)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return policy, ppo_cfg, payload


def checkpoint_use_search(payload):
    """Whether this checkpoint was trained with per-agent search heuristic."""
    return bool(payload.get("use_search", payload.get("hybrid_search", False)))
