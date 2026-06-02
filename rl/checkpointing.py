from pathlib import Path

import torch

from rl.algo import IPPO, MAPPO
from rl.policy import ActorCritic, MAPPOActorCritic
from rl.ppo import PPOConfig


def create_policy(algo, num_predators=3):
    '''Creates a policy object based on the chosen algorithm. When using MAPPO, the policy is set for a number of agents, 3 by default.'''
    if algo == MAPPO:
        return MAPPOActorCritic(num_predators=num_predators)
    return ActorCritic()


def save_checkpoint(path, policy, optimizer, update, ppo_config, extra=None):
    '''Saves to disk a policy checkpoint, including optimizer state, current number of updates, PPO configuration and the algorithm used.'''
    # Content to be saved
    checkpoint = {
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "update": update,
        "ppo_config": ppo_config.__dict__,
        "algo": MAPPO if isinstance(policy, MAPPOActorCritic) else IPPO,
    }
    # If using MAPPO, save the number of agents too
    if isinstance(policy, MAPPOActorCritic):
        checkpoint["num_predators"] = policy.num_predators
    # If there are extra items to save, add them. This is used to save best evaluation score and whether search was used
    if extra:
        checkpoint.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path, device, policy=None, optimizer=None, algo=None, num_predators=3):
    '''Loads a policy checkpoint from disk'''
    
    # Try-except block to handle different versions of PyTorch where weights_only does not exist yet
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    # Extract PPO configuration, algorithm and number of agents from checkpoint
    ppo_config = PPOConfig(**checkpoint.get("ppo_config", {}))
    resolved_algo = checkpoint.get("algo", algo or IPPO)
    resolved_predators = int(checkpoint.get("num_predators", num_predators))

    # If policy is not directly provided, create a new one with the state from the checkpoint and move it to device
    if policy is None:
        policy = create_policy(resolved_algo, num_predators=resolved_predators)
    policy.load_state_dict(checkpoint["policy_state"])
    policy.to(device)
    # If optimizer is provided, load its state from the checkpoint
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return policy, ppo_config, checkpoint


def checkpoint_use_search(checkpoint):
    """Checks whether this checkpoint was trained with per-agent search heuristic."""
    return bool(checkpoint.get("use_search", False))
