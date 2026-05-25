"""Action selection helpers shared by training, evaluation, and GUI."""

import numpy as np
import torch

import agent_utils as au
from rl.obs_encoding import IDX_TO_ACTION, action_mask, encode_obs


def agents_by_id(sim):
    return {agent.agent_id: agent for agent in sim.agents}


def collect_predator_transitions(sim, policy, device, raw_obs, deterministic=False):
    """Sample actions from prebuilt observations."""
    agent_lookup = agents_by_id(sim)
    transitions = []
    predator_actions = {}

    for agent_id in sim.predator_agent_ids():
        raw = raw_obs[agent_id]
        agent = agent_lookup[agent_id]
        obs_vec = encode_obs(raw, agent)
        mask_vec = action_mask(raw)
        obs = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(mask_vec, dtype=torch.bool, device=device).unsqueeze(0)

        action_idx, log_prob, value = policy.act(
            obs, mask, deterministic=deterministic,
        )
        action_int = int(action_idx.item())
        predator_actions[agent_id] = IDX_TO_ACTION[action_int]
        transitions.append(
            {
                "agent_id": agent_id,
                "obs": obs_vec,
                "mask": mask_vec,
                "action": action_int,
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
            }
        )

    return predator_actions, transitions


def select_predator_actions(sim, policy, device, deterministic=False):
    """Build obs for alive predators and return action strings."""
    raw_obs = sim.build_step_observations()
    predator_actions, _ = collect_predator_transitions(
        sim, policy, device, raw_obs, deterministic=deterministic,
    )
    return predator_actions, raw_obs


def make_rl_config(base=None, **overrides):
    from simulation import SimulationConfig

    config = SimulationConfig() if base is None else base
    config.mode = au.MODE_RL
    config.comms = "both"
    config.prey_defend = None
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
