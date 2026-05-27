"""Action selection helpers shared by training, evaluation, and GUI."""

import numpy as np
import torch

import agent_utils as au
from rl.algo import IPPO, MAPPO
from rl.obs_encoding import (
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    build_joint_predator_obs,
    encode_obs,
)
from rl.obs_encoding import action_mask as base_action_mask


def agents_by_id(sim):
    return {agent.agent_id: agent for agent in sim.agents}


def predator_slot_ids(sim):
    return sorted(
        body.agent_id
        for body in sim.env.agent_bodies.values()
        if body.team == au.TEAM_PREDATOR
    )


def predator_action_mask(sim, agent_id, raw_obs):
    """Legal actions for a predator; stunned predators may only STAY."""
    mask = base_action_mask(raw_obs)
    body = sim.env.agent_bodies[agent_id]
    if (
        sim.config.prey_defend == "stun"
        and body.team == au.TEAM_PREDATOR
        and body.stun_remaining > 0
    ):
        stay_only = np.zeros_like(mask)
        stay_only[ACTION_TO_IDX[au.STAY]] = True
        return stay_only
    return mask


def collect_predator_transitions(
    sim,
    policy,
    device,
    raw_obs,
    deterministic=False,
    algo=IPPO,
    slot_ids=None,
):
    """Sample predator actions from prebuilt observations."""
    agent_lookup = agents_by_id(sim)
    transitions = []
    predator_actions = {}
    team_value = None
    joint_obs = None

    if algo == MAPPO:
        slot_ids = slot_ids or predator_slot_ids(sim)
        joint_obs = build_joint_predator_obs(raw_obs, slot_ids, agent_lookup)
        joint_tensor = torch.tensor(
            joint_obs, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            team_value = float(policy.team_value(joint_tensor).item())

    for agent_id in sim.predator_agent_ids():
        raw = raw_obs[agent_id]
        agent = agent_lookup[agent_id]
        obs_vec = encode_obs(raw, agent)
        mask_vec = predator_action_mask(sim, agent_id, raw)
        obs = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(mask_vec, dtype=torch.bool, device=device).unsqueeze(0)

        if algo == MAPPO:
            action_idx, log_prob = policy.act(obs, mask, deterministic=deterministic)
        else:
            action_idx, log_prob, value = policy.act(
                obs, mask, deterministic=deterministic,
            )

        action_int = int(action_idx.item())
        predator_actions[agent_id] = IDX_TO_ACTION[action_int]
        transition = {
            "agent_id": agent_id,
            "obs": obs_vec,
            "mask": mask_vec,
            "action": action_int,
            "log_prob": float(log_prob.item()),
        }
        if algo != MAPPO:
            transition["value"] = float(value.item())
        transitions.append(transition)

    if algo == MAPPO:
        return predator_actions, transitions, team_value, joint_obs
    return predator_actions, transitions


def select_predator_actions(sim, policy, device, deterministic=False, algo=IPPO):
    """Build obs for alive predators and return action strings."""
    raw_obs = sim.build_step_observations()
    slot_ids = predator_slot_ids(sim) if algo == MAPPO else None
    result = collect_predator_transitions(
        sim,
        policy,
        device,
        raw_obs,
        deterministic=deterministic,
        algo=algo,
        slot_ids=slot_ids,
    )
    if algo == MAPPO:
        predator_actions, _transitions, _team_value, _joint_obs = result
    else:
        predator_actions, _transitions = result
    return predator_actions, raw_obs


def make_rl_config(base=None, **overrides):
    from simulation import SimulationConfig

    config = SimulationConfig() if base is None else base
    config.mode = au.MODE_RL
    config.comms = "both"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
