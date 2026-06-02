import numpy as np
import torch

import constants as co
from rl.algo import IPPO, MAPPO
from rl.obs_encoding import ACTION_TO_IDX, IDX_TO_ACTION, build_joint_predator_obs, encode_obs
from rl.obs_encoding import action_mask as base_action_mask
from rl.team_search import any_agent_in_search, choose_search_action, update_predator_search_headings


def agents_by_id(run):
    '''Maps agents to their IDs'''
    return {agent.agent_id: agent for agent in run.agents}


def predator_slot_ids(run):
    '''Returns the ordered IDs for the predators in the run. Used to build joint observations'''
    return sorted(agent_body.agent_id for agent_body in run.env.agent_bodies.values() if agent_body.team == co.TEAM_PREDATOR)


def predator_action_mask(run, agent_id, raw_obs):
    """Masks legal actions for a predator; stunned predators may only STAY. Returns a boolean mask of legal actions.
       Example: [True, True, False, False, True] for the 5 actions available"""
    # Get base mask of legal actions according to the raw observations (environment)
    mask = base_action_mask(raw_obs)
    agent_body = run.env.agent_bodies[agent_id]
    # If the predator is stunned, further mask the actions to only allow STAY
    if (run.config.prey_defend == "stun" and agent_body.team == co.TEAM_PREDATOR and agent_body.stun_remaining > 0):
        stay_only = np.zeros_like(mask)
        stay_only[ACTION_TO_IDX[co.STAY]] = True
        return stay_only
    return mask


def collect_predator_transitions(run, policy, device, all_obs, search_controller, deterministic=False, algo=IPPO, slot_ids=None):
    """Samples predator actions according to policy. When enabled, agents may search instead"""
    agent_lookup = agents_by_id(run)
    predator_ids = run.env.alive_predator_ids()
    transitions = []
    predator_actions = {}
    team_value = None
    joint_obs = None

    # If search is enabled, check if agents are in search
    # If searching, update search direction if agent just entered search or allies were seen
    # If not searching, clear search direction
    if search_controller is not None:
        agent_in_search, agent_just_entered = search_controller.update(all_obs, predator_ids)
        update_predator_search_headings(agent_lookup, all_obs, predator_ids, search_controller.search_logic, agent_in_search, agent_just_entered)
    # If search is not enabled, set all agents to not in search
    else:
        agent_in_search = {agent_id: False for agent_id in predator_ids}

    # If running MAPPO, centralized critic is used to estimate the value of the team
    if algo == MAPPO:
        slot_ids = slot_ids or predator_slot_ids(run)
        # Build joint observation for the team
        joint_obs = build_joint_predator_obs(all_obs, slot_ids, agent_lookup)
        joint_tensor = torch.tensor(joint_obs, dtype=torch.float32, device=device).unsqueeze(0)
        # Evaluate the value of the joint state using the centralized critic
        with torch.no_grad():
            team_value = float(policy.team_value(joint_tensor).item())

    # Collect transition for each predator
    for agent_id in predator_ids:
        raw = all_obs[agent_id]
        agent = agent_lookup[agent_id]
        # If agent is in search, get a search action instead of a policy action
        if agent_in_search.get(agent_id, False):
            action_str = choose_search_action(search_controller.search_logic, agent, raw)
            predator_actions[agent_id] = action_str
            continue
        # If agent is not in search, encode observation and mask legal actions
        obs_vec = encode_obs(raw, agent)
        mask_vec = predator_action_mask(run, agent_id, raw)
        obs = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(mask_vec, dtype=torch.bool, device=device).unsqueeze(0)

        # Get predator action from policy
        if algo == MAPPO:
            action_idx, log_prob = policy.act(obs, mask, deterministic=deterministic)
            action_int = int(action_idx.item())
            transition = {
                "agent_id": agent_id,
                "obs": obs_vec,
                "mask": mask_vec,
                "action": action_int,
                "log_prob": float(log_prob.item()),
            }
        else:
            action_idx, log_prob, value = policy.act(obs, mask, deterministic=deterministic)
            action_int = int(action_idx.item())
            transition = {
                "agent_id": agent_id,
                "obs": obs_vec,
                "mask": mask_vec,
                "action": action_int,
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
            }
        predator_actions[agent_id] = IDX_TO_ACTION[action_int]
        transitions.append(transition)
    # Check if any agent is in search to adapt rewards
    search_mode = any_agent_in_search(agent_in_search)
    if algo == MAPPO:
        return predator_actions, transitions, team_value, joint_obs, search_mode
    return predator_actions, transitions, search_mode


def select_predator_actions(run, policy, device, search_controller, deterministic=False, algo=IPPO):
    """Build observation for predators and return actions chosen by the policy"""
    # Build observations per agent
    all_obs = run.build_step_observations()
    slot_ids = predator_slot_ids(run) if algo == MAPPO else None
    # Build joint observations, collect transitions and actions selected by the policy
    result = collect_predator_transitions(run, policy, device, all_obs, search_controller, deterministic=deterministic, algo=algo, slot_ids=slot_ids)
    # Different algorithms return different tuples
    if algo == MAPPO:
        predator_actions, transitions, team_value, joint_obs, search_mode = result
    else:
        predator_actions, transitions, search_mode = result
    return predator_actions, all_obs


def make_rl_config(base=None, **overrides):
    '''Creats RL simulation configuration with possible overrides'''
    from simulation import SimulationConfig

    config = SimulationConfig() if base is None else base
    config.mode = co.MODE_RL
    config.comms = "both"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
