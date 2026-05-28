"""Shared-policy actor and actor-critic models for IPPO / MAPPO."""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.obs_encoding import NUM_ACTIONS, OBS_DIM


class SharedActor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=NUM_ACTIONS, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)

    def forward(self, obs, action_mask):
        hidden = self.backbone(obs)
        logits = self.policy_head(hidden)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        return Categorical(logits=logits)


class ActorCritic(nn.Module):
    """Decentralized actor-critic (IPPO)."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=NUM_ACTIONS, hidden=128):
        super().__init__()
        self.actor = SharedActor(obs_dim, act_dim, hidden)
        self.value_head = nn.Linear(hidden, 1)
        self._hidden = hidden
        self._obs_dim = obs_dim

    def forward(self, obs, action_mask):
        hidden = self.actor.backbone(obs)
        logits = self.actor.policy_head(hidden)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        dist = Categorical(logits=logits)
        value = self.value_head(hidden).squeeze(-1)
        return dist, value

    def act(self, obs, action_mask, deterministic=False):
        dist, value = self.forward(obs, action_mask)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_action(self, obs, action_mask, action):
        """Log-probability and value for a fixed action index."""
        dist, value = self.forward(obs, action_mask)
        if not torch.is_tensor(action):
            action = torch.tensor(action, device=obs.device, dtype=torch.long)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        log_prob = dist.log_prob(action)
        return log_prob, value


class MAPPOActorCritic(nn.Module):
    """Shared decentralized actor + centralized team critic (MAPPO)."""

    def __init__(
        self,
        num_predators=3,
        obs_dim=OBS_DIM,
        act_dim=NUM_ACTIONS,
        hidden=128,
    ):
        super().__init__()
        self.num_predators = num_predators
        self.obs_dim = obs_dim
        self.joint_obs_dim = num_predators * obs_dim
        self.actor = SharedActor(obs_dim, act_dim, hidden)
        self.critic = nn.Sequential(
            nn.Linear(self.joint_obs_dim, hidden * 2),
            nn.Tanh(),
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def act(self, obs, action_mask, deterministic=False):
        dist = self.actor(obs, action_mask)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, obs, action_mask, action):
        dist = self.actor(obs, action_mask)
        if not torch.is_tensor(action):
            action = torch.tensor(action, device=obs.device, dtype=torch.long)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        return dist.log_prob(action)

    def evaluate_actions(self, obs, action_mask, actions):
        dist = self.actor(obs, action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def team_value(self, joint_obs):
        return self.critic(joint_obs).squeeze(-1)
