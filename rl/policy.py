"""Shared actor-critic policy for IPPO."""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.obs_encoding import NUM_ACTIONS, OBS_DIM


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=NUM_ACTIONS, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs, action_mask):
        hidden = self.backbone(obs)
        logits = self.policy_head(hidden)
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
