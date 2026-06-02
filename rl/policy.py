import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.obs_encoding import NUM_ACTIONS, OBS_DIM


class SharedActor(nn.Module):
    '''Shared actor network for both IPPO and MAPPO. MLP with two hidden layers'''
    def __init__(self, obs_dim=OBS_DIM, act_dim=NUM_ACTIONS, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            # Hidden layer (maps input to hidden neurons)
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            # Hidden layer (maps first hidden neurons to second hidden neurons)
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Output layer (maps second hidden neurons to output)
        self.policy_head = nn.Linear(hidden, act_dim)

    def forward(self, obs, action_mask):
        hidden = self.backbone(obs)
        logits = self.policy_head(hidden)
        # If mask is provided, mask out illegal actions (logits of illegal actions set to -1e8)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        # Return probability distribution over actions
        return Categorical(logits=logits)


class ActorCritic(nn.Module):
    """Decentralized actor-critic for IPPO with single shared policy"""

    def __init__(self, obs_dim=OBS_DIM, act_dim=NUM_ACTIONS, hidden=128):
        super().__init__()
        self.actor = SharedActor(obs_dim, act_dim, hidden)
        # Value head - single linear layer to output value estimation
        self.value_head = nn.Linear(hidden, 1)
        self._hidden = hidden
        self._obs_dim = obs_dim

    def forward(self, obs, action_mask):
        hidden = self.actor.backbone(obs)
        logits = self.actor.policy_head(hidden)
        # If mask is provided, mask out illegal actions (logits of illegal actions set to -1e8)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        # Return probability distribution over actions
        dist = Categorical(logits=logits)
        # Return value estimation
        value = self.value_head(hidden).squeeze(-1)
        return dist, value

    def act(self, obs, action_mask, deterministic=False):
        # Get probability distribution over actions and value estimation
        dist, value = self.forward(obs, action_mask)
        # Output action from policy distribution depending on determinism
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        # Get log-probability of the action
        log_prob = dist.log_prob(action)
        return action, log_prob, value




class MAPPOActorCritic(nn.Module):
    """Decentralized actor network+ centralized team critic for MAPPO.
       Critic is an MLP with two hidden layers"""

    def __init__(self, num_predators=3, obs_dim=OBS_DIM, act_dim=NUM_ACTIONS, hidden=128):
        super().__init__()
        self.num_predators = num_predators
        self.obs_dim = obs_dim
        self.joint_obs_dim = num_predators * obs_dim
        self.actor = SharedActor(obs_dim, act_dim, hidden)
        # Critic network
        self.critic = nn.Sequential(
            # Hidden layer (maps joint observation to hidden neurons)
            nn.Linear(self.joint_obs_dim, hidden * 2),
            nn.Tanh(),
            # Hidden layer (maps hidden neurons to second hidden neurons)
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            # Output layer (maps second hidden neurons to output)
            nn.Linear(hidden, 1),
        )

    def act(self, obs, action_mask, deterministic=False):
        # Get probability distribution over actions
        dist = self.actor(obs, action_mask)
        # Output action from policy distribution depending on determinism
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        # Get log-probability of the action
        log_prob = dist.log_prob(action)
        return action, log_prob


    def evaluate_actions(self, obs, action_mask, actions):
        '''Returns log-probabilities (log pi(a|s)) and entropies for a batch of actions'''
        # Get probability distributions over actions
        dist = self.actor(obs, action_mask)
        # Get log-probabilities of the actions
        log_probs = dist.log_prob(actions)
        # Get entropies of each distribution
        entropy = dist.entropy()
        return log_probs, entropy

    def team_value(self, joint_obs):
        '''Returns team-wise value estimation from critic'''
        return self.critic(joint_obs).squeeze(-1)
