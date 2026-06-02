import numpy as np
import torch
import torch.nn as nn


class PPOConfig:
    '''PPO configuration class'''
    def __init__(
        self,
        rollout_steps = 4096,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_eps = 0.2,
        ppo_epochs = 4,
        minibatch_size = 256,
        lr = 3e-4,
        entropy_coef = 0.02,
        entropy_floor = 0.4,
        entropy_floor_coef = 0.05,
        value_coef = 0.5,
        max_grad_norm = 0.5,
    ):
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.entropy_floor = entropy_floor
        self.entropy_floor_coef = entropy_floor_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm


class RolloutBuffer:
    '''Stores batch of transitions' information for IPPO training, default up to 4096 transitions'''
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.rewards)

    def add(self, obs, mask, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, config):
        '''Computes returns and advantages for IPPO training using Generalized Advantage Estimation (GAE).
           Dones indicate whether the run ended at the given step'''
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = float(last_value)
        for step in reversed(range(len(rewards))):
            # If the run ended at the given step, bootstrap values are set to 0
            non_terminal = 1.0 - dones[step]
            # TD delta calculation (delta = r + gamma * V(s_{t+1}) - V(s_t))
            delta = (rewards[step] + config.gamma * next_value * non_terminal - values[step])
            # Advantage calculation (A_t = delta + gamma * lambda * A_{t+1})
            last_gae = delta + config.gamma * config.gae_lambda * non_terminal * last_gae
            advantages[step] = last_gae
            next_value = values[step]

        returns = advantages + values
        return returns, advantages


class MAPPOBuffer:
    '''Stores batch of transitions' information for MAPPO training, default up to 4096 transitions'''

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.step_indices = []
        self.joint_obs = []
        self.team_values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.obs)

    def add_step(self, joint_obs, team_value, reward, done, transitions):
        step_index = len(self.rewards)
        self.joint_obs.append(joint_obs)
        self.team_values.append(team_value)
        self.rewards.append(reward)
        self.dones.append(done)
        for transition in transitions:
            self.obs.append(transition["obs"])
            self.masks.append(transition["mask"])
            self.actions.append(transition["action"])
            self.log_probs.append(transition["log_prob"])
            self.step_indices.append(step_index)

    def compute_team_returns_and_advantages(self, last_value, config):
        '''Computes returns and advantages for MAPPO training using Generalized Advantage Estimation (GAE).
           Dones indicate whether the run ended at the given step.
           Values used are team-wise values'''
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.team_values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = float(last_value)
        for step in reversed(range(len(rewards))):
            # If the run ended at the given step, bootstrap values are set to 0
            non_terminal = 1.0 - dones[step]
            # TD delta calculation (delta = r + gamma * V(s_{t+1}) - V(s_t))
            delta = (rewards[step] + config.gamma * next_value * non_terminal - values[step])
            # Advantage calculation (A_t = delta + gamma * lambda * A_{t+1})
            last_gae = delta + config.gamma * config.gae_lambda * non_terminal * last_gae
            advantages[step] = last_gae
            next_value = values[step]

        returns = advantages + values
        return returns, advantages


def ppo_update(policy, optimizer, buffer, config, device):
    '''IPPO policy update function for training'''
    # Failsafe: if buffer is empty, no information for update
    if len(buffer) == 0:
        return {}

    # Get batch information stored in buffer
    obs = torch.tensor(np.asarray(buffer.obs), dtype=torch.float32, device=device)
    masks = torch.tensor(np.asarray(buffer.masks), dtype=torch.bool, device=device)
    actions = torch.tensor(np.asarray(buffer.actions), dtype=torch.int64, device=device)
    old_log_probs = torch.tensor(np.asarray(buffer.log_probs), dtype=torch.float32, device=device)
    old_values = torch.tensor(np.asarray(buffer.values), dtype=torch.float32, device=device)

    with torch.no_grad():
        # If the run ended, there is no future value
        if buffer.dones[-1]:
            last_value = 0.0
        else:
            _, last_value = policy(obs[-1:].to(device), masks[-1:].to(device))
            last_value = float(last_value.item())

    # Get returns and advantages from buffer
    returns, advantages = buffer.compute_returns_and_advantages(last_value, config)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    update_count = 0

    dataset_size = len(buffer)
    indices = np.arange(dataset_size)

    # For each epoch, update policy in batches
    for _ in range(config.ppo_epochs):
        np.random.shuffle(indices)
        # For each batch, update policy
        for start in range(0, dataset_size, config.minibatch_size):
            batch_idx = indices[start:start + config.minibatch_size]
            # Get batch information
            batch_obs = obs[batch_idx]
            batch_masks = masks[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_old_values = old_values[batch_idx]

            # Forward pass: calculate policy distribution and value estimation, distribution entropy and log-probabilities
            dist, values = policy(batch_obs, batch_masks)
            entropy = dist.entropy().mean()
            log_probs = dist.log_prob(batch_actions)

            # Calculate policy loss (actor)
            ratio = torch.exp(log_probs - batch_old_log_probs) # Policy ratio
            surr1 = ratio * batch_advantages # Ratio * advantage -> Unclipped surrogate loss
            surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) # Clip(ratio, 1-epsilon, 1+epsilon)
            surr2 = surr2 * batch_advantages # Clip * advantage -> Clipped surrogate loss
            policy_loss = -torch.min(surr1, surr2).mean() # Minimimum between clipped and unclipped surrogate losses

            # Calculate value loss (critic)
            value_pred_clipped = batch_old_values + torch.clamp(values - batch_old_values, -config.clip_eps, config.clip_eps) # Vold + clip(Vnew - Vold, -epsilon, epsilon) -> Clipped value prediction
            value_losses = (values - batch_returns).pow(2) # Unclipped value error
            value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean() # 1/2 * max(unclipped value error, clipped value predition - returns) -> Clipped value loss

            # Combine loss with entropy bonus
            loss = (policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy)
            
            # Apply entropy floor penalty to loss if enabled
            if config.entropy_floor is not None and config.entropy_floor > 0:
                floor_gap = torch.relu(torch.tensor(config.entropy_floor, device=entropy.device, dtype=entropy.dtype) - entropy)
                loss = loss + config.entropy_floor_coef * floor_gap

            # Backpropagation: calculate gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            update_count += 1

    return {
        "policy_loss": total_policy_loss / max(update_count, 1),
        "value_loss": total_value_loss / max(update_count, 1),
        "entropy": total_entropy / max(update_count, 1),
    }


def mappo_update(policy, optimizer, buffer, config, device):
    '''MAPPO policy update function for training'''
    # Failsafe: if buffer is empty, no information for update
    if len(buffer) == 0:
        return {}
    
    # Get batch information stored in buffer
    obs = torch.tensor(np.asarray(buffer.obs), dtype=torch.float32, device=device)
    masks = torch.tensor(np.asarray(buffer.masks), dtype=torch.bool, device=device)
    actions = torch.tensor(np.asarray(buffer.actions), dtype=torch.int64, device=device)
    old_log_probs = torch.tensor(np.asarray(buffer.log_probs), dtype=torch.float32, device=device)
    step_indices = torch.tensor(np.asarray(buffer.step_indices), dtype=torch.int64, device=device)

    joint_obs = torch.tensor(np.asarray(buffer.joint_obs), dtype=torch.float32, device=device)
    old_team_values = torch.tensor(np.asarray(buffer.team_values), dtype=torch.float32, device=device)


    with torch.no_grad():
        # If the run ended, there is no future value
        if buffer.dones[-1]:
            last_value = 0.0
        else:
            last_value = float(policy.team_value(joint_obs[-1:]).item())

    # Get returns and advantages from buffer, team-wise
    team_returns, team_advantages = buffer.compute_team_returns_and_advantages(last_value, config)
    team_returns = torch.tensor(team_returns, dtype=torch.float32, device=device)
    team_advantages = torch.tensor(team_advantages, dtype=torch.float32, device=device)
    advantages = team_advantages[step_indices]
    returns = team_returns[step_indices]
    old_values = old_team_values[step_indices]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    update_count = 0

    dataset_size = len(buffer)
    indices = np.arange(dataset_size)

    # For each epoch, update policy in batches
    for _ in range(config.ppo_epochs):
        np.random.shuffle(indices)
        # For each batch, update policy
        for start in range(0, dataset_size, config.minibatch_size):
            batch_idx = indices[start:start + config.minibatch_size]
            # Get batch information
            batch_obs = obs[batch_idx]
            batch_masks = masks[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            batch_old_values = old_values[batch_idx]
            batch_joint = joint_obs[step_indices[batch_idx]]

            # Forward pass: calculate distribution entropy and log-probabilities
            log_probs, entropy = policy.evaluate_actions(batch_obs, batch_masks, batch_actions)
            entropy_mean = entropy.mean()
            # Calculate policy loss (actor)
            ratio = torch.exp(log_probs - batch_old_log_probs) # Policy ratio
            surr1 = ratio * batch_advantages # Ratio * advantage -> Unclipped surrogate loss
            surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) # Clip(ratio, 1-epsilon, 1+epsilon)
            surr2 = surr2 * batch_advantages # Clip * advantage -> Clipped surrogate loss
            policy_loss = -torch.min(surr1, surr2).mean() # Minimimum between clipped and unclipped surrogate losses

            # Calculate value loss (critic)
            values = policy.team_value(batch_joint)
            value_pred_clipped = batch_old_values + torch.clamp(values - batch_old_values, -config.clip_eps, config.clip_eps) # Vold + clip(Vnew - Vold, -epsilon, epsilon) -> Clipped value prediction
            value_losses = (values - batch_returns).pow(2) # Unclipped value error
            value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean() # 1/2 * max(unclipped value error, clipped value predition - returns) -> Clipped value loss

            # Combine loss with entropy bonus
            loss = (policy_loss+ config.value_coef * value_loss - config.entropy_coef * entropy_mean)

            # Apply entropy floor penalty to loss if enabled
            if config.entropy_floor is not None and config.entropy_floor > 0:
                floor_gap = torch.relu(torch.tensor(config.entropy_floor, device=entropy_mean.device, dtype=entropy_mean.dtype) - entropy_mean)

                loss = loss + config.entropy_floor_coef * floor_gap

            # Backpropagation: calculate gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy_mean.item())
            update_count += 1

    return {
        "policy_loss": total_policy_loss / max(update_count, 1),
        "value_loss": total_value_loss / max(update_count, 1),
        "entropy": total_entropy / max(update_count, 1),
    }
