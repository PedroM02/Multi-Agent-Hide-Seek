"""PPO rollout buffer and update for shared-policy IPPO."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    rollout_steps: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    entropy_coef: float = 0.02
    entropy_floor: float | None = 0.4
    entropy_floor_coef: float = 0.05
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class RolloutBuffer:
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

    def compute_returns_and_advantages(self, last_value, cfg):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = float(last_value)
        for step in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[step]
            delta = (
                rewards[step]
                + cfg.gamma * next_value * non_terminal
                - values[step]
            )
            last_gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_gae
            advantages[step] = last_gae
            next_value = values[step]

        returns = advantages + values
        return returns, advantages


class MAPPOBuffer:
    """Rollout buffer with team-level critic inputs and per-agent actor data."""

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

    def compute_team_returns_and_advantages(self, last_value, cfg):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.team_values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = float(last_value)
        for step in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[step]
            delta = (
                rewards[step]
                + cfg.gamma * next_value * non_terminal
                - values[step]
            )
            last_gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_gae
            advantages[step] = last_gae
            next_value = values[step]

        returns = advantages + values
        return returns, advantages


def ppo_update(policy, optimizer, buffer, cfg, device):
    if len(buffer) == 0:
        return {}

    obs = torch.tensor(np.asarray(buffer.obs), dtype=torch.float32, device=device)
    masks = torch.tensor(np.asarray(buffer.masks), dtype=torch.bool, device=device)
    actions = torch.tensor(np.asarray(buffer.actions), dtype=torch.int64, device=device)
    old_log_probs = torch.tensor(
        np.asarray(buffer.log_probs), dtype=torch.float32, device=device,
    )
    old_values = torch.tensor(
        np.asarray(buffer.values), dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        if buffer.dones[-1]:
            last_value = 0.0
        else:
            _, last_value = policy(obs[-1:].to(device), masks[-1:].to(device))
            last_value = float(last_value.item())

    returns, advantages = buffer.compute_returns_and_advantages(last_value, cfg)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    update_count = 0

    dataset_size = len(buffer)
    indices = np.arange(dataset_size)

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, cfg.minibatch_size):
            batch_idx = indices[start:start + cfg.minibatch_size]
            batch_obs = obs[batch_idx]
            batch_masks = masks[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_old_values = old_values[batch_idx]

            dist, values = policy(batch_obs, batch_masks)
            entropy = dist.entropy().mean()
            log_probs = dist.log_prob(batch_actions)

            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            surr2 = surr2 * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred_clipped = batch_old_values + torch.clamp(
                values - batch_old_values,
                -cfg.clip_eps,
                cfg.clip_eps,
            )
            value_losses = (values - batch_returns).pow(2)
            value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
            )
            if cfg.entropy_floor is not None and cfg.entropy_floor > 0:
                floor_gap = torch.relu(
                    torch.tensor(
                        cfg.entropy_floor,
                        device=entropy.device,
                        dtype=entropy.dtype,
                    )
                    - entropy
                )
                loss = loss + cfg.entropy_floor_coef * floor_gap

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
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


def mappo_update(policy, optimizer, buffer, cfg, device):
    if len(buffer) == 0:
        return {}

    obs = torch.tensor(np.asarray(buffer.obs), dtype=torch.float32, device=device)
    masks = torch.tensor(np.asarray(buffer.masks), dtype=torch.bool, device=device)
    actions = torch.tensor(np.asarray(buffer.actions), dtype=torch.int64, device=device)
    old_log_probs = torch.tensor(
        np.asarray(buffer.log_probs), dtype=torch.float32, device=device,
    )
    step_indices = torch.tensor(
        np.asarray(buffer.step_indices), dtype=torch.int64, device=device,
    )

    joint_obs = torch.tensor(
        np.asarray(buffer.joint_obs), dtype=torch.float32, device=device,
    )
    old_team_values = torch.tensor(
        np.asarray(buffer.team_values), dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        if buffer.dones[-1]:
            last_value = 0.0
        else:
            last_value = float(policy.team_value(joint_obs[-1:]).item())

    team_returns, team_advantages = buffer.compute_team_returns_and_advantages(
        last_value, cfg,
    )
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

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, cfg.minibatch_size):
            batch_idx = indices[start:start + cfg.minibatch_size]
            batch_obs = obs[batch_idx]
            batch_masks = masks[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            batch_old_values = old_values[batch_idx]
            batch_joint = joint_obs[step_indices[batch_idx]]

            log_probs, entropy = policy.evaluate_actions(
                batch_obs, batch_masks, batch_actions,
            )
            entropy_mean = entropy.mean()
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            surr2 = surr2 * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            values = policy.team_value(batch_joint)
            value_pred_clipped = batch_old_values + torch.clamp(
                values - batch_old_values,
                -cfg.clip_eps,
                cfg.clip_eps,
            )
            value_losses = (values - batch_returns).pow(2)
            value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy_mean
            )
            if cfg.entropy_floor is not None and cfg.entropy_floor > 0:
                floor_gap = torch.relu(
                    torch.tensor(
                        cfg.entropy_floor,
                        device=entropy_mean.device,
                        dtype=entropy_mean.dtype,
                    )
                    - entropy_mean
                )
                loss = loss + cfg.entropy_floor_coef * floor_gap

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
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
