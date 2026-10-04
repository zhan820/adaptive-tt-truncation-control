"""Finite-episode Double-DQN for the restart-level RL controller.

Observations are scaled and clipped by the caller. Replay contains independent
NumPy float32 copies, so observations collected inside the solver's
``torch.inference_mode()`` can be used for training after an episode ends.
The learner uses CPU only and does not change the problem's random generators.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def bellman_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_online_q: torch.Tensor,
    next_target_q: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Detached Double-DQN targets; terminal states have no bootstrap value.

The online network chooses the next action and the target network evaluates it.
``where`` masks terminals before addition, including nonfinite terminal values.
"""
    if rewards.ndim != 1 or dones.shape != rewards.shape:
        raise ValueError("rewards and dones must have matching [batch] shapes")
    if (
        next_online_q.ndim != 2
        or next_target_q.shape != next_online_q.shape
        or next_online_q.shape[0] != rewards.shape[0]
        or next_online_q.shape[1] < 1
    ):
        raise ValueError("next Q values must have matching [batch, actions] shapes")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and lie in [0, 1]")
    with torch.no_grad():
        actions = next_online_q.argmax(dim=1, keepdim=True)
        next_values = next_target_q.gather(1, actions).squeeze(1)
        continuation = torch.where(dones.bool(), torch.zeros_like(next_values), next_values)
        return rewards.detach() + gamma * continuation


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class _Replay:
    """Fixed-capacity ring buffer; allocation does not use a random generator."""

    def __init__(self, capacity: int, observation_dim: int):
        self.capacity = capacity
        self.states = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        return self.size

    def append(self, state, action, reward, next_state, done) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def state_dict(self) -> dict[str, Any]:
        result = {"size": self.size, "position": self.position}
        for name in ("states", "next_states", "actions", "rewards", "dones"):
            result[name] = torch.from_numpy(getattr(self, name)[: self.size].copy())
        return result


class QAgent:
    """Small 2-hidden-layer Double-DQN trained exclusively from transitions.

    ``gamma=1`` optimizes undiscounted return over the caller's finite episode.
    Call ``observe`` during solver callbacks and ``train_steps`` after the
    episode. Action values are persisted in their original index order.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        observation_dim: int,
        actions: list[float],
        seed: int = 0,
        gamma: float = 1.0,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        capacity: int = 50000,
        hidden: int = 64,
        target_update: int = 100,
    ):
        self.observation_dim = _positive_int(observation_dim, "observation_dim")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.capacity = _positive_int(capacity, "capacity")
        self.hidden = _positive_int(hidden, "hidden")
        self.target_update = _positive_int(target_update, "target_update")
        if self.capacity < self.batch_size:
            raise ValueError("capacity must be at least batch_size")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = int(seed)
        self.actions = [float(action) for action in actions]
        if not self.actions or not all(math.isfinite(action) for action in self.actions):
            raise ValueError("actions must be a nonempty list of finite values")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("action values must be distinct")
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and lie in [0, 1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self.rng = np.random.default_rng(self.seed)
        # Seed only the CPU generator and restore it afterwards. manual_seed()
        # on torch itself would also alter CUDA generators belonging to a solver.
        with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.online = nn.Sequential(
                nn.Linear(self.observation_dim, self.hidden, dtype=torch.float32, device="cpu"),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden, dtype=torch.float32, device="cpu"),
                nn.ReLU(),
                nn.Linear(self.hidden, len(self.actions), dtype=torch.float32, device="cpu"),
            )
            self.target = copy.deepcopy(self.online)
        self.online.eval()
        self.target.eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.learning_rate)
        self.replay = _Replay(self.capacity, self.observation_dim)
        self.gradient_steps = 0
        self.metadata: dict[str, Any] = {"actions": self.actions.copy(), "observation_scaling": "external"}

    def _observation(self, observation) -> np.ndarray:
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.array(observation, dtype=np.float32, copy=True)
        if value.shape != (self.observation_dim,) or not np.isfinite(value).all():
            raise ValueError(f"observation must contain {self.observation_dim} finite float32 values")
        return value

    def select_action(self, observation: list[float], epsilon: float = 0.0) -> int:
        """Choose an epsilon-greedy action; epsilon=0 does not consume RNG state."""
        state = self._observation(observation)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be finite and lie in [0, 1]")
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(len(self.actions)))
        self.online.eval()
        with torch.inference_mode():
            values = self.online(torch.from_numpy(state).unsqueeze(0))
            if not torch.isfinite(values).all():
                raise FloatingPointError("nonfinite Q values during action selection")
            return int(values.argmax(dim=1).item())

    def observe(self, state, action: int, reward: float, next_state, done: bool) -> None:
        """Store a detached transition without gradients, training or RNG use."""
        state_copy = self._observation(state)
        next_copy = self._observation(next_state)
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not 0 <= action < len(self.actions)
        ):
            raise ValueError("action index is outside the action space")
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("done must be a boolean")
        with np.errstate(over="ignore", invalid="ignore"):
            reward_copy = np.float32(reward)
        if np.ndim(reward_copy) != 0 or not np.isfinite(reward_copy):
            raise ValueError("reward must be a finite float32 scalar")
        self.replay.append(state_copy, int(action), reward_copy, next_copy, bool(done))

    def train_steps(self, count: int = 1) -> dict[str, Any]:
        """Run replay updates after the episode, returning JSON-safe loss stats."""
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("train_steps must run after the solver leaves torch.inference_mode()")
        losses = []
        if len(self.replay) >= self.batch_size:
            self.online.train()
            try:
                with torch.enable_grad():
                    for _ in range(count):
                        indices = self.rng.choice(len(self.replay), self.batch_size, replace=False)
                        states = torch.from_numpy(self.replay.states[indices])
                        actions = torch.from_numpy(self.replay.actions[indices])
                        rewards = torch.from_numpy(self.replay.rewards[indices])
                        next_states = torch.from_numpy(self.replay.next_states[indices])
                        dones = torch.from_numpy(self.replay.dones[indices])
                        with torch.no_grad():
                            targets = bellman_targets(
                                rewards, dones, self.online(next_states), self.target(next_states), self.gamma
                            )
                        predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
                        loss = nn.functional.smooth_l1_loss(predicted, targets)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite Double-DQN loss")
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0, error_if_nonfinite=True)
                        self.optimizer.step()
                        self.gradient_steps += 1
                        if self.gradient_steps % self.target_update == 0:
                            self.target.load_state_dict(self.online.state_dict())
                        losses.append(float(loss.detach()))
            finally:
                self.online.eval()
        return {
            "updates": len(losses),
            "loss_mean": float(np.mean(losses)) if losses else None,
            "loss_last": losses[-1] if losses else None,
            "replay_size": len(self.replay),
            "gradient_steps": self.gradient_steps,
        }

    def save(self, path, metadata: dict[str, Any] | None = None) -> None:
        """Save policy and complete resume state in a tensor-only-safe checkpoint."""
        saved_metadata = copy.deepcopy(self.metadata)
        if metadata is not None:
            saved_metadata.update(copy.deepcopy(metadata))
        if saved_metadata.get("actions") != self.actions:
            raise ValueError("metadata actions must agree with the agent's action order")
        checkpoint = {
            "format_version": self.FORMAT_VERSION,
            "config": {name: getattr(self, name) for name in (
                "observation_dim", "actions", "seed", "gamma", "learning_rate",
                "batch_size", "capacity", "hidden", "target_update",
            )},
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "gradient_steps": self.gradient_steps,
            "metadata": saved_metadata,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)

    @classmethod
    def load(cls, path) -> "QAgent":
        """Restore greedy behavior and the exact replay/optimizer/RNG position."""
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported QAgent checkpoint format")
        agent = cls(**checkpoint["config"])
        metadata = checkpoint["metadata"]
        if metadata.get("actions") != agent.actions:
            raise ValueError("checkpoint metadata actions do not match the network action order")
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        replay = checkpoint["replay"]
        size, position = replay["size"], replay["position"]
        if not 0 <= size <= agent.capacity or not 0 <= position < agent.capacity:
            raise ValueError("invalid replay size or write position")
        if size < agent.capacity and position != size:
            raise ValueError("invalid partial replay write position")
        for name in ("states", "next_states", "actions", "rewards", "dones"):
            source = replay[name].cpu().numpy()
            destination = getattr(agent.replay, name)[:size]
            if source.shape != destination.shape or not np.isfinite(source).all():
                raise ValueError(f"invalid checkpoint replay {name}")
            if name == "actions" and ((source < 0).any() or (source >= len(agent.actions)).any()):
                raise ValueError("invalid checkpoint replay action index")
            destination[:] = source
        agent.replay.size, agent.replay.position = size, position
        agent.rng.bit_generator.state = checkpoint["rng_state"]
        agent.gradient_steps = int(checkpoint["gradient_steps"])
        if agent.gradient_steps < 0:
            raise ValueError("invalid checkpoint gradient_steps")
        agent.metadata = copy.deepcopy(metadata)
        agent.online.eval()
        agent.target.eval().requires_grad_(False)
        return agent
