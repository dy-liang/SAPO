"""SPO task-value tracking and trajectory-level advantage utilities.

This module is intentionally independent from the Ray trainer so the SPO
statistics and advantage semantics can be unit-tested without distributed
workers.  It follows the official Single-stream Policy Optimization recipe:
one Beta posterior per task, pre-update advantages ``r - p_hat``, and one
normalized trajectory advantage broadcast to every action token. Optional
environment action shaping is applied only after this SPO normalization.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _config_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class SPOTracker:
    """Persistent per-task Beta statistics used by SPO."""

    def __init__(self, config: Any):
        self.offline_n = int(_config_get(config, "offline_n", 8))
        self.init_alpha = float(_config_get(config, "init_alpha", 0.5))
        self.init_beta = float(_config_get(config, "init_beta", 0.5))
        self.seed = int(_config_get(config, "seed", 0))
        self._rng = random.Random(self.seed)

        initialization_config = _config_get(config, "initialization", {})
        self.initialization_mode = str(_config_get(initialization_config, "mode", "none"))
        self.strict_task_coverage = bool(
            _config_get(initialization_config, "strict_task_coverage", False)
        )
        if self.initialization_mode not in {"collect", "offline", "none"}:
            raise ValueError(
                "SPO initialization.mode must be one of: collect, offline, none"
            )

        rho_config = _config_get(config, "rho", {})
        self.rho_clip_lower = float(_config_get(rho_config, "clip_lower", 0.875))
        if not 0.0 <= self.rho_clip_lower < 1.0:
            raise ValueError("SPO rho.clip_lower must be in [0, 1)")
        if self.init_alpha <= 0.0 or self.init_beta <= 0.0:
            raise ValueError("SPO init_alpha and init_beta must be positive")
        if self.offline_n < 0:
            raise ValueError("SPO offline_n must be non-negative")

        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}
        self.sampled_count: dict[str, int] = {}
        self.initialized_task_ids: set[str] = set()
        self.initialization_complete = self.initialization_mode == "none"

        offline_path = _config_get(config, "offline_values", None)
        require_offline = bool(_config_get(config, "require_offline_values", False))
        if offline_path in (None, "", "null", "None"):
            if require_offline:
                raise ValueError(
                    "SPO requires algorithm.spo.offline_values, but no JSON path was provided"
                )
        else:
            self.load_offline_values(str(offline_path))

    @staticmethod
    def canonical_task_id(task_id: Any) -> str:
        if task_id is None:
            raise ValueError("SPO received a null task ID")
        task_id = str(task_id).strip()
        if not task_id:
            raise ValueError("SPO received an empty task ID")
        return task_id

    def ensure(self, task_id: Any) -> str:
        task_id = self.canonical_task_id(task_id)
        if task_id not in self.alpha:
            if self.strict_task_coverage and self.initialization_mode != "none":
                raise RuntimeError(
                    f"SPO task {task_id!r} has no initial-policy tracker values; "
                    "canonical SPO requires init rollouts for every training task"
                )
            self.alpha[task_id] = self.init_alpha
            self.beta[task_id] = self.init_beta
            self.sampled_count[task_id] = 0
        return task_id

    def initialize_task(self, task_id: Any, raw_scores: Sequence[Any]) -> None:
        """Initialize one task from canonical initial-policy rollout outcomes."""
        task_id = self.canonical_task_id(task_id)
        if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
            raise ValueError(f"Initial SPO scores for {task_id!r} must be a list")

        binary_scores = [int(float(score) > 0.0) for score in raw_scores]
        if len(binary_scores) != self.offline_n:
            raise ValueError(
                f"Initial SPO task {task_id!r} has {len(binary_scores)} scores; "
                f"expected exactly offline_n={self.offline_n}"
            )

        if self.offline_n == 0:
            alpha, beta = self.init_alpha, self.init_beta
        else:
            successes = float(sum(binary_scores))
            p_hat = (successes + 0.5) / (self.offline_n + 1.0)
            n_init = 1.0 / (1.0 - self.rho_clip_lower)
            alpha = n_init * p_hat
            beta = n_init * (1.0 - p_hat)

        self.alpha[task_id] = float(alpha)
        self.beta[task_id] = float(beta)
        self.sampled_count[task_id] = 0
        self.initialized_task_ids.add(task_id)

    def load_offline_values(self, path: str) -> None:
        source = Path(path).expanduser()
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("SPO offline-values JSON must map task IDs to reward lists")

        for raw_task_id, raw_scores in payload.items():
            task_id = self.canonical_task_id(raw_task_id)
            if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
                raise ValueError(f"Offline SPO scores for {task_id!r} must be a list")
            binary_scores = [int(float(score) > 0.0) for score in raw_scores]
            if len(binary_scores) < self.offline_n:
                raise ValueError(
                    f"Offline SPO task {task_id!r} has {len(binary_scores)} scores; "
                    f"offline_n={self.offline_n}"
                )
            if self.offline_n == 0:
                selected_scores = []
            elif len(binary_scores) > self.offline_n:
                selected_scores = self._rng.sample(binary_scores, self.offline_n)
            else:
                selected_scores = binary_scores
            self.initialize_task(task_id, selected_scores)

    def value(self, task_id: Any) -> float:
        task_id = self.ensure(task_id)
        alpha, beta = self.alpha[task_id], self.beta[task_id]
        return alpha / (alpha + beta)

    def values(self, task_ids: Sequence[Any]) -> list[float]:
        return [self.value(task_id) for task_id in task_ids]

    def sampling_weights(self, epsilon: float = 0.05) -> dict[str, float]:
        if epsilon < 0.0:
            raise ValueError("SPO sampling epsilon must be non-negative")
        return {
            task_id: math.sqrt(self.value(task_id) * (1.0 - self.value(task_id))) + epsilon
            for task_id in self.alpha
        }

    def update(self, task_ids: Sequence[Any], rewards: Sequence[float], rhos: Sequence[float]) -> None:
        if not (len(task_ids) == len(rewards) == len(rhos)):
            raise ValueError("SPO update task_ids, rewards, and rhos must have equal lengths")
        for raw_task_id, raw_reward, raw_rho in zip(task_ids, rewards, rhos):
            task_id = self.ensure(raw_task_id)
            reward = float(raw_reward)
            rho = float(raw_rho)
            if reward not in (0.0, 1.0):
                raise ValueError(f"SPO requires binary rewards, got {reward} for {task_id!r}")
            if not 0.0 <= rho <= 1.0:
                raise ValueError(f"SPO rho must be in [0, 1], got {rho}")
            self.alpha[task_id] = rho * self.alpha[task_id] + reward
            self.beta[task_id] = rho * self.beta[task_id] + (1.0 - reward)
            self.sampled_count[task_id] = self.sampled_count.get(task_id, 0) + 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "sampled_count": dict(self.sampled_count),
            "initialized_task_ids": sorted(self.initialized_task_ids),
            "initialization_complete": bool(self.initialization_complete),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        alpha = {self.canonical_task_id(k): float(v) for k, v in state["alpha"].items()}
        beta = {self.canonical_task_id(k): float(v) for k, v in state["beta"].items()}
        if alpha.keys() != beta.keys():
            raise ValueError("SPO checkpoint alpha/beta task sets differ")
        if any(value <= 0.0 for value in (*alpha.values(), *beta.values())):
            raise ValueError("SPO checkpoint contains non-positive Beta parameters")
        self.alpha = alpha
        self.beta = beta
        self.sampled_count = {
            task_id: int(state.get("sampled_count", {}).get(task_id, 0)) for task_id in alpha
        }
        self.initialized_task_ids = {
            self.canonical_task_id(task_id)
            for task_id in state.get("initialized_task_ids", alpha.keys())
        }
        if (
            "initialization_complete" not in state
            and self.strict_task_coverage
            and self.initialization_mode != "none"
        ):
            raise ValueError(
                "SPO checkpoint predates canonical init-8 metadata and cannot be "
                "resumed as an initialized main run"
            )
        if "initialization_complete" in state:
            self.initialization_complete = bool(state["initialization_complete"])
        else:
            # A legacy, non-strict checkpoint may continue with its existing
            # tracker values; unseen tasks retain the configured Beta fallback.
            # Never recollect with a policy that has already been updated.
            self.initialization_complete = bool(alpha) or self.initialization_mode == "none"
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])


def extract_trajectory_records(
    task_ids: Sequence[Any],
    trajectory_ids: Sequence[Any],
    binary_rewards: Sequence[Any],
) -> tuple[list[str], list[str], list[float], np.ndarray]:
    """Return one validated record per trajectory in first-seen order."""

    if not (len(task_ids) == len(trajectory_ids) == len(binary_rewards)):
        raise ValueError("SPO row metadata arrays must have equal lengths")

    record_index: dict[str, int] = {}
    record_tasks: list[str] = []
    record_trajectories: list[str] = []
    record_rewards: list[float] = []
    row_to_record = np.empty(len(task_ids), dtype=np.int64)

    for row, (raw_task, raw_trajectory, raw_reward) in enumerate(
        zip(task_ids, trajectory_ids, binary_rewards)
    ):
        task_id = SPOTracker.canonical_task_id(raw_task)
        trajectory_id = str(raw_trajectory)
        reward = float(raw_reward)
        if reward not in (0.0, 1.0):
            raise ValueError(
                f"SPO requires per-trajectory binary success, got {reward} for {trajectory_id!r}"
            )
        if trajectory_id not in record_index:
            record_index[trajectory_id] = len(record_tasks)
            record_tasks.append(task_id)
            record_trajectories.append(trajectory_id)
            record_rewards.append(reward)
        index = record_index[trajectory_id]
        if record_tasks[index] != task_id or record_rewards[index] != reward:
            raise ValueError(
                f"Inconsistent SPO metadata across turns of trajectory {trajectory_id!r}"
            )
        row_to_record[row] = index

    return record_tasks, record_trajectories, record_rewards, row_to_record


def compute_spo_advantages(
    *,
    tracker: SPOTracker,
    task_ids: Sequence[Any],
    trajectory_ids: Sequence[Any],
    binary_rewards: Sequence[Any],
    response_mask: torch.Tensor,
    unique_turn_mask: torch.Tensor,
    invalid_action_mask: Optional[torch.Tensor] = None,
    invalid_action_penalty_coef: float = 0.0,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Compute SPO trajectory advantages, then apply optional turn-only shaping."""

    if response_mask.ndim != 2:
        raise ValueError(f"SPO response_mask must be rank 2, got {response_mask.shape}")
    if unique_turn_mask.ndim != 1 or unique_turn_mask.shape[0] != response_mask.shape[0]:
        raise ValueError("SPO unique_turn_mask must contain one value per row")
    if invalid_action_penalty_coef < 0.0:
        raise ValueError("SPO invalid-action penalty coefficient must be non-negative")
    if invalid_action_mask is None:
        invalid_action_mask = torch.zeros(
            response_mask.shape[0], dtype=torch.bool, device=response_mask.device
        )
    if invalid_action_mask.ndim != 1 or invalid_action_mask.shape[0] != response_mask.shape[0]:
        raise ValueError("SPO invalid_action_mask must contain one value per row")

    record_tasks, record_trajectories, record_rewards, row_to_record = extract_trajectory_records(
        task_ids, trajectory_ids, binary_rewards
    )
    if not record_tasks:
        raise ValueError("SPO cannot compute advantages for an empty batch")

    device = response_mask.device
    dtype = torch.float32
    p_hats = torch.tensor(tracker.values(record_tasks), dtype=dtype, device=device)
    effective_ns = torch.tensor(
        [tracker.alpha[task_id] + tracker.beta[task_id] for task_id in record_tasks],
        dtype=dtype,
        device=device,
    )
    rewards = torch.tensor(record_rewards, dtype=dtype, device=device)
    raw_advantages = rewards - p_hats
    if normalize:
        if raw_advantages.numel() == 1:
            normalized_advantages = torch.zeros_like(raw_advantages)
        else:
            normalized_advantages = (
                raw_advantages - raw_advantages.mean()
            ) / (raw_advantages.std(unbiased=True) + float(epsilon))
    else:
        normalized_advantages = raw_advantages

    row_to_record_tensor = torch.as_tensor(row_to_record, dtype=torch.long, device=device)
    row_raw_advantages = raw_advantages[row_to_record_tensor]
    row_normalized_advantages = normalized_advantages[row_to_record_tensor]
    row_p_hats = p_hats[row_to_record_tensor]
    row_effective_ns = effective_ns[row_to_record_tensor]
    row_rewards = rewards[row_to_record_tensor]
    # Preserve canonical SPO normalization at the trajectory level, then apply
    # the shared environment shaping only to each turn's policy advantage. The
    # binary reward and Beta posterior never observe this auxiliary penalty.
    row_policy_advantages = row_normalized_advantages - float(
        invalid_action_penalty_coef
    ) * invalid_action_mask.to(device=device, dtype=dtype)
    policy_mask = response_mask.to(torch.bool) & unique_turn_mask.to(torch.bool).unsqueeze(-1)
    token_advantages = row_policy_advantages.unsqueeze(-1) * policy_mask.to(dtype)

    unique_trajectory_mask = torch.zeros(response_mask.shape[0], dtype=torch.bool, device=device)
    seen_trajectories: set[str] = set()
    for row, trajectory_id in enumerate(map(str, trajectory_ids)):
        if bool(unique_turn_mask[row]) and trajectory_id not in seen_trajectories:
            unique_trajectory_mask[row] = True
            seen_trajectories.add(trajectory_id)

    return {
        "advantages": token_advantages,
        "returns": token_advantages.clone(),
        "policy_mask": policy_mask,
        "unique_trajectory_mask": unique_trajectory_mask,
        "row_raw_advantages": row_raw_advantages,
        "row_normalized_advantages": row_normalized_advantages,
        "row_policy_advantages": row_policy_advantages,
        "row_p_hats": row_p_hats,
        "row_effective_ns": row_effective_ns,
        "row_rewards": row_rewards,
        "record_tasks": record_tasks,
        "record_trajectories": record_trajectories,
        "record_rewards": record_rewards,
    }


def compute_spo_metrics(batch: Any) -> dict[str, float]:
    mask = batch.batch["spo_unique_trajectory_mask"].to(torch.bool)
    raw = batch.batch["spo_raw_advantages"][mask].to(torch.float32)
    normalized = batch.batch["spo_normalized_advantages"][mask].to(torch.float32)
    p_hats = batch.batch["spo_p_hats"][mask].to(torch.float32)
    rewards = batch.batch["spo_binary_rewards"][mask].to(torch.float32)
    effective_n = batch.batch["spo_effective_ns"][mask].to(torch.float32)
    turn_mask = batch.batch["spo_unique_turn_mask"].to(torch.bool)
    policy_advantages = batch.batch["spo_policy_advantages"][turn_mask].to(torch.float32)
    normalized_turn_advantages = batch.batch["spo_normalized_advantages"][turn_mask].to(
        torch.float32
    )
    metrics = {
        "spo/reward": rewards.mean().item(),
        "spo/p_hat": p_hats.mean().item(),
        "spo/p_hat_std": p_hats.std(unbiased=False).item(),
        "spo/Neff": effective_n.mean().item(),
        "spo/adv_before_norm": raw.mean().item(),
        "spo/adv_after_norm_mean": normalized.mean().item(),
        "spo/adv_after_norm_std": normalized.std(unbiased=False).item(),
        "spo/policy_advantage_mean": policy_advantages.mean().item(),
        "spo/policy_advantage_std": policy_advantages.std(unbiased=False).item(),
        "spo/invalid_action_penalty_mean": (
            normalized_turn_advantages - policy_advantages
        ).mean().item(),
    }
    quantiles = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=normalized.device)
    values = torch.quantile(normalized, quantiles)
    for label, value in zip(("p10", "p30", "p50", "p70", "p90"), values):
        metrics[f"spo/adv_after_norm/{label}"] = value.item()
    return metrics
