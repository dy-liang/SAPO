# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
torch.set_printoptions(threshold=torch.inf)
import verl.utils.torch_functional as verl_F

_MAX_LOG_RATIO = 20.0

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_ppo_turn_gae_advantage_return(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    old_values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_ids: Sequence[object],
    turn_indices: torch.Tensor,
    gamma: float,
    lam: float,
    unique_turn_mask: torch.Tensor | None = None,
    invalid_action_mask: torch.Tensor | None = None,
    invalid_action_penalty_coef: float = 0.0,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute finite-rollout, turn-level GAE for PPO's separate critic.

    Each batch row is one environment turn. ``old_values`` is the scalar
    critic prediction at the pre-action boundary (``values[:, 0]`` in the
    current critic layout). Rows may be shuffled and may contain shape-only
    copies inserted by ``adjust_batch``; trajectories are therefore rebuilt
    from ``(trajectory_id, turn_index)`` and only the explicitly marked
    canonical row participates in recursion and normalization.

    The value target is the standard lambda return ``V_old + A_GAE``. The
    invalid-action penalty is deliberately policy-only: it is applied after
    GAE and never changes the reward, TD error, or critic target. A final
    non-terminal row is a finite-horizon truncation with zero bootstrap,
    matching the rollout contract (there is no extra critic evaluation after
    ``env.max_steps``).

    Returns use two different masks:

    * ``policy_mask`` broadcasts the turn advantage over valid response tokens.
    * ``value_mask`` selects only column zero, the pre-action critic boundary.

    This function is independent of ARQ-PPO and contains no Q-value or
    reserved-token logic.
    """
    turn_tensors = {
        "rewards": rewards,
        "dones": dones,
        "old_values": old_values,
        "turn_indices": turn_indices,
    }
    if any(tensor.ndim != 1 for tensor in turn_tensors.values()):
        shapes = {name: tuple(tensor.shape) for name, tensor in turn_tensors.items()}
        raise ValueError(f"PPO turn GAE tensors must be one-dimensional, got {shapes}")

    batch_size = rewards.shape[0]
    if any(tensor.shape[0] != batch_size for tensor in turn_tensors.values()):
        raise ValueError("PPO turn GAE rewards, dones, values, and turn indices must have identical lengths")
    if len(trajectory_ids) != batch_size:
        raise ValueError("PPO turn GAE trajectory_ids must contain one ID per batch row")
    if response_mask.ndim != 2 or response_mask.shape[0] != batch_size:
        raise ValueError("PPO turn GAE response_mask must have shape [turns, response_length]")
    if response_mask.shape[1] == 0:
        raise ValueError("PPO turn GAE requires a non-empty response dimension")
    if invalid_action_mask is not None and invalid_action_mask.shape != rewards.shape:
        raise ValueError("PPO turn GAE invalid_action_mask must have one value per turn")
    if unique_turn_mask is not None and unique_turn_mask.shape != rewards.shape:
        raise ValueError("PPO turn GAE unique_turn_mask must have one value per turn")

    with torch.no_grad():
        rewards = rewards.detach().to(torch.float32)
        dones = dones.detach().to(device=rewards.device, dtype=torch.bool)
        old_values = old_values.detach().to(device=rewards.device, dtype=torch.float32)
        turn_indices = turn_indices.detach().to(device=rewards.device, dtype=torch.long)
        response_mask = response_mask.detach().to(device=rewards.device, dtype=torch.bool)
        if unique_turn_mask is not None:
            unique_turn_mask = unique_turn_mask.detach().to(device=rewards.device, dtype=torch.bool)
        if invalid_action_mask is not None:
            invalid_action_mask = invalid_action_mask.detach().to(device=rewards.device, dtype=torch.float32)

        finite_tensors = {"rewards": rewards, "old_values": old_values}
        if invalid_action_mask is not None:
            finite_tensors["invalid_action_mask"] = invalid_action_mask
        for name, tensor in finite_tensors.items():
            if not torch.all(torch.isfinite(tensor)):
                raise ValueError(f"PPO turn GAE {name} contains NaN or Inf")
        if torch.any(response_mask.sum(dim=-1) == 0):
            raise ValueError("PPO turn GAE requires every turn to contain at least one response token")

        # Build a PPO-owned trajectory index. This intentionally does not call
        # or share ARQ's target implementation.
        key_to_rows: dict[tuple[str, int], list[int]] = defaultdict(list)
        trajectory_to_keys: dict[str, list[tuple[int, tuple[str, int]]]] = defaultdict(list)
        for row, (trajectory_id, turn_index) in enumerate(zip(trajectory_ids, turn_indices.tolist())):
            key_to_rows[(str(trajectory_id), int(turn_index))].append(row)

        canonical_rows: list[int] = []
        key_to_canonical: dict[tuple[str, int], int] = {}
        for key, rows in key_to_rows.items():
            if unique_turn_mask is None:
                canonical_row = rows[0]
            else:
                canonical_candidates = [row for row in rows if bool(unique_turn_mask[row].item())]
                if len(canonical_candidates) != 1:
                    raise ValueError(
                        f"PPO turn GAE turn {key} must have exactly one canonical row, "
                        f"got {len(canonical_candidates)}"
                    )
                canonical_row = canonical_candidates[0]

            key_to_canonical[key] = canonical_row
            canonical_rows.append(canonical_row)
            trajectory_to_keys[key[0]].append((key[1], key))

            for duplicate_row in rows:
                if duplicate_row == canonical_row:
                    continue
                # Recomputed old values can differ slightly across distributed
                # BF16 micro-batches. Validate only immutable transition data.
                for name, tensor in {
                    "rewards": rewards,
                    "dones": dones,
                    "turn_indices": turn_indices,
                    "response_mask": response_mask,
                }.items():
                    lhs = tensor[canonical_row]
                    rhs = tensor[duplicate_row]
                    if tensor.dtype.is_floating_point:
                        agrees = torch.all(torch.isclose(lhs, rhs, rtol=1e-5, atol=1e-6))
                    else:
                        agrees = torch.all(lhs == rhs)
                    if not bool(agrees.item()):
                        raise ValueError(f"PPO turn GAE duplicate turn {key} disagrees on {name}")
                if invalid_action_mask is not None and not bool(
                    torch.isclose(
                        invalid_action_mask[canonical_row],
                        invalid_action_mask[duplicate_row],
                        rtol=0.0,
                        atol=0.0,
                    ).item()
                ):
                    raise ValueError(f"PPO turn GAE duplicate turn {key} disagrees on invalid_action_mask")

        gae_advantages = torch.zeros_like(rewards)
        td_errors = torch.zeros_like(rewards)
        value_targets = torch.zeros_like(rewards)
        truncated = torch.zeros_like(dones)

        for trajectory_id, indexed_keys in trajectory_to_keys.items():
            indexed_keys.sort(key=lambda item: item[0])
            if indexed_keys[0][0] != 0:
                raise ValueError(
                    f"PPO turn GAE trajectory {trajectory_id!r} must start at turn 0, "
                    f"got {indexed_keys[0][0]}"
                )
            for position in range(1, len(indexed_keys)):
                previous_turn = indexed_keys[position - 1][0]
                current_turn = indexed_keys[position][0]
                if current_turn != previous_turn + 1:
                    raise ValueError(
                        f"PPO turn GAE trajectory {trajectory_id!r} has non-contiguous turns "
                        f"{previous_turn} and {current_turn}"
                    )

            next_gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
            for position in range(len(indexed_keys) - 1, -1, -1):
                _, key = indexed_keys[position]
                row = key_to_canonical[key]
                has_successor = position + 1 < len(indexed_keys)
                is_done = bool(dones[row].item())
                if is_done and has_successor:
                    raise ValueError(f"PPO turn GAE terminal turn {key} unexpectedly has a successor")

                if has_successor:
                    next_key = indexed_keys[position + 1][1]
                    next_row = key_to_canonical[next_key]
                    next_value = old_values[next_row]
                    continuation = 1.0
                else:
                    next_value = torch.zeros((), dtype=torch.float32, device=rewards.device)
                    continuation = 0.0
                    truncated[row] = not is_done

                td_target = rewards[row] + gamma * continuation * next_value
                td_error = td_target - old_values[row]
                gae = td_error + gamma * lam * continuation * next_gae

                td_errors[row] = td_error
                gae_advantages[row] = gae
                value_targets[row] = old_values[row] + gae
                next_gae = gae

        resolved_unique_turn_mask = torch.zeros(batch_size, dtype=torch.bool, device=rewards.device)
        resolved_unique_turn_mask[
            torch.as_tensor(canonical_rows, dtype=torch.long, device=rewards.device)
        ] = True

        # Keep diagnostics defined on every row while excluding padding copies
        # from all actor/critic gradients through the masks below.
        for key, rows in key_to_rows.items():
            canonical_row = key_to_canonical[key]
            for duplicate_row in rows:
                if duplicate_row == canonical_row:
                    continue
                gae_advantages[duplicate_row] = gae_advantages[canonical_row]
                td_errors[duplicate_row] = td_errors[canonical_row]
                value_targets[duplicate_row] = value_targets[canonical_row]
                truncated[duplicate_row] = truncated[canonical_row]

        raw_policy_advantages = gae_advantages.clone()
        if invalid_action_mask is not None:
            raw_policy_advantages = (
                raw_policy_advantages - invalid_action_penalty_coef * invalid_action_mask
            )

        turn_advantages = raw_policy_advantages
        if normalize:
            stats_values = raw_policy_advantages[resolved_unique_turn_mask]
            mean = stats_values.mean()
            variance = stats_values.var(unbiased=False)
            turn_advantages = (raw_policy_advantages - mean) * torch.rsqrt(variance + epsilon)

        policy_mask = response_mask & resolved_unique_turn_mask.unsqueeze(-1)
        value_mask = torch.zeros_like(response_mask)
        value_mask[:, 0] = resolved_unique_turn_mask
        token_advantages = turn_advantages.unsqueeze(-1) * policy_mask
        token_returns = value_targets.unsqueeze(-1) * value_mask

    return {
        "advantages": token_advantages,
        "returns": token_returns,
        "turn_advantages": turn_advantages,
        "raw_turn_advantages": raw_policy_advantages,
        "gae_advantages": gae_advantages,
        "td_errors": td_errors,
        "value_targets": value_targets,
        "old_values": old_values,
        "unique_turn_mask": resolved_unique_turn_mask,
        "policy_mask": policy_mask,
        "value_mask": value_mask,
        "truncated": truncated,
    }


def compute_arq_gae_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    old_v_values: torch.Tensor,
    old_q_values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_ids: Sequence[object],
    turn_indices: torch.Tensor,
    gamma: float,
    lam: float,
    unique_turn_mask: torch.Tensor | None = None,
    invalid_action_mask: torch.Tensor | None = None,
    invalid_action_penalty_coef: float = 0.0,
    normalize: bool = True,
    epsilon: float = 1e-6,
    q_target_mode: str = "td0",
    q_lambda: float | None = None,
    advantage_mode: str = "gae",
    q_advantage_mix_coef: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute turn-level V/Q targets and the configured policy advantage.

    ``q_target_mode`` selects TD(0), a V-bootstrapped lambda-return,
    finite-trajectory Monte Carlo, or on-policy SARSA TD(0) supervision for Q.
    ``advantage_mode`` independently selects GAE, ``Q_old - V_old``, or a
    convex mixture for PPO.  All old predictions and targets are detached, so
    bootstrap values never receive gradients through this target computation.

    Rows may have been shuffled and may contain padding duplicates.  Trajectories
    are reconstructed from ``(trajectory_id, turn_index)`` and each unique turn
    participates exactly once in the reverse-time recursion and normalization.
    Computed scalars are then copied back to every duplicate row.

    A final non-terminal row is treated as a finite-horizon truncation and uses
    zero bootstrap.  This matches the rollout contract, which does not run an
    additional model forward on the observation beyond ``env.max_steps``.
    """
    turn_tensors = {
        "rewards": rewards,
        "dones": dones,
        "old_v_values": old_v_values,
        "old_q_values": old_q_values,
        "turn_indices": turn_indices,
    }
    if any(tensor.ndim != 1 for tensor in turn_tensors.values()):
        shapes = {name: tuple(tensor.shape) for name, tensor in turn_tensors.items()}
        raise ValueError(f"ARQ turn tensors must be one-dimensional, got {shapes}")
    batch_size = rewards.shape[0]
    if batch_size == 0:
        raise ValueError("ARQ GAE requires at least one turn")
    if any(tensor.shape[0] != batch_size for tensor in turn_tensors.values()):
        raise ValueError("ARQ rewards, dones, V, Q, and turn indices must have identical lengths")
    if len(trajectory_ids) != batch_size:
        raise ValueError("ARQ trajectory_ids must contain one ID per batch row")
    if response_mask.ndim != 2 or response_mask.shape[0] != batch_size:
        raise ValueError("ARQ response_mask must have shape [turns, response_length]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"ARQ gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"ARQ lambda must be in [0, 1], got {lam}")
    allowed_q_target_modes = {"td0", "lambda_return", "monte_carlo", "sarsa_td0"}
    if q_target_mode not in allowed_q_target_modes:
        raise ValueError(
            f"Unsupported ARQ q_target_mode {q_target_mode!r}; "
            f"expected one of {sorted(allowed_q_target_modes)}"
        )
    resolved_q_lambda = lam if q_lambda is None else q_lambda
    if not 0.0 <= resolved_q_lambda <= 1.0:
        raise ValueError(f"ARQ q_lambda must be in [0, 1], got {resolved_q_lambda}")
    allowed_advantage_modes = {"gae", "qv", "gae_qv_mix"}
    if advantage_mode not in allowed_advantage_modes:
        raise ValueError(
            f"Unsupported ARQ advantage_mode {advantage_mode!r}; "
            f"expected one of {sorted(allowed_advantage_modes)}"
        )
    if not 0.0 <= q_advantage_mix_coef <= 1.0:
        raise ValueError(
            "ARQ q_advantage_mix_coef must be in [0, 1], "
            f"got {q_advantage_mix_coef}"
        )
    if invalid_action_penalty_coef < 0:
        raise ValueError("ARQ invalid_action_penalty_coef must be non-negative")
    if invalid_action_mask is not None and invalid_action_mask.shape != rewards.shape:
        raise ValueError("ARQ invalid_action_mask must have one value per turn")
    if unique_turn_mask is not None and unique_turn_mask.shape != rewards.shape:
        raise ValueError("ARQ unique_turn_mask must have one value per turn")

    with torch.no_grad():
        rewards = rewards.detach().to(torch.float32)
        dones = dones.detach().to(torch.bool)
        old_v_values = old_v_values.detach().to(torch.float32)
        old_q_values = old_q_values.detach().to(torch.float32)
        turn_indices = turn_indices.detach().to(torch.long)
        if unique_turn_mask is not None:
            unique_turn_mask = unique_turn_mask.detach().to(device=rewards.device, dtype=torch.bool)

        for name, tensor in {
            "rewards": rewards,
            "old_v_values": old_v_values,
            "old_q_values": old_q_values,
        }.items():
            if not torch.all(torch.isfinite(tensor)):
                raise ValueError(f"ARQ {name} contains NaN or Inf")

        # Reconstruct unique turns after adjust_batch duplication and sequence
        # length balancing. Immutable transition metadata must still agree.
        key_to_rows: dict[tuple[str, int], list[int]] = defaultdict(list)
        trajectory_to_keys: dict[str, list[tuple[int, tuple[str, int]]]] = defaultdict(list)
        for row, (trajectory_id, turn_index) in enumerate(zip(trajectory_ids, turn_indices.tolist())):
            key_to_rows[(str(trajectory_id), int(turn_index))].append(row)

        canonical_rows = []
        key_to_canonical: dict[tuple[str, int], int] = {}
        for key, rows in key_to_rows.items():
            if unique_turn_mask is None:
                canonical_row = rows[0]
            else:
                canonical_candidates = [row for row in rows if bool(unique_turn_mask[row].item())]
                if len(canonical_candidates) != 1:
                    raise ValueError(
                        f"ARQ turn {key} must have exactly one canonical row, got {len(canonical_candidates)}"
                    )
                canonical_row = canonical_candidates[0]
            key_to_canonical[key] = canonical_row
            canonical_rows.append(canonical_row)
            trajectory_to_keys[key[0]].append((key[1], key))
            for duplicate_row in rows:
                if duplicate_row == canonical_row:
                    continue
                # adjust_batch copies can be evaluated on different ranks or in
                # different BF16 micro-batches. Their recomputed old V/Q may
                # therefore differ slightly even though the source transition
                # is identical. Validate immutable transition data only; all
                # derived targets use the explicitly marked canonical row and
                # duplicate rows are masked from every training loss.
                for name, tensor in {
                    "rewards": rewards,
                    "dones": dones,
                    "turn_indices": turn_indices,
                }.items():
                    lhs = tensor[canonical_row]
                    rhs = tensor[duplicate_row]
                    if tensor.dtype.is_floating_point:
                        agrees = torch.isclose(lhs, rhs, rtol=1e-5, atol=1e-6)
                    else:
                        agrees = lhs == rhs
                    if not bool(agrees.item()):
                        raise ValueError(f"ARQ duplicate turn {key} disagrees on {name}")
                if invalid_action_mask is not None and not bool(
                    (invalid_action_mask[canonical_row] == invalid_action_mask[duplicate_row]).item()
                ):
                    raise ValueError(f"ARQ duplicate turn {key} disagrees on invalid_action_mask")

        gae_advantages = torch.zeros_like(rewards)
        td_errors = torch.zeros_like(rewards)
        v_targets = torch.zeros_like(rewards)
        q_targets = torch.zeros_like(rewards)
        truncated = torch.zeros_like(dones)

        for trajectory_id, indexed_keys in trajectory_to_keys.items():
            indexed_keys.sort(key=lambda item: item[0])
            if indexed_keys[0][0] != 0:
                raise ValueError(
                    f"ARQ trajectory {trajectory_id!r} must start at turn 0, got {indexed_keys[0][0]}"
                )
            for position in range(1, len(indexed_keys)):
                previous_turn = indexed_keys[position - 1][0]
                current_turn = indexed_keys[position][0]
                if current_turn != previous_turn + 1:
                    raise ValueError(
                        f"ARQ trajectory {trajectory_id!r} has non-contiguous turns "
                        f"{previous_turn} and {current_turn}"
                    )

            next_gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
            next_q_return = torch.zeros((), dtype=torch.float32, device=rewards.device)
            for position in range(len(indexed_keys) - 1, -1, -1):
                _, key = indexed_keys[position]
                row = key_to_canonical[key]
                has_successor = position + 1 < len(indexed_keys)
                is_done = bool(dones[row].item())
                if is_done and has_successor:
                    raise ValueError(f"ARQ terminal turn {key} unexpectedly has a successor")

                if has_successor:
                    next_key = indexed_keys[position + 1][1]
                    next_row = key_to_canonical[next_key]
                    next_value = old_v_values[next_row]
                    continuation = 1.0
                else:
                    # Both true environment terminals and max-step horizons are
                    # terminal states of the finite rollout objective.
                    next_value = torch.zeros((), dtype=torch.float32, device=rewards.device)
                    continuation = 0.0
                    truncated[row] = not is_done

                td0_target = rewards[row] + gamma * continuation * next_value
                td_error = td0_target - old_v_values[row]
                gae = td_error + gamma * lam * continuation * next_gae
                v_target = old_v_values[row] + gae

                if q_target_mode == "td0":
                    q_target = td0_target
                elif q_target_mode == "lambda_return":
                    q_target = rewards[row] + gamma * continuation * (
                        (1.0 - resolved_q_lambda) * next_value
                        + resolved_q_lambda * next_q_return
                    )
                elif q_target_mode == "monte_carlo":
                    q_target = rewards[row] + gamma * continuation * next_q_return
                else:  # sarsa_td0
                    next_q_value = (
                        old_q_values[next_row]
                        if has_successor
                        else torch.zeros((), dtype=torch.float32, device=rewards.device)
                    )
                    q_target = rewards[row] + gamma * continuation * next_q_value

                q_targets[row] = q_target
                td_errors[row] = td_error
                gae_advantages[row] = gae
                v_targets[row] = v_target
                next_gae = gae
                if q_target_mode in {"lambda_return", "monte_carlo"}:
                    next_q_return = q_target

        resolved_unique_turn_mask = torch.zeros(batch_size, dtype=torch.bool, device=rewards.device)
        resolved_unique_turn_mask[
            torch.as_tensor(canonical_rows, dtype=torch.long, device=rewards.device)
        ] = True

        # Map canonical results back onto padding duplicates.
        for key, rows in key_to_rows.items():
            canonical_row = key_to_canonical[key]
            for duplicate_row in rows:
                if duplicate_row == canonical_row:
                    continue
                gae_advantages[duplicate_row] = gae_advantages[canonical_row]
                td_errors[duplicate_row] = td_errors[canonical_row]
                v_targets[duplicate_row] = v_targets[canonical_row]
                q_targets[duplicate_row] = q_targets[canonical_row]
                truncated[duplicate_row] = truncated[canonical_row]

        q_advantages = old_q_values - old_v_values
        if advantage_mode == "gae":
            raw_policy_advantage = gae_advantages.clone()
        elif advantage_mode == "qv":
            raw_policy_advantage = q_advantages.clone()
        else:  # gae_qv_mix
            raw_policy_advantage = (
                (1.0 - q_advantage_mix_coef) * gae_advantages
                + q_advantage_mix_coef * q_advantages
            )

        if invalid_action_mask is not None:
            invalid_action_mask = invalid_action_mask.detach().to(
                device=raw_policy_advantage.device, dtype=raw_policy_advantage.dtype
            )
            raw_policy_advantage = (
                raw_policy_advantage - invalid_action_penalty_coef * invalid_action_mask
            )

        normalized_advantage = raw_policy_advantage
        if normalize:
            stats_values = raw_policy_advantage[resolved_unique_turn_mask]
            mean = stats_values.mean()
            variance = stats_values.var(unbiased=False)
            normalized_advantage = (raw_policy_advantage - mean) * torch.rsqrt(variance + epsilon)

        token_advantage = normalized_advantage.unsqueeze(-1) * response_mask
        token_returns = v_targets.unsqueeze(-1) * response_mask

    return {
        "advantages": token_advantage,
        "returns": token_returns,
        "turn_advantages": normalized_advantage,
        "raw_turn_advantages": raw_policy_advantage,
        "gae_advantages": gae_advantages,
        "td_errors": td_errors,
        "v_targets": v_targets,
        "q_targets": q_targets,
        "q_advantages": q_advantages,
        "unique_turn_mask": resolved_unique_turn_mask,
        "truncated": truncated,
    }


def compute_arq_value_loss(
    current_probs: torch.Tensor,
    old_probs: torch.Tensor,
    target_probs: torch.Tensor,
    cliprange: float,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn-level PPO-style clipped regression loss for bounded V or Q.

    ``sample_mask`` excludes shape-only duplicates inserted by ``adjust_batch``
    without changing the physical batch size required by distributed workers.
    """
    if current_probs.ndim != 1 or old_probs.ndim != 1 or target_probs.ndim != 1:
        raise ValueError("ARQ value loss expects one scalar per turn")
    if not (current_probs.shape == old_probs.shape == target_probs.shape):
        raise ValueError("ARQ current, old, and target probabilities must have identical shapes")
    if cliprange < 0:
        raise ValueError(f"cliprange must be non-negative, got {cliprange}")

    old_probs = old_probs.detach().to(torch.float32)
    target_probs = target_probs.detach().to(torch.float32)
    current_probs = current_probs.to(torch.float32)
    clipped_probs = old_probs + torch.clamp(current_probs - old_probs, -cliprange, cliprange)
    loss_unclipped = (current_probs - target_probs).square()
    loss_clipped = (clipped_probs - target_probs).square()
    loss_per_turn = 0.5 * torch.maximum(loss_unclipped, loss_clipped)
    clipped = (torch.abs(current_probs - old_probs) > cliprange).to(torch.float32)
    if sample_mask is None:
        loss = loss_per_turn.mean()
        clip_fraction = clipped.mean()
    else:
        if sample_mask.shape != current_probs.shape:
            raise ValueError("ARQ value sample_mask must have one value per turn")
        sample_mask = sample_mask.to(device=current_probs.device, dtype=torch.float32)
        denominator = sample_mask.sum().clamp_min(1.0)
        loss = (loss_per_turn * sample_mask).sum() / denominator
        clip_fraction = (clipped * sample_mask).sum() / denominator
    return loss, clip_fraction


def compute_arq_q_loss(
    current_q_probs: torch.Tensor,
    current_v_probs: torch.Tensor,
    old_q_probs: torch.Tensor,
    old_v_probs: torch.Tensor,
    q_target_probs: torch.Tensor,
    q_advantage_target_probs: torch.Tensor,
    cliprange: float,
    loss_mode: str = "absolute",
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute absolute-Q or action-residual Q regression.

    Residual mode supervises ``Q(s, a) - stopgrad(V(s))``.  It therefore asks
    the post-action readout to explain the selected target's increment over the
    pre-action baseline without sending a direct Q-loss gradient through the V
    output.  The shared backbone can still transfer information indirectly.
    """
    if loss_mode == "absolute":
        return compute_arq_value_loss(
            current_probs=current_q_probs,
            old_probs=old_q_probs,
            target_probs=q_target_probs,
            cliprange=cliprange,
            sample_mask=sample_mask,
        )
    if loss_mode == "advantage_residual":
        return compute_arq_value_loss(
            current_probs=current_q_probs - current_v_probs.detach(),
            old_probs=old_q_probs - old_v_probs,
            target_probs=q_advantage_target_probs,
            cliprange=cliprange,
            sample_mask=sample_mask,
        )
    raise ValueError(
        f"Unsupported ARQ q_loss_mode {loss_mode!r}; "
        "expected 'absolute' or 'advantage_residual'"
    )


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    compute_mean_std_cross_steps: bool = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo: # y
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    #ratio = torch.exp(negative_approx_kl)

    # Clamp before exp.  Clipping ``ratio`` afterwards is too late: exp can
    # overflow first and produce a NaN gradient through 0 * Inf in backward.
    safe_negative_approx_kl = torch.clamp(negative_approx_kl, max=_MAX_LOG_RATIO)
    ratio = torch.exp(safe_negative_approx_kl)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "seq-mean-token-mean",
):
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # clamp before exp
        safe_kl = torch.clamp(kl, max=_MAX_LOG_RATIO)
        ratio = torch.exp(safe_kl)
        kld = (ratio - safe_kl - 1).contiguous()

        # old
        # ratio = torch.exp(kl)
        # kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
