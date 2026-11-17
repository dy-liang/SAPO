# Copyright 2026 SAPO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""FSDP-safe gradient-alignment diagnostics for shared objectives.

The actor drives each diagnostic objective through an independent forward and
standard backward pass. This module snapshots the resulting reduced ``.grad``
shards and computes full-model statistics by summing scalar sufficient
statistics across ranks. It deliberately does not call ``torch.autograd.grad``:
standard backward preserves FSDP hooks and gradient-checkpointing semantics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import torch


_LAYER_PATTERN = re.compile(r"(?:layers|h)\.(\d+)(?:\.|$)")
_STAT_SIZE = 6
_POLICY_SQ = 0
_VALUE_SQ = 1
_POLICY_VALUE_DOT = 2
_Q_SQ = 3
_POLICY_Q_DOT = 4
_VALUE_Q_DOT = 5


def _as_local_dense_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local dense payload for Tensor, DTensor, or sparse grads."""
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    return tensor


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, name) is not None for pattern in patterns)


def select_named_trainable_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    include_patterns: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Select trainable parameters using optional regular-expression filters."""
    selected = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if include_patterns and not _matches_any(name, include_patterns):
            continue
        if exclude_patterns and _matches_any(name, exclude_patterns):
            continue
        selected.append((name, parameter))
    if not selected:
        raise ValueError("Gradient diagnostics selected no trainable parameters")
    return tuple(selected)


def _infer_parameter_groups(names: Sequence[str]) -> dict[str, str]:
    """Map FSDP/original parameter names to broad reviewer-facing groups."""
    layer_indices = []
    for name in names:
        match = _LAYER_PATTERN.search(name)
        if match is not None:
            layer_indices.append(int(match.group(1)))
    num_layers = max(layer_indices) + 1 if layer_indices else 0

    groups = {}
    for name in names:
        lower_name = name.lower()
        match = _LAYER_PATTERN.search(name)
        if match is not None and num_layers > 0:
            layer_index = int(match.group(1))
            bucket = min(2, (3 * layer_index) // num_layers)
            groups[name] = ("lower", "middle", "upper")[bucket]
        elif "embed" in lower_name:
            groups[name] = "embedding"
        elif "lm_head" in lower_name or "output_layer" in lower_name:
            groups[name] = "lm_head"
        else:
            # FSDP(use_orig_params=False) may combine parameters from several
            # semantic regions into a flat parameter. Such a shard cannot be
            # split post hoc without changing the training configuration.
            groups[name] = "other"
    return groups


def _distributed_sum_(tensor: torch.Tensor) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)


def _safe_cosine(dot: float, left_sq: float, right_sq: float, eps: float) -> float:
    denominator = math.sqrt(max(left_sq, 0.0) * max(right_sq, 0.0))
    return dot / denominator if denominator > eps else 0.0


def _group_metrics(
    prefix: str,
    values: Sequence[float],
    value_weight: float,
    q_weight: float,
    q_active: bool,
    eps: float,
) -> dict[str, float]:
    (
        policy_sq,
        value_sq,
        policy_value_dot,
        q_sq,
        policy_q_dot,
        value_q_dot,
    ) = values
    policy_norm = math.sqrt(max(policy_sq, 0.0))
    value_norm = math.sqrt(max(value_sq, 0.0))
    policy_value_cosine = _safe_cosine(
        policy_value_dot, policy_sq, value_sq, eps
    )

    metrics = {
        f"{prefix}/policy_grad_norm": policy_norm,
        f"{prefix}/value_grad_norm": value_norm,
        f"{prefix}/policy_gradient_present": float(policy_sq > eps),
        f"{prefix}/value_gradient_present": float(value_sq > eps),
        f"{prefix}/policy_value_cosine": policy_value_cosine,
        f"{prefix}/policy_value_conflict": float(
            policy_value_dot < 0.0 and policy_sq > eps and value_sq > eps
        ),
        f"{prefix}/weighted_value_to_policy_norm": (
            abs(value_weight) * value_norm / policy_norm
            if policy_norm > eps
            else 0.0
        ),
        f"{prefix}/weighted_value_policy_projection": (
            value_weight * policy_value_dot / policy_sq
            if policy_sq > eps
            else 0.0
        ),
    }

    weighted_projection = metrics[f"{prefix}/weighted_value_policy_projection"]
    weighted_norm_upper_bound = metrics[
        f"{prefix}/weighted_value_to_policy_norm"
    ]
    if q_active:
        q_norm = math.sqrt(max(q_sq, 0.0))
        q_projection = (
            q_weight * policy_q_dot / policy_sq if policy_sq > eps else 0.0
        )
        q_norm_ratio = (
            abs(q_weight) * q_norm / policy_norm if policy_norm > eps else 0.0
        )
        metrics.update(
            {
                f"{prefix}/q_grad_norm": q_norm,
                f"{prefix}/q_gradient_present": float(q_sq > eps),
                f"{prefix}/policy_q_cosine": _safe_cosine(
                    policy_q_dot, policy_sq, q_sq, eps
                ),
                f"{prefix}/policy_q_conflict": float(
                    policy_q_dot < 0.0 and policy_sq > eps and q_sq > eps
                ),
                f"{prefix}/value_q_cosine": _safe_cosine(
                    value_q_dot, value_sq, q_sq, eps
                ),
                f"{prefix}/value_q_conflict": float(
                    value_q_dot < 0.0 and value_sq > eps and q_sq > eps
                ),
                f"{prefix}/weighted_q_to_policy_norm": q_norm_ratio,
                f"{prefix}/weighted_q_policy_projection": q_projection,
            }
        )
        weighted_projection += q_projection
        weighted_norm_upper_bound += q_norm_ratio
        auxiliary_sq = (
            value_weight**2 * value_sq
            + q_weight**2 * q_sq
            + 2.0 * value_weight * q_weight * value_q_dot
        )
    else:
        auxiliary_sq = value_weight**2 * value_sq

    # For gradient descent, positive alignment means that the joint gradient
    # remains a first-order descent direction for the actor objective.
    total_policy_alignment = (
        1.0 + weighted_projection if policy_sq > eps else 0.0
    )
    joint_gradient_sq = (
        policy_sq + auxiliary_sq + 2.0 * policy_sq * weighted_projection
    )
    metrics.update(
        {
            f"{prefix}/total_policy_alignment": total_policy_alignment,
            f"{prefix}/total_is_policy_descent": float(
                policy_sq > eps and total_policy_alignment > 0.0
            ),
            f"{prefix}/joint_aux_to_policy_norm": (
                math.sqrt(max(auxiliary_sq, 0.0)) / policy_norm
                if policy_norm > eps
                else 0.0
            ),
            f"{prefix}/joint_grad_to_policy_norm": (
                math.sqrt(max(joint_gradient_sq, 0.0)) / policy_norm
                if policy_norm > eps
                else 0.0
            ),
            f"{prefix}/aux_to_policy_norm_upper_bound": weighted_norm_upper_bound,
        }
    )
    return metrics


class SharedGradientDiagnosticAccumulator:
    """Accumulate pairwise statistics from independent backward passes.

    Call order is ``capture_policy_gradient()``,
    ``accumulate_auxiliary_gradient("value")``, and optionally
    ``accumulate_auxiliary_gradient("q")``. The caller owns zeroing gradients
    between passes and before resuming the real optimizer update.
    """

    def __init__(
        self,
        named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        *,
        report_parameter_groups: bool = True,
        offload_gradient_snapshots: bool = True,
        reduction_device: torch.device | None = None,
    ) -> None:
        if not named_parameters:
            raise ValueError("named_parameters must not be empty")
        self.named_parameters = tuple(named_parameters)
        self.report_parameter_groups = report_parameter_groups
        self.offload_gradient_snapshots = offload_gradient_snapshots
        self.reduction_device = reduction_device or self.device
        self._parameter_groups = _infer_parameter_groups(
            tuple(name for name, _ in self.named_parameters)
        )
        active_groups = {"all"}
        if report_parameter_groups:
            active_groups.update(self._parameter_groups.values())
        stats_device = (
            torch.device("cpu") if offload_gradient_snapshots else self.device
        )
        self._stats = {
            group: torch.zeros(_STAT_SIZE, dtype=torch.float64, device=stats_device)
            for group in sorted(active_groups)
        }
        self._policy_gradients: tuple[torch.Tensor | None, ...] | None = None
        self._value_gradients: tuple[torch.Tensor | None, ...] | None = None
        self._captured_auxiliary_components: set[str] = set()

    @property
    def device(self) -> torch.device:
        return self.named_parameters[0][1].device

    def _copy_gradient(self, parameter: torch.nn.Parameter) -> torch.Tensor | None:
        if parameter.grad is None:
            return None
        gradient = _as_local_dense_tensor(parameter.grad.detach())
        target_device = (
            torch.device("cpu")
            if self.offload_gradient_snapshots
            else gradient.device
        )
        return gradient.to(device=target_device, dtype=torch.float32, copy=True)

    def _add_stat(self, name: str, index: int, value: torch.Tensor) -> None:
        self._stats["all"][index].add_(value)
        if self.report_parameter_groups:
            self._stats[self._parameter_groups[name]][index].add_(value)

    def capture_policy_gradient(self) -> None:
        """Snapshot reduced policy-gradient shards from ``parameter.grad``."""
        if self._policy_gradients is not None:
            raise RuntimeError("Policy gradient has already been captured")
        snapshots = []
        for name, parameter in self.named_parameters:
            gradient = self._copy_gradient(parameter)
            snapshots.append(gradient)
            if gradient is not None:
                self._add_stat(
                    name,
                    _POLICY_SQ,
                    torch.sum(gradient * gradient, dtype=torch.float64),
                )
        self._policy_gradients = tuple(snapshots)

    def accumulate_auxiliary_gradient(self, component: str) -> None:
        """Compare the current reduced ``.grad`` shards with policy snapshots."""
        if self._policy_gradients is None:
            raise RuntimeError("Capture the policy gradient before auxiliary gradients")
        if component == "value":
            square_index, dot_index = _VALUE_SQ, _POLICY_VALUE_DOT
        elif component == "q":
            square_index, dot_index = _Q_SQ, _POLICY_Q_DOT
        else:
            raise ValueError(f"Unsupported gradient component: {component}")
        if component in self._captured_auxiliary_components:
            raise RuntimeError(f"Gradient component {component} was already captured")
        if component == "q" and self._value_gradients is None:
            raise RuntimeError("Capture the value gradient before the Q gradient")

        auxiliary_snapshots = []
        for parameter_index, ((name, parameter), policy_gradient) in enumerate(
            zip(self.named_parameters, self._policy_gradients)
        ):
            auxiliary_gradient = self._copy_gradient(parameter)
            if component == "value":
                auxiliary_snapshots.append(auxiliary_gradient)
            if auxiliary_gradient is not None:
                self._add_stat(
                    name,
                    square_index,
                    torch.sum(
                        auxiliary_gradient * auxiliary_gradient, dtype=torch.float64
                    ),
                )
            if policy_gradient is not None and auxiliary_gradient is not None:
                self._add_stat(
                    name,
                    dot_index,
                    torch.sum(
                        policy_gradient * auxiliary_gradient, dtype=torch.float64
                    ),
                )
            if (
                component == "q"
                and self._value_gradients is not None
                and self._value_gradients[parameter_index] is not None
                and auxiliary_gradient is not None
            ):
                self._add_stat(
                    name,
                    _VALUE_Q_DOT,
                    torch.sum(
                        self._value_gradients[parameter_index] * auxiliary_gradient,
                        dtype=torch.float64,
                    ),
                )
        if component == "value":
            self._value_gradients = tuple(auxiliary_snapshots)
        self._captured_auxiliary_components.add(component)

    def finalize(
        self,
        *,
        value_weight: float,
        q_weight: float,
        q_active: bool,
        eps: float = 1.0e-12,
    ) -> dict[str, float]:
        """Reduce scalar statistics across ranks and create logging metrics."""
        if self._policy_gradients is None:
            raise RuntimeError("No policy gradient was captured")
        if "value" not in self._captured_auxiliary_components:
            raise RuntimeError("No value gradient was captured")
        if q_active and "q" not in self._captured_auxiliary_components:
            raise RuntimeError("No Q gradient was captured")
        metrics = {}
        for group in sorted(self._stats):
            # NCCL/HCCL require device tensors, whereas offloaded snapshots and
            # sufficient statistics live on CPU to minimize accelerator memory.
            reduced_stats = self._stats[group].to(self.reduction_device)
            _distributed_sum_(reduced_stats)
            metrics.update(
                _group_metrics(
                    prefix=f"gradient_diagnostics/{group}",
                    values=reduced_stats.detach().cpu().tolist(),
                    value_weight=value_weight,
                    q_weight=q_weight,
                    q_active=q_active,
                    eps=eps,
                )
            )
        metrics["gradient_diagnostics/parameter_tensors"] = float(
            len(self.named_parameters)
        )
        return metrics
