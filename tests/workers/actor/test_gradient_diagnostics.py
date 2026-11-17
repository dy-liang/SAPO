# Copyright 2026 SAPO contributors

import pytest
import torch

from verl.workers.actor.gradient_diagnostics import (
    SharedGradientDiagnosticAccumulator,
    select_named_trainable_parameters,
)


def test_shared_gradient_diagnostics_exact_alignment_and_weights():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    accumulator = SharedGradientDiagnosticAccumulator(
        (("model.layers.0.weight", parameter),),
    )
    parameter.grad = torch.tensor([1.0, 1.0])
    accumulator.capture_policy_gradient()
    parameter.grad = torch.tensor([1.0, -1.0])
    accumulator.accumulate_auxiliary_gradient("value")
    parameter.grad = torch.tensor([-1.0, -1.0])
    accumulator.accumulate_auxiliary_gradient("q")
    metrics = accumulator.finalize(
        value_weight=0.5,
        q_weight=0.25,
        q_active=True,
    )

    prefix = "gradient_diagnostics/all"
    assert metrics[f"{prefix}/policy_value_cosine"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/policy_q_cosine"] == pytest.approx(-1.0)
    assert metrics[f"{prefix}/value_q_cosine"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/policy_value_conflict"] == 0.0
    assert metrics[f"{prefix}/policy_q_conflict"] == 1.0
    assert metrics[f"{prefix}/value_q_conflict"] == 0.0
    assert metrics[f"{prefix}/weighted_value_to_policy_norm"] == pytest.approx(0.5)
    assert metrics[f"{prefix}/weighted_q_to_policy_norm"] == pytest.approx(0.25)
    assert metrics[f"{prefix}/weighted_value_policy_projection"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/weighted_q_policy_projection"] == pytest.approx(-0.25)
    assert metrics[f"{prefix}/total_policy_alignment"] == pytest.approx(0.75)
    assert metrics[f"{prefix}/total_is_policy_descent"] == 1.0
    assert metrics[f"{prefix}/joint_aux_to_policy_norm"] == pytest.approx(
        (0.5**2 + 0.25**2) ** 0.5
    )
    assert metrics[f"{prefix}/joint_grad_to_policy_norm"] == pytest.approx(
        0.8125**0.5
    )


def test_parameter_filters_and_unused_parameters():
    used = torch.nn.Parameter(torch.tensor([2.0]))
    unused = torch.nn.Parameter(torch.tensor([3.0]))
    selected = select_named_trainable_parameters(
        (("model.layers.0.weight", used), ("lm_head.weight", unused)),
        include_patterns=(r"layers",),
    )
    assert tuple(name for name, _ in selected) == ("model.layers.0.weight",)

    accumulator = SharedGradientDiagnosticAccumulator(
        (
            ("model.layers.0.weight", used),
            ("model.layers.1.weight", unused),
        ),
    )
    used.grad = torch.tensor([4.0])
    unused.grad = None
    accumulator.capture_policy_gradient()
    used.grad = torch.tensor([2.0])
    unused.grad = None
    accumulator.accumulate_auxiliary_gradient("value")
    metrics = accumulator.finalize(
        value_weight=0.1,
        q_weight=0.0,
        q_active=False,
    )
    prefix = "gradient_diagnostics/all"
    assert metrics[f"{prefix}/policy_grad_norm"] == pytest.approx(4.0)
    assert metrics[f"{prefix}/value_grad_norm"] == pytest.approx(2.0)
    assert metrics[f"{prefix}/policy_value_cosine"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/joint_aux_to_policy_norm"] == pytest.approx(0.05)
    assert metrics[f"{prefix}/joint_grad_to_policy_norm"] == pytest.approx(1.05)


def test_diagnostic_gradients_can_be_cleared_before_joint_backward():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    accumulator = SharedGradientDiagnosticAccumulator(
        (("model.layers.0.weight", parameter),),
    )

    parameter.sum().backward()
    accumulator.capture_policy_gradient()
    parameter.grad = None
    (parameter[0] - parameter[1]).backward()
    accumulator.accumulate_auxiliary_gradient("value")
    parameter.grad = None
    (-parameter.sum()).backward()
    accumulator.accumulate_auxiliary_gradient("q")

    parameter.grad = None
    joint_loss = (
        parameter.sum()
        + 0.5 * (parameter[0] - parameter[1])
        + 0.25 * (-parameter.sum())
    )
    joint_loss.backward()
    torch.testing.assert_close(parameter.grad, torch.tensor([1.25, 0.25]))


def test_zero_policy_gradient_is_not_reported_as_descent():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    accumulator = SharedGradientDiagnosticAccumulator(
        (("model.layers.0.weight", parameter),),
    )
    parameter.grad = torch.zeros_like(parameter)
    accumulator.capture_policy_gradient()
    parameter.grad = torch.ones_like(parameter)
    accumulator.accumulate_auxiliary_gradient("value")
    metrics = accumulator.finalize(
        value_weight=0.1,
        q_weight=0.0,
        q_active=False,
    )
    prefix = "gradient_diagnostics/all"
    assert metrics[f"{prefix}/policy_gradient_present"] == 0.0
    assert metrics[f"{prefix}/total_policy_alignment"] == 0.0
    assert metrics[f"{prefix}/total_is_policy_descent"] == 0.0

