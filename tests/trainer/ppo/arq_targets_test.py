# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_arq_gae_targets,
    compute_arq_q_loss,
    compute_arq_value_loss,
)


def _compute_targets(**overrides):
    kwargs = {
        "rewards": torch.tensor([1.0, 2.0, 3.0]),
        "dones": torch.tensor([False, False, True]),
        "old_v_values": torch.tensor([0.5, 1.0, 1.5]),
        "old_q_values": torch.tensor([0.75, 1.5, 2.5]),
        "response_mask": torch.ones(3, 2),
        "trajectory_ids": ["trajectory"] * 3,
        "turn_indices": torch.arange(3),
        "gamma": 0.5,
        "lam": 0.5,
        "normalize": False,
    }
    kwargs.update(overrides)
    return compute_arq_gae_targets(**kwargs)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("td0", [1.5, 2.75, 3.0]),
        ("lambda_return", [2.03125, 3.125, 3.0]),
        ("monte_carlo", [2.75, 3.5, 3.0]),
        ("sarsa_td0", [1.75, 3.25, 3.0]),
    ],
)
def test_q_target_modes(mode, expected):
    result = _compute_targets(q_target_mode=mode)
    torch.testing.assert_close(result["q_targets"], torch.tensor(expected))


def test_default_target_and_advantage_match_legacy_td0_gae_path():
    result = _compute_targets()
    expected_q_targets = torch.tensor([1.5, 2.75, 3.0])
    expected_td_errors = torch.tensor([1.0, 1.75, 1.5])
    expected_gae = torch.tensor([1.53125, 2.125, 1.5])

    torch.testing.assert_close(result["q_targets"], expected_q_targets)
    torch.testing.assert_close(result["td_errors"], expected_td_errors)
    torch.testing.assert_close(result["gae_advantages"], expected_gae)
    torch.testing.assert_close(result["v_targets"], expected_gae + torch.tensor([0.5, 1.0, 1.5]))
    torch.testing.assert_close(result["raw_turn_advantages"], expected_gae)


def test_default_absolute_q_loss_matches_legacy_value_loss_and_gradient():
    direct_q = torch.tensor([0.4, -0.2], requires_grad=True)
    wrapped_q = direct_q.detach().clone().requires_grad_(True)
    old_q = torch.tensor([0.3, -0.1])
    q_target = torch.tensor([0.2, 0.0])

    expected_loss, expected_clipfrac = compute_arq_value_loss(
        current_probs=direct_q,
        old_probs=old_q,
        target_probs=q_target,
        cliprange=0.2,
    )
    actual_loss, actual_clipfrac = compute_arq_q_loss(
        current_q_probs=wrapped_q,
        current_v_probs=torch.tensor([0.9, 0.8], requires_grad=True),
        old_q_probs=old_q,
        old_v_probs=torch.tensor([0.7, 0.6]),
        q_target_probs=q_target,
        q_advantage_target_probs=torch.tensor([-0.5, -0.6]),
        cliprange=0.2,
    )
    expected_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_clipfrac, expected_clipfrac)
    torch.testing.assert_close(wrapped_q.grad, direct_q.grad)


def test_q_lambda_endpoints_match_td0_and_monte_carlo():
    td0 = _compute_targets(q_target_mode="td0")["q_targets"]
    q_lambda_zero = _compute_targets(q_target_mode="lambda_return", q_lambda=0.0)[
        "q_targets"
    ]
    monte_carlo = _compute_targets(q_target_mode="monte_carlo")["q_targets"]
    q_lambda_one = _compute_targets(q_target_mode="lambda_return", q_lambda=1.0)[
        "q_targets"
    ]

    torch.testing.assert_close(q_lambda_zero, td0)
    torch.testing.assert_close(q_lambda_one, monte_carlo)


def test_lambda_q_target_matches_v_target_when_lambdas_match():
    result = _compute_targets(q_target_mode="lambda_return")
    torch.testing.assert_close(result["q_targets"], result["v_targets"])
    torch.testing.assert_close(
        result["q_targets"] - torch.tensor([0.5, 1.0, 1.5]),
        result["gae_advantages"],
    )


def test_policy_advantage_modes_are_independent_of_q_target():
    gae = _compute_targets(advantage_mode="gae")["raw_turn_advantages"]
    qv = _compute_targets(advantage_mode="qv")["raw_turn_advantages"]
    mixed = _compute_targets(
        advantage_mode="gae_qv_mix", q_advantage_mix_coef=0.25
    )["raw_turn_advantages"]

    torch.testing.assert_close(qv, torch.tensor([0.25, 0.5, 1.0]))
    torch.testing.assert_close(mixed, 0.75 * gae + 0.25 * qv)


def test_advantage_residual_q_loss_detaches_current_v_branch():
    current_q = torch.tensor([0.4, -0.2], requires_grad=True)
    current_v = torch.tensor([0.1, -0.1], requires_grad=True)
    loss, _ = compute_arq_q_loss(
        current_q_probs=current_q,
        current_v_probs=current_v,
        old_q_probs=torch.tensor([0.3, -0.1]),
        old_v_probs=torch.tensor([0.1, -0.1]),
        q_target_probs=torch.tensor([0.2, 0.0]),
        q_advantage_target_probs=torch.tensor([0.0, 0.0]),
        cliprange=1.0,
        loss_mode="advantage_residual",
    )
    loss.backward()

    assert current_q.grad is not None
    assert torch.isfinite(current_q.grad).all()
    assert current_v.grad is None


def test_invalid_arq_modes_fail_fast():
    with pytest.raises(ValueError, match="q_target_mode"):
        _compute_targets(q_target_mode="not-a-target")
    with pytest.raises(ValueError, match="advantage_mode"):
        _compute_targets(advantage_mode="not-an-advantage")
