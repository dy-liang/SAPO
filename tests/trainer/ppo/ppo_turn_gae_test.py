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
    compute_ppo_turn_gae_advantage_return,
    compute_value_loss,
)


def _compute_turn_gae(**overrides):
    kwargs = {
        "rewards": torch.tensor([1.0, 2.0, 3.0]),
        "dones": torch.tensor([False, False, True]),
        "old_values": torch.tensor([0.5, 1.0, 1.5]),
        "response_mask": torch.ones(3, 2, dtype=torch.bool),
        "trajectory_ids": ["trajectory"] * 3,
        "turn_indices": torch.arange(3),
        "gamma": 0.5,
        "lam": 0.5,
        "normalize": False,
    }
    kwargs.update(overrides)
    return compute_ppo_turn_gae_advantage_return(**kwargs)


def test_turn_gae_matches_hand_computation_and_uses_boundary_return_only():
    result = _compute_turn_gae()
    expected_td_errors = torch.tensor([1.0, 1.75, 1.5])
    expected_gae = torch.tensor([1.53125, 2.125, 1.5])
    expected_targets = torch.tensor([2.03125, 3.125, 3.0])

    torch.testing.assert_close(result["td_errors"], expected_td_errors)
    torch.testing.assert_close(result["gae_advantages"], expected_gae)
    torch.testing.assert_close(result["value_targets"], expected_targets)
    torch.testing.assert_close(result["advantages"], expected_gae[:, None].expand(-1, 2))
    torch.testing.assert_close(
        result["returns"],
        torch.tensor([[2.03125, 0.0], [3.125, 0.0], [3.0, 0.0]]),
    )
    assert torch.equal(
        result["value_mask"],
        torch.tensor([[True, False], [True, False], [True, False]]),
    )


def test_lambda_endpoints_are_td0_and_finite_rollout_monte_carlo():
    td0 = _compute_turn_gae(lam=0.0)
    monte_carlo = _compute_turn_gae(lam=1.0)

    torch.testing.assert_close(td0["value_targets"], torch.tensor([1.5, 2.75, 3.0]))
    torch.testing.assert_close(monte_carlo["value_targets"], torch.tensor([2.75, 3.5, 3.0]))


def test_shuffled_trajectories_and_padding_duplicate_are_reconstructed_once():
    # Canonical order in the input is deliberately B0, A1, A0, B1, A2.
    # The final row is an adjust_batch-style duplicate of A1.
    result = compute_ppo_turn_gae_advantage_return(
        rewards=torch.tensor([4.0, 2.0, 1.0, 5.0, 3.0, 2.0]),
        dones=torch.tensor([False, False, False, True, True, False]),
        old_values=torch.tensor([2.0, 1.0, 0.5, 0.5, 1.5, 1.01]),
        response_mask=torch.tensor(
            [[1, 1], [1, 1], [1, 0], [1, 1], [1, 1], [1, 1]],
            dtype=torch.bool,
        ),
        trajectory_ids=["B", "A", "A", "B", "A", "A"],
        turn_indices=torch.tensor([0, 1, 0, 1, 2, 1]),
        gamma=0.5,
        lam=0.5,
        unique_turn_mask=torch.tensor([True, True, True, True, True, False]),
        normalize=False,
    )

    # A trajectory targets: [2.03125, 3.125, 3.0].
    torch.testing.assert_close(
        result["value_targets"],
        torch.tensor([5.375, 3.125, 2.03125, 5.0, 3.0, 3.125]),
    )
    assert torch.equal(
        result["unique_turn_mask"],
        torch.tensor([True, True, True, True, True, False]),
    )
    assert not result["policy_mask"][-1].any()
    assert not result["value_mask"][-1].any()
    assert torch.count_nonzero(result["advantages"][-1]) == 0
    assert torch.count_nonzero(result["returns"][-1]) == 0


def test_invalid_action_penalty_is_policy_only_and_normalizes_unique_turns():
    unique_mask = torch.tensor([True, True, True, False])
    result = compute_ppo_turn_gae_advantage_return(
        rewards=torch.tensor([1.0, 2.0, 3.0, 2.0]),
        dones=torch.tensor([False, False, True, False]),
        old_values=torch.tensor([0.5, 1.0, 1.5, 1.01]),
        response_mask=torch.ones(4, 2, dtype=torch.bool),
        trajectory_ids=["A", "A", "A", "A"],
        turn_indices=torch.tensor([0, 1, 2, 1]),
        gamma=0.5,
        lam=0.5,
        unique_turn_mask=unique_mask,
        invalid_action_mask=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        invalid_action_penalty_coef=2.0,
        normalize=True,
    )

    expected_gae = torch.tensor([1.53125, 2.125, 1.5, 2.125])
    expected_raw = expected_gae - torch.tensor([0.0, 2.0, 0.0, 2.0])
    torch.testing.assert_close(result["gae_advantages"], expected_gae)
    torch.testing.assert_close(result["value_targets"], torch.tensor([2.03125, 3.125, 3.0, 3.125]))
    torch.testing.assert_close(result["raw_turn_advantages"], expected_raw)
    torch.testing.assert_close(
        result["turn_advantages"][unique_mask].mean(),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result["turn_advantages"][unique_mask].var(unbiased=False),
        torch.tensor(1.0),
        atol=5e-6,
        rtol=0.0,
    )


def test_final_nonterminal_turn_is_marked_truncated_and_zero_bootstrapped():
    result = _compute_turn_gae(dones=torch.tensor([False, False, False]))
    assert torch.equal(result["truncated"], torch.tensor([False, False, True]))
    torch.testing.assert_close(result["td_errors"][-1], torch.tensor(1.5))


def test_targets_and_old_values_are_detached():
    old_values = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
    result = _compute_turn_gae(old_values=old_values)

    assert not result["old_values"].requires_grad
    assert not result["value_targets"].requires_grad
    assert not result["advantages"].requires_grad


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"turn_indices": torch.tensor([1, 2, 3])}, "start at turn 0"),
        ({"turn_indices": torch.tensor([0, 2, 3])}, "non-contiguous"),
        ({"dones": torch.tensor([True, False, True])}, "terminal turn"),
        ({"old_values": torch.tensor([0.5, float("nan"), 1.5])}, "NaN or Inf"),
    ],
)
def test_invalid_trajectory_contract_fails_fast(overrides, message):
    with pytest.raises(ValueError, match=message):
        _compute_turn_gae(**overrides)


def test_clipped_critic_loss_only_backpropagates_at_turn_value_boundary():
    current_values = torch.tensor(
        [[0.4, 10.0, -10.0], [0.2, 20.0, -20.0]],
        requires_grad=True,
    )
    old_values = torch.zeros_like(current_values)
    targets = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    value_mask = torch.tensor(
        [[True, False, False], [True, False, False]],
    )

    loss, _ = compute_value_loss(
        vpreds=current_values,
        values=old_values,
        returns=targets,
        response_mask=value_mask,
        cliprange_value=0.5,
    )
    loss.backward()

    assert torch.count_nonzero(current_values.grad[:, 0]) == 2
    assert torch.count_nonzero(current_values.grad[:, 1:]) == 0
