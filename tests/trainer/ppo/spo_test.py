import json

import numpy as np
import pytest
import torch

from verl.trainer.ppo.spo import SPOTracker, compute_spo_advantages


def test_offline_initialization_matches_official_recipe(tmp_path):
    path = tmp_path / "offline.json"
    path.write_text(json.dumps({"task-a": [1, 1, 0, 0]}), encoding="utf-8")
    tracker = SPOTracker(
        {
            "offline_values": str(path),
            "offline_n": 4,
            "rho": {"clip_lower": 0.875},
        }
    )

    assert tracker.alpha["task-a"] + tracker.beta["task-a"] == pytest.approx(8.0)
    assert tracker.value("task-a") == pytest.approx(0.5)


def test_advantage_is_per_trajectory_and_padding_copy_is_masked():
    tracker = SPOTracker({})
    tracker.alpha.update({"a": 1.0, "b": 3.0})
    tracker.beta.update({"a": 3.0, "b": 1.0})
    tracker.sampled_count.update({"a": 0, "b": 0})

    result = compute_spo_advantages(
        tracker=tracker,
        task_ids=np.array(["a", "a", "b", "a"], dtype=object),
        trajectory_ids=np.array(["ta", "ta", "tb", "ta"], dtype=object),
        binary_rewards=np.array([1.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 3, dtype=torch.bool),
        unique_turn_mask=torch.tensor([True, True, True, False]),
        invalid_action_mask=torch.tensor([False, True, False, True]),
        invalid_action_penalty_coef=1.0,
    )

    # Two trajectories are normalized; every canonical turn in ta shares one value.
    assert result["row_normalized_advantages"][0] == result["row_normalized_advantages"][1]
    assert result["row_policy_advantages"][0] == result["row_normalized_advantages"][0]
    assert result["row_policy_advantages"][1] == result["row_normalized_advantages"][1] - 1.0
    assert torch.all(result["advantages"][3] == 0)
    assert result["unique_trajectory_mask"].tolist() == [True, False, True, False]
    assert result["record_tasks"] == ["a", "b"]
    # Auxiliary action shaping must not mutate the Bayesian task statistics.
    assert tracker.alpha == {"a": 1.0, "b": 3.0}
    assert tracker.beta == {"a": 3.0, "b": 1.0}


def test_tracker_updates_once_for_each_supplied_trajectory():
    tracker = SPOTracker({})
    tracker.update(["a", "a"], [1.0, 0.0], [0.9, 0.9])
    assert tracker.sampled_count["a"] == 2
    assert tracker.alpha["a"] == pytest.approx(0.9 * (0.9 * 0.5 + 1.0))
    assert tracker.beta["a"] == pytest.approx(0.9 * (0.9 * 0.5) + 1.0)


def test_non_binary_reward_is_rejected():
    tracker = SPOTracker({})
    with pytest.raises(ValueError, match="binary"):
        compute_spo_advantages(
            tracker=tracker,
            task_ids=["a"],
            trajectory_ids=["ta"],
            binary_rewards=[0.5],
            response_mask=torch.ones(1, 2, dtype=torch.bool),
            unique_turn_mask=torch.ones(1, dtype=torch.bool),
        )
