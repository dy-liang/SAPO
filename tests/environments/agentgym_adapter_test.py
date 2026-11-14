# Copyright 2025 Individual Contributor: SAPO contributors
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

from unittest.mock import patch

import pytest

import agent_system.environments.env_package.agentgym.envs as agentgym_envs
import agent_system.environments.env_package.agentgym.projection as projection
from agent_system.environments.prompts.agentgym import (
    SCIWORLD_ACTIONS,
    SCIWORLD_TEMPLATE,
    SCIWORLD_TEMPLATE_NO_HIS,
    WEBARENA_ACTION_FORMAT,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, method, endpoint, timeout, **kwargs):
        self.requests.append((method, endpoint, timeout, kwargs))
        return FakeResponse(self.responses.pop(0))

    def close(self):
        pass


def test_sciworld_projection_requires_original_sapo_tags():
    actions, valids = projection.sciworld_projection(
        [
            "<think>move there</think><action>Go To Kitchen</action>",
            "Thought:\ninspect\nAction:\nlook around",
            "<action>look around</action>",
            "<think>检查</think><action>look around</action>",
        ]
    )
    assert actions[0] == "go to kitchen"
    assert valids == [1, 0, 0, 0]


def test_webarena_projection_preserves_tagged_action_case_for_server():
    actions, valids = projection.webarena_projection(
        [
            "<think>fill the form</think><action>type [42] [CaseSensitive] [1]</action>",
            "```click [42]```",
        ]
    )
    assert actions[0] == "```type [42] [CaseSensitive] [1]```"
    assert valids == [1, 0]


def test_agentgym_prompts_use_original_sapo_action_tags():
    assert "<think>" in SCIWORLD_TEMPLATE_NO_HIS
    assert "<action>" in SCIWORLD_TEMPLATE_NO_HIS
    assert "Thought:" not in SCIWORLD_TEMPLATE_NO_HIS
    assert "<think>" in WEBARENA_ACTION_FORMAT
    assert "<action>" in WEBARENA_ACTION_FORMAT


def test_sciworld_task_description_is_kept_once_via_full_history():
    task = "measure the boiling point"
    initial_state = f"{task}\nYou are in the lab."
    initial_prompt = SCIWORLD_TEMPLATE_NO_HIS.format(
        action_forms=SCIWORLD_ACTIONS,
        current_observation=initial_state,
    )
    assert initial_prompt.count(task) == 1

    history = f"[Observation 1: '{initial_state}', Action 1: 'look around']"
    later_prompt = SCIWORLD_TEMPLATE.format(
        action_forms=SCIWORLD_ACTIONS,
        action_history=history,
        current_observation="A thermometer is on the table.",
    )
    assert later_prompt.count(task) == 1


def test_sciworld_cumulative_score_is_converted_to_step_deltas():
    fake = FakeSession(
        [
            {"id": 7},
            {
                "task_description": "measure temperature",
                "observation": "You are in the lab.",
                "task_name": "task-1",
                "var_num": 2,
                "score": 5,
            },
            {"observation": "partial", "reward": 10, "score": 10, "done": False},
            {"observation": "complete", "reward": 15, "score": 25, "done": True},
        ]
    )
    with patch.object(agentgym_envs.requests, "Session", return_value=fake):
        client = agentgym_envs.AgentGymHttpClient(
            "sciworld", "http://env", reward_scale=0.1
        )
        observation, info = client.reset(12)
        _, reward_1, done_1, _ = client.step("look around")
        _, reward_2, done_2, info_2 = client.step("focus on thermometer")

    assert "measure temperature" in observation
    assert info["task_id"] == 12
    assert (reward_1, reward_2) == pytest.approx((1.0, 1.5))
    assert reward_1 + reward_2 == pytest.approx(info_2["task_score"] * 0.1)
    assert info_2["reward_scale"] == pytest.approx(0.1)
    assert not done_1 and done_2


def test_webarena_only_emits_terminal_evaluator_reward():
    fake = FakeSession(
        [
            {"env_idx": 3},
            {"observation": "raw", "sites": ["reddit"], "object": "find a post"},
            "full WebArena prompt",
            {
                "observation": "next",
                "reward": 0.75,
                "terminated": False,
                "truncated": False,
            },
            {
                "observation": "done",
                "reward": 1,
                "terminated": True,
                "truncated": False,
            },
        ]
    )
    with patch.object(agentgym_envs.requests, "Session", return_value=fake):
        client = agentgym_envs.AgentGymHttpClient(
            "webarena", "http://env", reward_scale=10.0
        )
        observation, _ = client.reset(9)
        _, reward_1, done_1, _ = client.step("```click [1]```")
        _, reward_2, done_2, info = client.step("```stop [answer]```")

    assert observation == "full WebArena prompt"
    assert reward_1 == 0.0
    assert reward_2 == 10.0
    assert not done_1 and done_2
    assert info["won"] is True
    assert info["task_score"] == 1.0
    assert info["reward_scale"] == 10.0


def test_task_id_validation_and_group_expansion_inputs():
    assert agentgym_envs._task_ids_from_kwargs([{"task_id": 4}, {"task_id": "5"}]) == [4, 5]
    with pytest.raises(ValueError, match="task_id"):
        agentgym_envs._task_ids_from_kwargs([{"other": 1}])


def test_reward_scale_must_be_positive():
    with pytest.raises(ValueError, match="reward_scale"):
        agentgym_envs.AgentGymHttpClient(
            "sciworld", "http://env", reward_scale=0.0
        )
