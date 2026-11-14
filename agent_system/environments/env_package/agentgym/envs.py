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

import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import ray
import requests


class AgentGymProtocolError(RuntimeError):
    pass


def _json_or_error(response: requests.Response, endpoint: str, expect_object: bool = True):
    if response.status_code != 200:
        raise AgentGymProtocolError(
            f"AgentGym {endpoint} returned HTTP {response.status_code}: {response.text[:500]}"
        )
    payload = response.json()
    if isinstance(payload, dict) and "error" in payload:
        raise AgentGymProtocolError(f"AgentGym {endpoint} failed: {payload['error']}")
    if expect_object and not isinstance(payload, dict):
        raise AgentGymProtocolError(f"AgentGym {endpoint} returned {type(payload).__name__}, expected JSON object")
    return payload


class AgentGymHttpClient:
    """Small, dependency-local client for the SciWorld and WebArena servers."""

    def __init__(
        self,
        task_name: str,
        env_addr: str,
        timeout: float = 600.0,
        max_retries: int = 3,
        reward_scale: float = 1.0,
    ):
        task_name = task_name.lower()
        if task_name not in {"sciworld", "webarena"}:
            raise ValueError(f"Unsupported AgentGym task: {task_name}")
        self.task_name = task_name
        self.env_addr = env_addr.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.reward_scale = float(reward_scale)
        if self.reward_scale <= 0.0:
            raise ValueError(f"reward_scale must be positive, got {self.reward_scale}")
        self.session = requests.Session()
        self.env_id = self._create()
        self.last_score = 0.0
        self.last_observation = ""
        self.last_info: Dict[str, Any] = {}

    def _request(
        self,
        method: str,
        path: str,
        *,
        retry: bool = False,
        expect_object: bool = True,
        **kwargs,
    ):
        endpoint = f"{self.env_addr}/{path.lstrip('/')}"
        attempts = self.max_retries + 1 if retry else 1
        for attempt in range(attempts):
            try:
                response = self.session.request(method, endpoint, timeout=self.timeout, **kwargs)
                return _json_or_error(response, endpoint, expect_object=expect_object)
            except (requests.RequestException, ValueError, AgentGymProtocolError) as exc:
                if attempt + 1 >= attempts:
                    raise AgentGymProtocolError(f"AgentGym request failed at {endpoint}: {exc}") from exc
                time.sleep(min(5.0, 0.5 * (2**attempt)))
        raise AssertionError("unreachable")

    def _create(self) -> int:
        payload = self._request("POST", "create", retry=True)
        key = "id" if self.task_name == "sciworld" else "env_idx"
        if key not in payload:
            raise AgentGymProtocolError(f"AgentGym create response is missing {key!r}: {payload}")
        return int(payload[key])

    def reset(self, task_id: int):
        task_id = int(task_id)
        self.last_score = 0.0
        if self.task_name == "sciworld":
            payload = self._request(
                "POST",
                "reset",
                json={"id": self.env_id, "data_idx": task_id},
            )
            task_description = str(payload.get("task_description", "")).strip()
            observation = str(payload.get("observation", ""))
            # AgentGym-RL explicitly initializes the rollout score to zero after
            # reset. Keeping that baseline also preserves any score earned by the
            # server's implicit initial "look around" in the final episode score.
            self.last_score = 0.0
            self.last_observation = f"{task_description}\n{observation}".strip()
            self.last_info = {
                "task_id": task_id,
                "task_name": payload.get("task_name"),
                "variation_idx": payload.get("var_num"),
                "task_description": task_description,
                "task_score": 0.0,
                "won": False,
            }
        else:
            payload = self._request(
                "POST",
                "reset",
                json={"env_idx": self.env_id, "seed": 0, "idx": task_id},
            )
            if payload.get("observation") == "TimeoutError":
                raise TimeoutError(f"WebArena reset timed out for task {task_id}")
            observed = self._request(
                "GET",
                "observation",
                params={"env_idx": self.env_id},
                expect_object=False,
            )
            if isinstance(observed, dict):
                observed = observed.get("text", observed.get("observation", ""))
            self.last_observation = str(observed)
            self.last_info = {
                "task_id": task_id,
                "sites": payload.get("sites"),
                "objective": payload.get("object"),
                "task_score": 0.0,
                "won": False,
            }
        return self.last_observation, dict(self.last_info)

    def step(self, action: str):
        if self.task_name == "sciworld":
            payload = self._request(
                "POST",
                "step",
                json={"id": self.env_id, "action": action},
            )
            score = float(payload["score"])
            reward = (score - self.last_score) * self.reward_scale
            self.last_score = score
            done = bool(payload["done"])
            self.last_observation = str(payload["observation"])
            self.last_info.update(
                {
                    "raw_reward": float(payload.get("reward", 0.0)),
                    "task_score": score,
                    "reward_scale": self.reward_scale,
                    "won": done and score >= 100.0,
                }
            )
        else:
            payload = self._request(
                "POST",
                "step",
                json={"env_idx": self.env_id, "action": action},
            )
            done = bool(payload["terminated"])
            raw_reward = float(payload["reward"]) if done else 0.0
            reward = raw_reward * self.reward_scale
            self.last_score = raw_reward if done else self.last_score
            self.last_observation = str(payload["observation"])
            self.last_info.update(
                {
                    "truncated": bool(payload.get("truncated", False)),
                    "task_score": self.last_score,
                    "reward_scale": self.reward_scale,
                    "won": done and raw_reward > 0.0,
                }
            )
        return self.last_observation, reward, done, dict(self.last_info)

    def close(self):
        if self.env_id is None:
            return
        id_key = "id" if self.task_name == "sciworld" else "env_idx"
        try:
            self._request("POST", "close", json={id_key: self.env_id})
        finally:
            self.env_id = None
            self.session.close()


class AgentGymWorker:
    def __init__(
        self,
        task_name: str,
        env_addr: str,
        timeout: float,
        max_retries: int,
        reward_scale: float,
    ):
        self.client = AgentGymHttpClient(
            task_name,
            env_addr,
            timeout,
            max_retries,
            reward_scale,
        )
        self.reset_error = None

    def reset(self, task_id: int):
        self.reset_error = None
        try:
            return self.client.reset(task_id)
        except Exception as exc:
            # AgentGym-RL converts a failed reset into a zero-score finished
            # rollout. SAPO needs one step to carry that terminal flag.
            self.reset_error = f"{type(exc).__name__}: {exc}"
            self.client.last_score = 0.0
            self.client.last_observation = f"Environment reset failed: {self.reset_error}"
            self.client.last_info = {
                "task_id": int(task_id),
                "task_score": 0.0,
                "won": False,
                "env_error": self.reset_error,
            }
            return self.client.last_observation, dict(self.client.last_info)

    def step(self, action: str):
        if self.reset_error is not None:
            return self.client.last_observation, 0.0, True, dict(self.client.last_info)
        try:
            return self.client.step(action)
        except Exception as exc:
            # Previous SciWorld deltas may already have been accumulated. Emit
            # their negative sum so the failed rollout's final score is zero,
            # matching AgentGym-RL's exception path.
            reward = (
                -self.client.last_score * self.client.reward_scale
                if self.client.task_name == "sciworld"
                else 0.0
            )
            self.client.last_score = 0.0
            info = dict(self.client.last_info)
            info.update(
                {
                    "task_score": 0.0,
                    "won": False,
                    "env_error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.client.last_info = info
            return self.client.last_observation, reward, True, info

    def close(self):
        try:
            self.client.close()
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None


def _task_ids_from_kwargs(kwargs: Any) -> Optional[list[int]]:
    if kwargs is None:
        return None
    if isinstance(kwargs, dict):
        kwargs = [kwargs]
    result = []
    for item in kwargs:
        if isinstance(item, np.ndarray) and item.ndim == 0:
            item = item.item()
        if not isinstance(item, dict) or "task_id" not in item:
            raise ValueError("AgentGym env_kwargs entries must be dictionaries containing 'task_id'")
        result.append(int(item["task_id"]))
    return result


class AgentGymRemoteEnv:
    def __init__(
        self,
        task_name: str,
        env_addr: str,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool,
        timeout: float,
        max_retries: int,
        reward_scale: float,
        fallback_task_ids: Optional[Iterable[int]] = None,
    ):
        if not ray.is_initialized():
            ray.init()
        self.task_name = task_name.lower()
        self.env_num = int(env_num)
        self.group_n = int(group_n)
        self.num_processes = self.env_num * self.group_n
        self.is_train = bool(is_train)
        self._rng = np.random.RandomState(seed)
        self._fallback_task_ids = list([] if fallback_task_ids is None else fallback_task_ids)
        worker_cls = ray.remote(**resources_per_worker)(AgentGymWorker)
        self._workers = [
            worker_cls.remote(
                self.task_name,
                env_addr,
                timeout,
                max_retries,
                reward_scale,
            )
            for _ in range(self.num_processes)
        ]
        self._dones = [False] * self.num_processes
        self._last = [None] * self.num_processes
        self._closed = False

    def reset(self, kwargs=None):
        task_ids = _task_ids_from_kwargs(kwargs)
        if task_ids is None:
            if not self._fallback_task_ids:
                raise ValueError(
                    "No AgentGym task IDs were supplied. Run examples/data_preprocess/prepare_agentgym.py "
                    "and use its parquet files, or configure fallback task_ids."
                )
            sampled = self._rng.choice(
                self._fallback_task_ids,
                size=self.env_num,
                replace=self.env_num > len(self._fallback_task_ids),
            )
            task_ids = np.repeat(sampled, self.group_n).astype(int).tolist()
        elif len(task_ids) == self.env_num and self.group_n > 1:
            task_ids = np.repeat(task_ids, self.group_n).astype(int).tolist()
        if len(task_ids) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} AgentGym task IDs, got {len(task_ids)}")

        results = ray.get([worker.reset.remote(task_id) for worker, task_id in zip(self._workers, task_ids)])
        self._dones = [False] * self.num_processes
        self._last = [(obs, info) for obs, info in results]
        return [item[0] for item in results], [item[1] for item in results]

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} actions, got {len(actions)}")
        futures, active_indices = [], []
        for index, (worker, action, done) in enumerate(zip(self._workers, actions, self._dones)):
            if not done:
                futures.append(worker.step.remote(action))
                active_indices.append(index)
        active_results = ray.get(futures)
        by_index = dict(zip(active_indices, active_results))

        observations, rewards, dones, infos = [], [], [], []
        for index in range(self.num_processes):
            if index in by_index:
                obs, reward, done, info = by_index[index]
                self._dones[index] = bool(done)
                self._last[index] = (obs, info)
            else:
                obs, info = self._last[index]
                reward, done = 0.0, True
            observations.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)
        return observations, rewards, dones, infos

    def close(self):
        if self._closed:
            return
        futures = [worker.close.remote() for worker in self._workers]
        ray.get(futures)
        for worker in self._workers:
            ray.kill(worker)
        self._closed = True


def build_agentgym_envs(
    task_name: str,
    env_addr: str,
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool,
    timeout: float = 600.0,
    max_retries: int = 3,
    reward_scale: float = 1.0,
    fallback_task_ids: Optional[Iterable[int]] = None,
):
    return AgentGymRemoteEnv(
        task_name=task_name,
        env_addr=env_addr,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        timeout=timeout,
        max_retries=max_retries,
        reward_scale=reward_scale,
        fallback_task_ids=fallback_task_ids,
    )
