"""Ray-vectorized Docker environment for SWE-bench-style tasks."""

from __future__ import annotations

import io
import os
import shlex
import tarfile
import uuid
from typing import Any, Dict, List

import ray

from .reward import compute_reward
from .utils import canonical_image_name


def _clip_output(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    half = max(1, limit // 2)
    omitted = len(text) - (2 * half)
    return f"{text[:half]}\n... <{omitted} characters omitted> ...\n{text[-half:]}"


class SWEBenchWorker:
    """One long-lived Ray actor that hosts at most one Docker container."""

    def __init__(self, env_config: Dict[str, Any]):
        self.config = dict(env_config)
        self.client = None
        self.container = None
        self.metadata: Dict[str, Any] | None = None
        self.instance_id: str | None = None
        self._result: Dict[str, Any] | None = None

    def _docker(self):
        if self.client is None:
            try:
                import docker
            except ImportError as exc:
                raise RuntimeError(
                    "The SWE-bench environment requires Docker SDK. Install "
                    "`pip install -r requirements-swebench.txt`."
                ) from exc
            self.client = docker.from_env()
            self.client.ping()
        return self.client

    def _cleanup_container(self) -> None:
        if self.container is None:
            return
        try:
            self.container.remove(force=True)
        except Exception:
            pass
        finally:
            self.container = None

    def _put_text_file(self, path: str, content: str, mode: int = 0o644) -> None:
        if self.container is None:
            raise RuntimeError("SWE-bench container is not running")
        directory, filename = os.path.split(path)
        payload = content.encode("utf-8")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            info = tarfile.TarInfo(filename)
            info.size = len(payload)
            info.mode = mode
            tar.addfile(info, io.BytesIO(payload))
        archive.seek(0)
        if not self.container.put_archive(directory or "/", archive.getvalue()):
            raise RuntimeError(f"Failed to write {path} into container")

    def _exec(self, command: str, timeout: float) -> tuple[int, str]:
        if self.container is None:
            raise RuntimeError("SWE-bench container is not running")
        seconds = max(1, int(timeout))
        wrapped = (
            f"timeout --signal=TERM --kill-after=5s {seconds}s "
            f"bash -lc {shlex.quote(command)}"
        )
        result = self.container.exec_run(
            ["bash", "-lc", wrapped],
            workdir="/testbed",
            stdout=True,
            stderr=True,
        )
        output = result.output.decode("utf-8", errors="replace")
        return int(result.exit_code), output

    def _execute_eval_script(self, script: str, timeout: float) -> tuple[int, str]:
        path = f"/tmp/sapo_swebench_eval_{uuid.uuid4().hex}.sh"
        self._put_text_file(path, script, mode=0o700)
        return self._exec(f"bash {shlex.quote(path)} 2>&1", timeout)

    def _sanitize_git_history(self) -> None:
        """Rebuild Git from the visible base tree so the gold commit is unreachable."""
        command = r"""
git ls-files -z > /tmp/sapo_tracked_files
rm -rf .git
git init -q
git config user.name 'SAPO SWE-bench'
git config user.email 'sapo-swebench@localhost'
git config commit.gpgsign false
xargs -0 git add -f -- < /tmp/sapo_tracked_files
git commit -q -m 'SWE-bench base snapshot'
""".strip()
        code, output = self._exec(
            command,
            timeout=float(self.config.get("setup_timeout", 300)),
        )
        if code != 0:
            raise RuntimeError(f"Failed to sanitize Git history (exit {code}): {output[-2000:]}")
        code, base_commit = self._exec("git rev-parse HEAD", timeout=30)
        if code != 0 or not base_commit.strip():
            raise RuntimeError(f"Failed to resolve sanitized base commit: {base_commit}")
        self.metadata["base_commit"] = base_commit.strip()

    def reset(self, sample: Dict[str, Any]):
        self._cleanup_container()
        self.metadata = dict(sample)
        self.instance_id = str(self.metadata["instance_id"])
        self._result = None

        image = self.metadata.get("image") or canonical_image_name(
            self.instance_id,
            prefix=self.config.get("image_prefix", "swebench/sweb.eval.x86_64"),
        )
        client = self._docker()
        if self.config.get("pull_images", True):
            try:
                client.images.get(image)
            except Exception:
                client.images.pull(image)

        container_name = f"sapo-swebench-{uuid.uuid4().hex[:12]}"
        create_kwargs: Dict[str, Any] = {
            "image": image,
            "name": container_name,
            "command": ["tail", "-f", "/dev/null"],
            "detach": True,
            "working_dir": "/testbed",
            "network_disabled": bool(self.config.get("network_disabled", True)),
            "labels": {
                "sapo.environment": "swebench",
                "sapo.instance_id": self.instance_id,
            },
        }
        user = self.config.get("container_user")
        if user:
            create_kwargs["user"] = user
        if self.config.get("cap_add_sys_admin", False):
            create_kwargs["cap_add"] = ["SYS_ADMIN"]

        self.container = client.containers.create(**create_kwargs)
        try:
            self.container.start()

            if self.config.get("sanitize_git_history", True):
                self._sanitize_git_history()

            code, status = self._exec(
                "git status --short && printf '\\nRepository: ' && git remote get-url origin 2>/dev/null || true",
                timeout=float(self.config.get("command_timeout", 120)),
            )
        except Exception:
            self._cleanup_container()
            raise
        observation = (
            f"Repository for {self.instance_id} is ready at /testbed.\n"
            f"Initial status (exit {code}):\n{status.strip() or '<clean>'}"
        )
        info = {
            "instance_id": self.instance_id,
            "image": image,
            "won": False,
            "resolved": False,
            "eval_completed": False,
        }
        return observation, info

    def step(self, action: str):
        if self._result is not None:
            return "Episode already finished.", 0.0, True, self._info_from_result(self._result)

        action = (action or "").strip()
        if not action:
            return "Invalid action: expected one shell command or submit.", 0.0, False, {
                "instance_id": self.instance_id,
                "won": False,
                "resolved": False,
                "eval_completed": False,
            }
        if action.lower() == "submit":
            return self._evaluate()

        exit_code, output = self._exec(
            action,
            timeout=float(self.config.get("command_timeout", 120)),
        )
        clipped = _clip_output(output, int(self.config.get("output_limit", 12000)))
        observation = f"Command exit code: {exit_code}\n{clipped.strip()}"
        if exit_code == 124:
            observation = f"Command timed out.\n{observation}"
        return observation, 0.0, False, {
            "instance_id": self.instance_id,
            "won": False,
            "resolved": False,
            "eval_completed": False,
            "command_exit_code": exit_code,
        }

    def _info_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "won": bool(result.get("resolved", False)),
            "resolved": bool(result.get("resolved", False)),
            "eval_completed": bool(result.get("eval_completed", False)),
            "eval_exit_code": result.get("eval_exit_code"),
            "eval_execution_time": result.get("eval_execution_time"),
            "eval_report": result.get("eval_report"),
        }

    def _evaluate(self):
        if self.metadata is None:
            raise RuntimeError("Cannot evaluate before reset")
        try:
            self._result = compute_reward(
                self.metadata,
                execute_script=self._execute_eval_script,
                eval_timeout=float(self.config.get("eval_timeout", 600)),
            )
            resolved = bool(self._result["resolved"])
            reward = float(self.config.get("reward_success", 10.0)) if resolved else 0.0
            observation = "Evaluation finished: resolved." if resolved else "Evaluation finished: unresolved."
            return observation, reward, True, self._info_from_result(self._result)
        finally:
            self._cleanup_container()

    def finalize(self, active: bool):
        if not active:
            info = self._info_from_result(self._result or {})
            return 0.0, True, info
        _, reward, done, info = self._evaluate()
        return reward, done, info

    def apply_gold_patch(self) -> tuple[int, str]:
        """Smoke-test helper; never called by the training environment."""
        if self.metadata is None or "patch" not in self.metadata:
            raise RuntimeError("Dataset row does not contain a gold patch")
        path = "/tmp/sapo_gold.patch"
        self._put_text_file(path, str(self.metadata["patch"]))
        return self._exec(f"git apply --whitespace=fix {path}", timeout=120)

    def close(self):
        self._cleanup_container()
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None


class SWEBenchMultiProcessEnv:
    """Vectorized collection of per-instance Docker workers."""

    def __init__(
        self,
        env_num: int,
        group_n: int,
        resources_per_worker: Dict[str, Any],
        env_config: Dict[str, Any],
    ) -> None:
        self.batch_size = env_num * group_n
        worker_cls = ray.remote(**resources_per_worker)(SWEBenchWorker)
        self._workers = [worker_cls.remote(env_config) for _ in range(self.batch_size)]
        self._closed = False

    def reset(self, kwargs: List[Dict[str, Any]]):
        if kwargs is None:
            raise ValueError("SWE-bench reset requires per-sample env_kwargs")
        samples = list(kwargs)
        if len(samples) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} SWE-bench samples, got {len(samples)}")
        results = ray.get([worker.reset.remote(dict(sample)) for worker, sample in zip(self._workers, samples)])
        observations, infos = map(list, zip(*results))
        return observations, infos

    def step(self, actions: List[str]):
        if len(actions) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} actions, got {len(actions)}")
        results = ray.get([worker.step.remote(action) for worker, action in zip(self._workers, actions)])
        observations, rewards, dones, infos = map(list, zip(*results))
        return observations, rewards, dones, infos

    def finalize(self, active_masks: List[bool]):
        if len(active_masks) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} active masks, got {len(active_masks)}")
        results = ray.get(
            [worker.finalize.remote(bool(active)) for worker, active in zip(self._workers, active_masks)]
        )
        rewards, dones, infos = map(list, zip(*results))
        return rewards, dones, infos

    def close(self):
        if self._closed:
            return
        ray.get([worker.close.remote() for worker in self._workers])
        for worker in self._workers:
            ray.kill(worker)
        self._closed = True


def build_swebench_envs(
    env_num: int,
    group_n: int,
    resources_per_worker: Dict[str, Any],
    env_config: Dict[str, Any],
    **_: Any,
):
    return SWEBenchMultiProcessEnv(
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        env_config=env_config,
    )
