"""Official SWE-bench-style evaluation inside an already-running container."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Tuple


START_TEST_OUTPUT = "SWE_BENCH_TEST_OUTPUT_START"
END_TEST_OUTPUT = "SWE_BENCH_TEST_OUTPUT_END"
_PATCH_DELIMITER = "EOF_SAPO_SWEBENCH_TEST_PATCH_4F6A"


def _json_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def build_eval_script(metadata: Dict[str, Any]) -> str:
    """Build the same test sequence used by the official SWE-bench harness.

    Importing ``swebench`` is intentionally delayed until evaluation so the
    rest of the repository remains usable without the optional dependency.
    """

    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        from swebench.harness.test_spec.python import get_test_directives
        from swebench.harness.utils import get_modified_files
    except ImportError as exc:
        raise RuntimeError(
            "SWE-bench evaluation requires optional dependencies. Install "
            "`pip install -r requirements-swebench.txt`."
        ) from exc

    repo = metadata["repo"]
    version = str(metadata["version"])
    base_commit = metadata["base_commit"]
    test_patch = metadata["test_patch"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    test_files = get_modified_files(test_patch)
    if test_files:
        quoted_files = " ".join(_shell_quote(path) for path in test_files)
        reset_tests = f"git checkout {_shell_quote(base_commit)} -- {quoted_files}"
    else:
        reset_tests = "echo 'No test files to reset'"

    test_cmd = specs["test_cmd"]
    directives = get_test_directives(metadata)
    command = " ".join([test_cmd, *directives])

    commands = [
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        "cd /testbed",
        "git config --global --add safe.directory /testbed",
        "git status --short",
        f"git -c core.fileMode=false diff {_shell_quote(base_commit)}",
    ]
    commands.extend(specs.get("eval_commands", []))
    if "install" in specs:
        commands.append(specs["install"])
    commands.extend(
        [
            reset_tests,
            f"git apply -v - <<'{_PATCH_DELIMITER}'\n{test_patch}\n{_PATCH_DELIMITER}",
            f"echo '{START_TEST_OUTPUT}'",
            command,
            f"echo '{END_TEST_OUTPUT}'",
            reset_tests,
        ]
    )
    return "\n".join(["#!/bin/bash", "set -uxo pipefail", *commands, ""])


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def parse_eval_output(metadata: Dict[str, Any], output: str) -> Dict[str, Any]:
    """Parse test output and apply official F2P/P2P resolution semantics."""

    try:
        from swebench.harness.constants import FAIL_ONLY_REPOS, EvalType, ResolvedStatus
        from swebench.harness.grading import get_eval_tests_report, get_resolution_status
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    except ImportError as exc:
        raise RuntimeError(
            "SWE-bench evaluation requires optional dependencies. Install "
            "`pip install -r requirements-swebench.txt`."
        ) from exc

    report: Dict[str, Any] = {
        "resolved": False,
        "found_eval_status": False,
        "status_map": {},
        "test_status": None,
    }
    if START_TEST_OUTPUT not in output or END_TEST_OUTPUT not in output:
        return report

    test_output = output.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    repo = metadata["repo"]
    status_map = MAP_REPO_TO_PARSER[repo](test_output, None)
    report["found_eval_status"] = True
    report["status_map"] = status_map

    reference = {
        "instance_id": metadata["instance_id"],
        "FAIL_TO_PASS": _json_list(metadata.get("FAIL_TO_PASS")),
        "PASS_TO_PASS": _json_list(metadata.get("PASS_TO_PASS")),
    }
    eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
    test_status = get_eval_tests_report(status_map, reference, eval_type=eval_type)
    report["test_status"] = test_status
    report["resolved"] = get_resolution_status(test_status) == ResolvedStatus.FULL.value
    return report


def compute_reward(
    metadata: Dict[str, Any],
    execute_script: Callable[[str, float], Tuple[int, str]],
    eval_timeout: float = 600.0,
) -> Dict[str, Any]:
    """Execute and grade one instance.

    ``execute_script`` is supplied by the Docker worker, keeping Docker details
    out of the reward implementation and making the evaluator easy to test.
    """

    import time

    script = build_eval_script(metadata)
    started = time.perf_counter()
    exit_code, output = execute_script(script, eval_timeout)
    elapsed = time.perf_counter() - started
    eval_report = parse_eval_output(metadata, output)
    return {
        "eval_completed": exit_code == 0,
        "eval_exit_code": int(exit_code),
        "eval_execution_time": elapsed,
        "eval_report": eval_report,
        "resolved": bool(eval_report["resolved"]),
    }
