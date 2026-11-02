from __future__ import annotations

import sys
import types
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "agent_system/environments/env_package/swebench"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


projection_module = _load_module("sapo_swebench_projection", "projection.py")
reward_module = _load_module("sapo_swebench_reward", "reward.py")
utils_module = _load_module("sapo_swebench_utils", "utils.py")

swebench_projection = projection_module.swebench_projection
START_TEST_OUTPUT = reward_module.START_TEST_OUTPUT
END_TEST_OUTPUT = reward_module.END_TEST_OUTPUT
_json_list = reward_module._json_list
parse_eval_output = reward_module.parse_eval_output
canonical_image_name = utils_module.canonical_image_name


class ProjectionTest(unittest.TestCase):
    def test_extracts_shell_action(self):
        actions, valids = swebench_projection(
            ["<think>inspect files</think><action>rg -n 'bug' src</action>"]
        )
        self.assertEqual(actions, ["rg -n 'bug' src"])
        self.assertEqual(valids, [1])

    def test_accepts_submit(self):
        actions, valids = swebench_projection(["<think>tests pass</think><action>submit</action>"])
        self.assertEqual(actions, ["submit"])
        self.assertEqual(valids, [1])

    def test_rejects_multiple_actions_or_missing_thought(self):
        actions, valids = swebench_projection(
            [
                "<think>x</think><action>pwd</action><action>submit</action>",
                "<action>pwd</action>",
            ]
        )
        self.assertEqual(actions, ["", "pwd"])
        self.assertEqual(valids, [0, 0])


class HelpersTest(unittest.TestCase):
    def test_canonical_image_name(self):
        self.assertEqual(
            canonical_image_name("django__django-11099"),
            "swebench/sweb.eval.x86_64.django_1776_django-11099",
        )

    def test_json_list(self):
        self.assertEqual(_json_list('["a", "b"]'), ["a", "b"])
        self.assertEqual(_json_list(["a"]), ["a"])


class RewardParsingTest(unittest.TestCase):
    @staticmethod
    def _fake_swebench_modules():
        constants = types.ModuleType("swebench.harness.constants")

        class EvalType:
            FAIL_ONLY = "fail-only"
            PASS_AND_FAIL = "pass-and-fail"

        class ResolvedStatus:
            FULL = types.SimpleNamespace(value="full")

        constants.FAIL_ONLY_REPOS = set()
        constants.EvalType = EvalType
        constants.ResolvedStatus = ResolvedStatus

        grading = types.ModuleType("swebench.harness.grading")
        grading.get_eval_tests_report = lambda status, reference, eval_type: {
            "status": status,
            "reference": reference,
            "eval_type": eval_type,
        }
        grading.get_resolution_status = lambda report: (
            "full" if report["status"].get("target") == "PASSED" else "unresolved"
        )

        parsers = types.ModuleType("swebench.harness.log_parsers")
        parsers.MAP_REPO_TO_PARSER = {"owner/repo": lambda output, _: {"target": output.strip()}}

        return {
            "swebench": types.ModuleType("swebench"),
            "swebench.harness": types.ModuleType("swebench.harness"),
            "swebench.harness.constants": constants,
            "swebench.harness.grading": grading,
            "swebench.harness.log_parsers": parsers,
        }

    def test_missing_markers_is_unresolved(self):
        with patch.dict(sys.modules, self._fake_swebench_modules()):
            report = parse_eval_output(
                {"repo": "owner/repo", "instance_id": "x", "FAIL_TO_PASS": [], "PASS_TO_PASS": []},
                "no test markers",
            )
        self.assertFalse(report["resolved"])
        self.assertFalse(report["found_eval_status"])

    def test_official_parser_result_controls_resolution(self):
        output = f"before\n{START_TEST_OUTPUT}\nPASSED\n{END_TEST_OUTPUT}\nafter"
        with patch.dict(sys.modules, self._fake_swebench_modules()):
            report = parse_eval_output(
                {
                    "repo": "owner/repo",
                    "instance_id": "owner__repo-1",
                    "FAIL_TO_PASS": '["target"]',
                    "PASS_TO_PASS": "[]",
                },
                output,
            )
        self.assertTrue(report["found_eval_status"])
        self.assertTrue(report["resolved"])


if __name__ == "__main__":
    unittest.main()
