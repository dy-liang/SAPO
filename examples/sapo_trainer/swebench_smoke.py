"""Run the SWE evaluator on one empty or gold-patched repository.

This is intentionally independent of GPUs and model inference.  It validates
dataset metadata, Docker image provisioning, patch application, official test
parsing, reward scaling, and container cleanup before an expensive RL run.
"""

from __future__ import annotations

import argparse
import json

from datasets import load_dataset

from agent_system.environments.env_package.swebench.envs import SWEBenchWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--gold", action="store_true", help="Apply the gold patch before evaluation")
    parser.add_argument("--keep-network", action="store_true")
    parser.add_argument("--eval-timeout", type=float, default=600)
    args = parser.parse_args()

    dataset = load_dataset("parquet", data_files=args.parquet, split="train")
    row = dataset[args.index]
    sample = dict(row["env_kwargs"])
    worker = SWEBenchWorker(
        {
            "pull_images": True,
            "network_disabled": not args.keep_network,
            "sanitize_git_history": True,
            "container_user": "root",
            "setup_timeout": 300,
            "command_timeout": 120,
            "eval_timeout": args.eval_timeout,
            "output_limit": 12000,
            "reward_success": 10.0,
        }
    )
    try:
        observation, info = worker.reset(sample)
        print(observation)
        print(json.dumps(info, indent=2, default=str))
        if args.gold:
            code, output = worker.apply_gold_patch()
            print(f"gold patch exit_code={code}\n{output}")
            if code != 0:
                raise SystemExit("Gold patch failed to apply")
        observation, reward, done, info = worker.step("submit")
        print(observation)
        print(f"reward={reward} done={done}")
        print(json.dumps(info, indent=2, default=str))

        expected_resolved = bool(args.gold)
        if bool(info["resolved"]) != expected_resolved:
            mode = "gold" if args.gold else "empty"
            raise SystemExit(f"Unexpected {mode}-patch result: resolved={info['resolved']}")
    finally:
        worker.close()


if __name__ == "__main__":
    main()
