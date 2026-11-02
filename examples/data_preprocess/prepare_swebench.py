"""Convert SWE-bench-style datasets to SAPO's per-environment parquet format.

Examples:

    python -m examples.data_preprocess.prepare_swebench \
        --dataset princeton-nlp/SWE-bench_Verified \
        --split test \
        --output ~/data/sapo-swe/swe_bench_verified.parquet

Use a SWE-bench-compatible training split and reserve SWE-bench Verified for
validation.  The installed ``swebench`` evaluator must recognize every
repository/version pair in the source dataset.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

from datasets import load_dataset

from agent_system.environments.env_package.swebench.utils import canonical_image_name


def _first(example: Dict[str, Any], *names: str, default=None):
    for name in names:
        if name in example and example[name] is not None:
            return example[name]
    return default


def convert_example(example: Dict[str, Any], index: int, image_prefix: str) -> Dict[str, Any]:
    instance_id = str(_first(example, "instance_id", "id"))
    problem_statement = str(_first(example, "problem_statement", "problem", "issue"))
    if instance_id == "None" or problem_statement == "None":
        raise ValueError("Each SWE row must contain instance_id and problem_statement")

    image = _first(example, "image", "docker_image", "image_name")
    if not image:
        image = canonical_image_name(instance_id, prefix=image_prefix)

    metadata = {
        "instance_id": instance_id,
        "repo": _first(example, "repo", "repository"),
        "version": str(_first(example, "version", default="")),
        "base_commit": _first(example, "base_commit"),
        "patch": _first(example, "patch", default=""),
        "test_patch": _first(example, "test_patch", default=""),
        "problem_statement": problem_statement,
        "FAIL_TO_PASS": _first(example, "FAIL_TO_PASS", "fail_to_pass", default="[]"),
        "PASS_TO_PASS": _first(example, "PASS_TO_PASS", "pass_to_pass", default="[]"),
        "image": str(image),
    }
    required = ("repo", "base_commit", "test_patch")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(f"{instance_id} is missing required evaluator fields: {missing}")

    return {
        "data_source": "swebench",
        "prompt": [{"role": "user", "content": problem_statement}],
        "ability": "agent",
        "extra_info": {"split_index": index, "instance_id": instance_id},
        # This field is consumed by TrajectoryCollector.reset and never rendered
        # into the model prompt.  Hidden tests and the gold patch stay host-side.
        "env_kwargs": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name or local dataset path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--image-prefix", default="swebench/sweb.eval.x86_64")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_instances is not None:
        dataset = dataset.select(range(min(max(args.max_instances, 0), len(dataset))))

    dataset = dataset.map(
        lambda example, index: convert_example(example, index, args.image_prefix),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Converting SWE-bench rows",
    )
    output = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    dataset.to_parquet(output)
    print(f"Wrote {len(dataset)} SWE instances to {output}")


if __name__ == "__main__":
    main()
