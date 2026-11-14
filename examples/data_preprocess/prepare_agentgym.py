#!/usr/bin/env python3
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

"""Create SAPO parquet datasets from the official AgentGym-RL task-ID split."""

import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen

from datasets import Dataset


DATASET_BASE = "https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID/resolve/main"


def _load_records(source: str):
    if source.startswith(("http://", "https://")):
        with urlopen(source, timeout=60) as response:
            return json.load(response)
    with open(os.path.expanduser(source), encoding="utf-8") as handle:
        return json.load(handle)


def _task_id(record, environment: str) -> int:
    item_id = str(record["item_id"])
    prefix = f"{environment}_"
    if not item_id.startswith(prefix):
        raise ValueError(f"Unexpected item_id {item_id!r}; expected prefix {prefix!r}")
    return int(item_id[len(prefix) :])


def _rows(records, environment: str, split: str, limit: int):
    if limit > 0:
        records = records[:limit]
    rows = []
    for index, record in enumerate(records):
        task_id = _task_id(record, environment)
        rows.append(
            {
                "data_source": environment,
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "env_kwargs": {"task_id": task_id},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "item_id": record["item_id"],
                },
            }
        )
    if not rows:
        raise ValueError(f"No {split} records found for {environment}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True, choices=["sciworld", "webarena"])
    parser.add_argument("--local_dir", default="~/data/verl-agent/agentgym")
    parser.add_argument("--train_id_file", default=None)
    parser.add_argument("--val_id_file", default=None)
    parser.add_argument("--train_data_size", type=int, default=0, help="0 keeps the complete official split")
    parser.add_argument("--val_data_size", type=int, default=0, help="0 keeps the complete official split")
    args = parser.parse_args()

    train_source = args.train_id_file or (
        f"{DATASET_BASE}/train/{args.environment}_train.json"
    )
    val_source = args.val_id_file or (
        f"{DATASET_BASE}/eval/{args.environment}_test.json"
    )
    output_dir = Path(os.path.expanduser(args.local_dir)) / args.environment
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _rows(
        _load_records(train_source), args.environment, "train", args.train_data_size
    )
    val_rows = _rows(
        _load_records(val_source), args.environment, "test", args.val_data_size
    )
    Dataset.from_list(train_rows).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(val_rows).to_parquet(str(output_dir / "test.parquet"))
    print(
        f"Prepared {len(train_rows)} train and {len(val_rows)} test tasks in {output_dir}"
    )


if __name__ == "__main__":
    main()
