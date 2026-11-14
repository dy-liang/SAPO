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
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from datasets import Dataset


HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
DEFAULT_DATASET_REPO = os.environ.get(
    "AGENTGYM_DATASET_REPO",
    f"{HF_ENDPOINT}/datasets/AgentGym/AgentGym-RL-Data-ID",
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "AGENTGYM_DATA_DIR",
    "~/data/verl-agent/agentgym",
)


class _MirrorRedirectHandler(HTTPRedirectHandler):
    """Add 308 support for mirrors when the Python runtime lacks it."""

    def http_error_308(self, request, response, code, message, headers):
        return self.http_error_302(request, response, 302, message, headers)


def _dataset_file(dataset_repo: str, relative_path: str) -> str:
    """Resolve a file from an HF repo URL, mirror URL, or local clone."""
    dataset_repo = os.path.expanduser(dataset_repo).rstrip("/")
    if dataset_repo.startswith(("http://", "https://")):
        if dataset_repo.endswith("/tree/main"):
            dataset_repo = dataset_repo[: -len("/tree/main")]
        if not dataset_repo.endswith("/resolve/main"):
            dataset_repo = f"{dataset_repo}/resolve/main"
        return f"{dataset_repo}/{relative_path}"
    return str(Path(dataset_repo) / relative_path)


def _load_records(source: str):
    if source.startswith(("http://", "https://")):
        request = Request(
            source,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
            },
        )
        try:
            with build_opener(_MirrorRedirectHandler()).open(
                request, timeout=60
            ) as response:
                return json.load(response)
        except (HTTPError, URLError, ValueError) as exc:
            raise RuntimeError(
                f"Unable to download AgentGym task IDs from {source!r}: {exc}. "
                "If the Hugging Face endpoint blocks Python downloads, clone "
                "AgentGym-RL-Data-ID and set AGENTGYM_DATASET_REPO to that "
                "local directory."
            ) from exc
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
    parser.add_argument(
        "--local_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root output directory. Defaults to AGENTGYM_DATA_DIR or "
            "~/data/verl-agent/agentgym. The environment name is appended once."
        ),
    )
    parser.add_argument(
        "--dataset_repo",
        default=DEFAULT_DATASET_REPO,
        help=(
            "Hugging Face dataset repository URL or a local git clone. "
            "AGENTGYM_DATASET_REPO and HF_ENDPOINT are honored when this "
            "option is omitted."
        ),
    )
    parser.add_argument("--train_id_file", default=None)
    parser.add_argument("--val_id_file", default=None)
    parser.add_argument("--train_data_size", type=int, default=0, help="0 keeps the complete official split")
    parser.add_argument("--val_data_size", type=int, default=0, help="0 keeps the complete official split")
    args = parser.parse_args()

    train_source = args.train_id_file or _dataset_file(
        args.dataset_repo,
        f"train/{args.environment}_train.json",
    )
    val_source = args.val_id_file or _dataset_file(
        args.dataset_repo,
        f"eval/{args.environment}_test.json",
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
