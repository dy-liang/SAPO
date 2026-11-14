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

import re
from typing import List, Tuple


_CHINESE = re.compile(r"[\u4e00-\u9fff]")


def _extract_action(text: str) -> Tuple[str, bool]:
    """Apply the same <think>/<action> validity contract as existing envs."""
    original = str(text or "")
    start_tag, end_tag = "<action>", "</action>"
    start_idx = original.find(start_tag)
    end_idx = original.find(end_tag, start_idx + len(start_tag))
    if start_idx == -1 or end_idx == -1:
        return original[-20:], False

    action = original[start_idx + len(start_tag) : end_idx].strip()
    valid = bool(action)
    if original.find("<think>") == -1 or original.find("</think>") == -1:
        valid = False
    if _CHINESE.search(original):
        valid = False
    return action, valid


def sciworld_projection(actions: List[str]):
    projected, valids = [], []
    for text in actions:
        action, valid = _extract_action(text)
        projected.append(action.lower())
        valids.append(int(valid))
    return projected, valids


def webarena_projection(actions: List[str]):
    """Keep SAPO tags model-side; translate only the extracted server action."""
    projected, valids = [], []
    for text in actions:
        action, valid = _extract_action(text)
        projected.append(f"```{action}```")
        valids.append(int(valid))
    return projected, valids
