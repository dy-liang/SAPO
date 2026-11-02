"""Project model text into one SWE-bench shell/control action."""

from __future__ import annotations

import re
from typing import List, Tuple


_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def swebench_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Extract exactly one ``<action>`` block from every model response.

    The action is either a shell command executed from ``/testbed`` or the
    literal ``submit``.  Malformed responses are returned as empty actions and
    marked invalid so SAPO can apply its invalid-action penalty.
    """

    projected: List[str] = []
    valids: List[int] = []

    for response in actions:
        matches = _ACTION_RE.findall(response or "")
        has_one_thought = len(_THINK_RE.findall(response or "")) == 1
        valid = len(matches) == 1 and has_one_thought
        action = matches[0].strip() if len(matches) == 1 else ""

        # Empty commands and nested action tags are always invalid.
        if not action or "<action>" in action.lower() or "</action>" in action.lower():
            valid = False
            action = ""

        projected.append(action)
        valids.append(int(valid))

    return projected, valids
