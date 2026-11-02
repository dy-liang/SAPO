"""Small dependency-free helpers shared by preprocessing and runtime."""

from __future__ import annotations


def canonical_image_name(
    instance_id: str,
    prefix: str = "swebench/sweb.eval.x86_64",
) -> str:
    normalized = instance_id.lower().replace("__", "_1776_").replace("/", "_1776_")
    return f"{prefix}.{normalized}"
