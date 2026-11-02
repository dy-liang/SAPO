"""SWE-bench environment integration.

The heavy Docker/Ray dependencies are imported lazily so preprocessing and
projection unit tests do not require a running Docker daemon.
"""

from .projection import swebench_projection


def build_swebench_envs(*args, **kwargs):
    from .envs import build_swebench_envs as _build_swebench_envs

    return _build_swebench_envs(*args, **kwargs)


__all__ = ["build_swebench_envs", "swebench_projection"]
