"""Prompts for the bash-only SWE-bench agent scaffold."""

SWEBENCH_TEMPLATE_NO_HIS = """
You are an autonomous software-engineering agent. A repository is available at
/testbed. Resolve the issue below by inspecting and editing non-test source files.

Issue:
{task_description}

Each action runs from /testbed in an isolated container. You may use normal shell
commands to inspect files, edit files, and run relevant tests. Do not modify existing
tests, do not look for a gold patch, and do not access the network. When the solution
is ready, use the literal action `submit`.

Current observation:
{current_observation}

Respond with exactly one reasoning block and one action block:
<think>brief reasoning about the next step</think>
<action>one shell command, or submit</action>
""".strip()


SWEBENCH_TEMPLATE = """
You are an autonomous software-engineering agent. A repository is available at
/testbed. Resolve the issue below by inspecting and editing non-test source files.

Issue:
{task_description}

You have taken {step_count} action(s). Recent interaction history:
{action_history}

Current observation at step {current_step}:
{current_observation}

Each action runs from /testbed. Use one shell command per turn. Do not modify
existing tests. When the implementation is ready, use the literal action `submit`.

Respond with exactly:
<think>brief reasoning about the next step</think>
<action>one shell command, or submit</action>
""".strip()
