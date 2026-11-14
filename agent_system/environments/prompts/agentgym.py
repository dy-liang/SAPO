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

SCIWORLD_ACTIONS = (
    "open/close OBJ; activate/deactivate OBJ; connect OBJ to OBJ; disconnect OBJ; "
    "use OBJ [on OBJ]; look around; look at OBJ; look in OBJ; read OBJ; "
    "move OBJ to OBJ; pick up OBJ; put down OBJ; pour OBJ into OBJ; "
    "dunk OBJ into OBJ; mix OBJ; go to LOC; eat OBJ; flush OBJ; "
    "focus on OBJ; wait; wait1; examine OBJ; task; inventory"
)

SCIWORLD_TEMPLATE_NO_HIS = """
You are an autonomous agent in ScienceWorld. Complete the scientific task using one valid action per turn.
Available action forms: {action_forms}.

Current task and observation:
{current_observation}

Respond exactly as:
<think>your concise reasoning</think>
<action>one environment action</action>
"""

SCIWORLD_TEMPLATE = """
You are an autonomous agent in ScienceWorld. Complete the scientific task using one valid action per turn.
Available action forms: {action_forms}.

Recent interaction history:
{action_history}

Current observation:
{current_observation}

Respond exactly as:
<think>your concise reasoning</think>
<action>one environment action</action>
"""

WEBARENA_ACTION_FORMAT = """

Output-format override for this training interface: do not use colon-labelled
reasoning/actions or triple-backtick action blocks. Respond with exactly one
reasoning block and one action block in this format:
<think>your concise reasoning</think>
<action>one valid WebArena action</action>
"""

WEBARENA_TEMPLATE = """
Complete interaction history before the current WebArena state:
{action_history}

Current WebArena state:
{current_observation}
{action_format}
"""
