# Copyright 2026
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

import pytest
import torch
import torch.nn.functional as F

from verl.trainer.ppo.arq import mask_arq_policy_logits_, project_arq_boundary_logits
from verl.utils.torch_functional import entropy_from_logits


def _reference_outputs(logits, labels, excluded_token_ids):
    kept_token_ids = [
        token_id
        for token_id in range(logits.shape[-1])
        if token_id not in excluded_token_ids
    ]
    kept_logits = logits[..., kept_token_ids]
    kept_log_probs = torch.log_softmax(kept_logits, dim=-1)
    entropy = -(kept_log_probs.exp() * kept_log_probs).sum(dim=-1)
    label_remap = {
        token_id: kept_index for kept_index, token_id in enumerate(kept_token_ids)
    }
    kept_labels = torch.as_tensor(
        [label_remap[int(label)] for label in labels.reshape(-1)],
        dtype=torch.long,
        device=labels.device,
    ).reshape_as(labels)
    selected_log_probs = kept_log_probs.gather(
        -1, kept_labels.unsqueeze(-1)
    ).squeeze(-1)
    return selected_log_probs, entropy


@pytest.mark.parametrize("shape", [(3, 11), (2, 3, 13)])
def test_inplace_mask_matches_dense_exclusion_reference(shape):
    torch.manual_seed(7)
    excluded_token_ids = (2, shape[-1] - 2)
    labels = torch.randint(0, shape[-1], shape[:-1])
    for token_id in excluded_token_ids:
        labels = torch.where(labels == token_id, torch.zeros_like(labels), labels)

    original_logits = torch.randn(*shape, dtype=torch.float64)
    policy_logits = original_logits.clone()
    mask_arq_policy_logits_(policy_logits, excluded_token_ids)
    actual_log_probs = F.log_softmax(policy_logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    actual_entropy = entropy_from_logits(policy_logits)
    expected_log_probs, expected_entropy = _reference_outputs(
        original_logits, labels, excluded_token_ids
    )

    torch.testing.assert_close(actual_log_probs, expected_log_probs)
    torch.testing.assert_close(actual_entropy, expected_entropy)
    assert torch.isfinite(actual_entropy).all()


def test_inplace_mask_has_zero_policy_gradient_for_value_tokens():
    base_logits = torch.randn(4, 17, dtype=torch.float64, requires_grad=True)
    policy_logits = base_logits * 1.0  # non-leaf, matching model output semantics
    excluded_token_ids = (2, 13)
    labels = torch.tensor([0, 1, 3, 4])

    mask_arq_policy_logits_(policy_logits, excluded_token_ids)
    log_probs = F.log_softmax(policy_logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    (log_probs.sum() + entropy_from_logits(policy_logits).sum()).backward()

    torch.testing.assert_close(base_logits.grad[:, 2], torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(base_logits.grad[:, 13], torch.zeros(4, dtype=torch.float64))
    assert torch.isfinite(base_logits.grad).all()


@pytest.mark.parametrize("remove_padding", [False, True])
def test_boundary_projection_matches_full_lm_head(remove_padding):
    torch.manual_seed(11)
    hidden_states = torch.randn(2, 5, 7, dtype=torch.float64)
    boundary_positions = torch.tensor([[1, 4], [0, 3]])
    if remove_padding:
        hidden_states = hidden_states.reshape(1, 10, 7)
        boundary_positions = torch.tensor([[1, 4], [5, 8]])
    weight = torch.randn(13, 7, dtype=torch.float64)
    bias = torch.randn(13, dtype=torch.float64)
    value_token_ids = (2, 11)

    actual = project_arq_boundary_logits(
        hidden_states=hidden_states,
        boundary_positions=boundary_positions,
        lm_head_weight=weight,
        lm_head_bias=bias,
        value_token_ids=value_token_ids,
    )
    full_logits = F.linear(hidden_states, weight, bias)
    if remove_padding:
        full_logits = full_logits.squeeze(0)
        expected = full_logits[
            boundary_positions.unsqueeze(-1),
            torch.tensor(value_token_ids).view(1, 1, -1),
        ]
    else:
        batch_indices = torch.arange(hidden_states.shape[0]).view(-1, 1, 1)
        expected = full_logits[
            batch_indices,
            boundary_positions.unsqueeze(-1),
            torch.tensor(value_token_ids).view(1, 1, -1),
        ]
    torch.testing.assert_close(actual, expected)


def test_boundary_projection_gradients_are_sparse_in_token_and_sequence_axes():
    hidden_states = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(13, 7, dtype=torch.float64, requires_grad=True)
    boundary_positions = torch.tensor([[1, 4], [0, 3]])
    value_token_ids = (2, 11)

    projected = project_arq_boundary_logits(
        hidden_states=hidden_states,
        boundary_positions=boundary_positions,
        lm_head_weight=weight,
        value_token_ids=value_token_ids,
    )
    projected.sum().backward()

    hidden_non_boundary = torch.ones(2, 5, dtype=torch.bool)
    hidden_non_boundary[torch.arange(2).unsqueeze(-1), boundary_positions] = False
    assert torch.count_nonzero(hidden_states.grad[hidden_non_boundary]) == 0
    weight_non_value = torch.ones(13, dtype=torch.bool)
    weight_non_value[list(value_token_ids)] = False
    assert torch.count_nonzero(weight.grad[weight_non_value]) == 0
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_inplace_mask_rejects_invalid_token_id():
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        mask_arq_policy_logits_(torch.randn(2, 7), (2, 7))
