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

from verl.trainer.ppo.arq import mask_arq_policy_logits_
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


def test_value_gradient_survives_policy_mask_without_policy_leakage():
    base_logits = torch.randn(3, 11, dtype=torch.float64, requires_grad=True)
    raw_logits = base_logits * 1.0
    value_token_ids = torch.tensor([2, 9])
    boundary_rows = torch.tensor([0, 2])
    value_logits = raw_logits[
        boundary_rows.unsqueeze(-1), value_token_ids.unsqueeze(0)
    ]

    # Match the actor: extract value scalars first, then reuse raw-logit
    # storage for temperature scaling and policy masking.
    policy_logits = raw_logits.div_(0.7)
    mask_arq_policy_logits_(policy_logits, value_token_ids.tolist())
    labels = torch.tensor([0, 1, 3])
    policy_log_probs = F.log_softmax(policy_logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    (policy_log_probs.sum() + value_logits.sum()).backward()

    expected_reserved_grad = torch.zeros(3, 2, dtype=torch.float64)
    expected_reserved_grad[boundary_rows] = 1.0
    torch.testing.assert_close(base_logits.grad[:, value_token_ids], expected_reserved_grad)
    assert torch.isfinite(base_logits.grad).all()


def test_inplace_mask_rejects_invalid_token_id():
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        mask_arq_policy_logits_(torch.randn(2, 7), (2, 7))
