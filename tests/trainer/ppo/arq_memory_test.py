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

import torch

from verl.trainer.ppo.arq import mask_policy_logits_inplace


def test_inplace_policy_mask_matches_removed_vocabulary_normalization():
    logits = torch.tensor(
        [[1.0, 2.0, -1.0, 0.5, 3.0], [-0.5, 1.5, 2.5, 0.0, -2.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 3])
    masked_logits = logits.clone()
    mask_policy_logits_inplace(masked_logits, (1, 4))

    actual = torch.log_softmax(masked_logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    kept_logits = logits[:, [0, 2, 3]]
    kept_labels = torch.tensor([0, 2])
    expected = torch.log_softmax(kept_logits, dim=-1).gather(
        -1, kept_labels.unsqueeze(-1)
    ).squeeze(-1)
    torch.testing.assert_close(actual, expected)


def test_policy_mask_preserves_reserved_token_value_gradient_branch():
    source = torch.randn(4, 8, requires_grad=True)
    logits = source * 1.0
    # ARQ gathers its scalar branch before policy masking.
    value_branch = logits[:, 2].clone()
    mask_policy_logits_inplace(logits, (2, 6))
    labels = torch.tensor([0, 1, 3, 7])
    policy_branch = torch.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).sum()

    (policy_branch + value_branch.sum()).backward()
    assert torch.isfinite(source.grad).all()
    # Reserved token 2 receives value gradient even though policy probability
    # and policy gradient for it are zero.
    torch.testing.assert_close(source.grad[:, 2], torch.ones(4))


def test_policy_mask_keeps_entropy_finite():
    logits = torch.randn(3, 11, dtype=torch.bfloat16)
    mask_policy_logits_inplace(logits, (2, 7))
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    assert torch.isfinite(entropy).all()
