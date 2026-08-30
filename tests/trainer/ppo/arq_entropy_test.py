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

from verl.trainer.ppo.arq import masked_policy_logprobs_and_entropy


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
def test_chunked_masked_entropy_matches_dense_reference(shape):
    torch.manual_seed(7)
    excluded_token_ids = (2, shape[-1] - 2)
    labels = torch.randint(0, shape[-1], shape[:-1])
    for token_id in excluded_token_ids:
        labels = torch.where(labels == token_id, torch.zeros_like(labels), labels)
    logits = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)

    actual_log_probs, actual_entropy = masked_policy_logprobs_and_entropy(
        logits=logits,
        labels=labels,
        excluded_token_ids=excluded_token_ids,
        calculate_entropy=True,
        entropy_vocab_chunk_size=3,
    )
    expected_log_probs, expected_entropy = _reference_outputs(
        reference_logits, labels, excluded_token_ids
    )

    torch.testing.assert_close(actual_log_probs, expected_log_probs)
    torch.testing.assert_close(actual_entropy, expected_entropy)

    weights = torch.linspace(
        0.25, 1.25, actual_entropy.numel(), dtype=torch.float64
    ).reshape_as(actual_entropy)
    (actual_log_probs.sum() + (actual_entropy * weights).sum()).backward()
    (expected_log_probs.sum() + (expected_entropy * weights).sum()).backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_chunked_masked_entropy_has_zero_reserved_token_gradient():
    logits = torch.randn(4, 17, dtype=torch.float32, requires_grad=True)
    _, entropy = masked_policy_logprobs_and_entropy(
        logits=logits,
        labels=torch.tensor([0, 1, 3, 4]),
        excluded_token_ids=(2, 13),
        calculate_entropy=True,
        entropy_vocab_chunk_size=4,
    )
    entropy.sum().backward()

    torch.testing.assert_close(logits.grad[:, 2], torch.zeros(4))
    torch.testing.assert_close(logits.grad[:, 13], torch.zeros(4))
    assert torch.isfinite(logits.grad).all()


def test_chunked_masked_entropy_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError, match="chunk size must be positive"):
        masked_policy_logprobs_and_entropy(
            logits=torch.randn(2, 7),
            labels=torch.tensor([0, 1]),
            excluded_token_ids=(2, 5),
            calculate_entropy=True,
            entropy_vocab_chunk_size=0,
        )
