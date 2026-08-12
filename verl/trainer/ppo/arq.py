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
"""Utilities shared by the autoregressive actor-Q PPO implementation.

This module intentionally has no dependency on Ray or vLLM.  Keeping the
indexing and probability transforms here makes them directly unit testable and
keeps the regular PPO/GRPO paths unchanged.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


ARQ_OLD_TENSOR_KEYS = (
    "old_v_probs",
    "old_q_probs",
    "old_v_values",
    "old_q_values",
    "old_v_logit_gaps",
    "old_q_logit_gaps",
)

# vLLM's string-based ``bad_words`` path is disproportionately expensive for
# large batched rollouts in some supported vLLM releases.  A two-entry logit
# bias uses the native token-id path and makes the value tokens effectively
# impossible to sample without changing the rest of the policy distribution.
ARQ_ROLLOUT_LOGIT_BIAS = -100.0


def add_arq_rollout_logit_bias(
    sampling_kwargs: dict,
    excluded_token_ids: Iterable[int],
) -> dict:
    """Return sampling kwargs with efficient token-id ARQ exclusions.

    Existing user-supplied logit biases are preserved.  ARQ only strengthens
    the bias for its two reserved value tokens.
    """
    merged_kwargs = dict(sampling_kwargs)
    configured_bias = merged_kwargs.get("logit_bias") or {}
    logit_bias = {int(token_id): float(bias) for token_id, bias in dict(configured_bias).items()}

    for token_id in excluded_token_ids:
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError(f"ARQ excluded token ID must be non-negative, got {token_id}")
        previous_bias = logit_bias.get(token_id)
        logit_bias[token_id] = (
            ARQ_ROLLOUT_LOGIT_BIAS
            if previous_bias is None
            else min(previous_bias, ARQ_ROLLOUT_LOGIT_BIAS)
        )

    merged_kwargs["logit_bias"] = logit_bias
    return merged_kwargs


def _as_single_token_id(tokenizer, token: Optional[str], token_id: Optional[int], name: str) -> tuple[int, str]:
    """Resolve one configured value token and verify that it is atomic."""
    resolved_from_text = None
    if token is not None:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"ARQ {name}_token must encode to exactly one token, got {encoded!r} for {token!r}")
        resolved_from_text = int(encoded[0])

    if token_id is None and resolved_from_text is None:
        raise ValueError(f"ARQ requires either {name}_token or {name}_token_id")

    resolved_id = int(token_id) if token_id is not None else resolved_from_text
    if resolved_from_text is not None and resolved_id != resolved_from_text:
        raise ValueError(
            f"ARQ {name}_token_id={resolved_id} disagrees with tokenizer encoding "
            f"{resolved_from_text} for {token!r}"
        )

    if resolved_id < 0 or resolved_id >= len(tokenizer):
        raise ValueError(f"ARQ {name}_token_id={resolved_id} is outside tokenizer vocabulary [0, {len(tokenizer)})")

    if token is None:
        token = tokenizer.decode([resolved_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [resolved_id]:
            raise ValueError(
                f"ARQ {name}_token_id={resolved_id} cannot be represented as a stable single-token string; "
                "configure the token string explicitly"
            )

    return resolved_id, token


def resolve_value_tokens(tokenizer, config) -> dict[str, object]:
    """Resolve and validate the positive/negative ARQ token pair.

    The tokens are allowed to be otherwise-special, low-frequency vocabulary
    entries (for example unused vision delimiters in a text-only run), but not
    BOS/EOS/PAD or the Qwen chat-template delimiters.
    """
    positive_id, positive_token = _as_single_token_id(
        tokenizer,
        config.get("positive_token"),
        config.get("positive_token_id"),
        "positive",
    )
    negative_id, negative_token = _as_single_token_id(
        tokenizer,
        config.get("negative_token"),
        config.get("negative_token_id"),
        "negative",
    )

    if positive_id == negative_id:
        raise ValueError("ARQ positive and negative value tokens must be distinct")

    forbidden_ids = {
        token_id
        for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    if positive_id in forbidden_ids or negative_id in forbidden_ids:
        raise ValueError("ARQ value tokens cannot be BOS, EOS, or PAD")

    forbidden_chat_tokens = {"<|im_start|>", "<|im_end|>"}
    if positive_token in forbidden_chat_tokens or negative_token in forbidden_chat_tokens:
        raise ValueError("ARQ value tokens cannot be Qwen chat-template delimiters")

    return {
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
        "positive_token": positive_token,
        "negative_token": negative_token,
    }


def get_arq_boundary_positions(attention_mask: torch.Tensor, response_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded-sequence positions for V(s) and Q(s,a) next-token logits.

    Prompts are left padded to a fixed width and responses are right padded.
    The logit at position i predicts token i+1, so the last prompt position is
    the pre-action V boundary and the last valid response position is the
    post-action Q boundary.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be rank 2, got shape {tuple(attention_mask.shape)}")
    if response_length <= 0 or response_length >= attention_mask.shape[1]:
        raise ValueError(
            f"response_length must be in [1, sequence_length), got {response_length} for {attention_mask.shape[1]}"
        )

    batch_size, sequence_length = attention_mask.shape
    prompt_width = sequence_length - response_length
    response_mask = attention_mask[:, -response_length:]
    valid_response_lengths = response_mask.to(torch.long).sum(dim=-1)
    if torch.any(valid_response_lengths <= 0):
        raise ValueError("ARQ requires every training row to contain at least one valid response token")

    v_positions = torch.full(
        (batch_size,), prompt_width - 1, dtype=torch.long, device=attention_mask.device
    )
    q_positions = prompt_width + valid_response_lengths - 1

    batch_indices = torch.arange(batch_size, device=attention_mask.device)
    if not torch.all(attention_mask[batch_indices, v_positions].bool()):
        raise ValueError("ARQ V boundary does not point at a valid prompt token; expected left-padded prompts")
    if not torch.all(attention_mask[batch_indices, q_positions].bool()):
        raise ValueError("ARQ Q boundary does not point at a valid response token")
    return v_positions, q_positions


def padded_positions_to_unpadded_indices(
    padded_positions: torch.Tensor,
    unpad_indices: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Map one padded position per row to indices in flash-attn unpadded data."""
    batch_indices = torch.arange(padded_positions.shape[0], device=padded_positions.device)
    flattened_positions = (batch_indices * sequence_length + padded_positions).to(unpad_indices.dtype)
    compact_positions = torch.searchsorted(unpad_indices, flattened_positions)
    if torch.any(compact_positions >= unpad_indices.numel()) or not torch.equal(
        unpad_indices[compact_positions], flattened_positions
    ):
        raise ValueError("ARQ boundary points at padding or is missing from the unpadded sequence")
    return compact_positions


def _kept_vocab_slices(vocab_size: int, excluded_token_ids: Sequence[int]) -> list[slice]:
    excluded = sorted(set(int(token_id) for token_id in excluded_token_ids))
    kept_slices = []
    start = 0
    for token_id in excluded:
        if start < token_id:
            kept_slices.append(slice(start, token_id))
        start = token_id + 1
    if start < vocab_size:
        kept_slices.append(slice(start, vocab_size))
    return kept_slices


def logsumexp_excluding(logits: torch.Tensor, excluded_token_ids: Sequence[int]) -> torch.Tensor:
    """Compute logsumexp over vocabulary except a small set, without cloning logits.

    Reducing contiguous kept slices means excluded LM-head rows are absent from
    the autograd graph entirely; the cancellation is exact rather than relying
    on subtracting their probability mass from a full-vocabulary reduction.
    """
    if not excluded_token_ids:
        return torch.logsumexp(logits, dim=-1)
    kept_slices = _kept_vocab_slices(logits.shape[-1], excluded_token_ids)
    if not kept_slices:
        raise ValueError("Cannot exclude the entire policy vocabulary")
    kept_lse_parts = [torch.logsumexp(logits[..., kept_slice], dim=-1) for kept_slice in kept_slices]
    return torch.logsumexp(torch.stack(kept_lse_parts, dim=-1), dim=-1)


def masked_policy_logprobs_and_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    excluded_token_ids: Iterable[int],
    calculate_entropy: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Selected-token log-probs and entropy with ARQ tokens removed.

    This is equivalent to setting the excluded vocabulary logits to -inf, but
    avoids allocating a second full logits tensor.  Entropy still requires a
    full-vocabulary softmax, as does the regular PPO entropy implementation.
    """
    excluded_token_ids = tuple(int(token_id) for token_id in excluded_token_ids)
    if any(token_id < 0 or token_id >= logits.shape[-1] for token_id in excluded_token_ids):
        raise ValueError("An excluded ARQ token ID is outside the model vocabulary")
    kept_lse = logsumexp_excluding(logits, excluded_token_ids)
    selected_logits = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_probs = selected_logits - kept_lse
    sampled_excluded = torch.zeros_like(labels, dtype=torch.bool)
    for token_id in excluded_token_ids:
        sampled_excluded = sampled_excluded | labels.eq(token_id)
    log_probs = log_probs.masked_fill(sampled_excluded, -torch.inf)

    if not calculate_entropy:
        return log_probs, None

    entropy = torch.zeros_like(kept_lse)
    for kept_slice in _kept_vocab_slices(logits.shape[-1], excluded_token_ids):
        kept_log_probs = logits[..., kept_slice] - kept_lse.unsqueeze(-1)
        entropy = entropy - (torch.exp(kept_log_probs) * kept_log_probs).sum(dim=-1)
    return log_probs, entropy


def value_outputs_from_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    value_temperature: float,
    reward_max: float,
) -> dict[str, torch.Tensor]:
    """Turn the two-token raw logit gap into bounded probability and value."""
    if value_temperature <= 0:
        raise ValueError(f"value_temperature must be positive, got {value_temperature}")
    if reward_max <= 0:
        raise ValueError(f"reward_max must be positive, got {reward_max}")
    logit_gaps = (positive_logits - negative_logits) / value_temperature
    scores = 2.0 * torch.sigmoid(logit_gaps) - 1
    return {"probs": scores, "values": scores * reward_max, "logit_gaps": logit_gaps}
