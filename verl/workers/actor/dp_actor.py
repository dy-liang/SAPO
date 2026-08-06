# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#torch.set_printoptions(threshold=float("inf"))

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.arq import (
    get_arq_boundary_positions,
    masked_policy_logprobs_and_entropy,
    padded_positions_to_unpadded_indices,
    value_outputs_from_logits,
)
from verl.trainer.ppo.core_algos import agg_loss, compute_arq_value_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor
from transformers import PreTrainedTokenizer
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def create_action_mask(batch):
    """
    generate mask within tag <action></action>。
    Args:
        input_ids: (batch_size, seq_len)
    Returns:
        action_mask: (batch_size, seq_len) tokens in tag using 1, otherwise 0
    """
    response_ids = batch['responses']
    B, L = response_ids.shape
    device = response_ids.device

    is_start_marker = (response_ids[:, :-2] == 27) & (response_ids[:, 1:-1] == 1311) #& (response_ids[:, 2:] == 29)
    start_pos = torch.zeros((B, L), device=device, dtype=torch.bool)
    start_pos[:, 2:] = is_start_marker

    is_end_marker = (response_ids[:, :-2] == 522) & (response_ids[:, 1:-1] == 1311) #& (response_ids[:, 2:] == 397)
    end_pos = torch.zeros((B, L), device=device, dtype=torch.bool)
    end_pos[:, :-2] = is_end_marker
    #print("marker:",is_start_marker[0],is_end_marker[0])

    in_action_zone = torch.cumsum(start_pos.long(), dim=-1) - torch.cumsum(end_pos.long(), dim=-1)

    action_mask = (in_action_zone > 0).float()

    # ex '>' token：
    action_mask = action_mask * (~start_pos)
    #print("mask:",action_mask[0] * response_ids[0])

    return action_mask

class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.novelty_coef = self.config.get("novelty_coef", 0.001)
        print(f"Novelty_coef={self.novelty_coef}")
        self.novelty_max = self.config.get("novelty_max", 0.1)
        print(f"Novelty_max={self.novelty_max}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

        self.arq_config = self.config.get("arq", None)
        self.arq_enabled = self.arq_config is not None and self.arq_config.get("enable", False)
        self.arq_token_ids = ()
        if self.arq_enabled:
            self.arq_token_ids = (
                int(self.arq_config.positive_token_id),
                int(self.arq_config.negative_token_id),
            )
            if self.use_fused_kernels:
                raise ValueError("ARQ-PPO v1 does not support fused actor kernels")
            if self.use_ulysses_sp:
                raise ValueError("ARQ-PPO v1 does not support Ulysses sequence parallelism")

    def _extract_arq_outputs(self, raw_logits, attention_mask, response_length, unpad_indices=None):
        """Extract bounded V/Q scalars from raw next-token logits."""
        v_positions, q_positions = get_arq_boundary_positions(attention_mask, response_length)
        if unpad_indices is not None:
            sequence_length = attention_mask.shape[1]
            v_indices = padded_positions_to_unpadded_indices(v_positions, unpad_indices, sequence_length)
            q_indices = padded_positions_to_unpadded_indices(q_positions, unpad_indices, sequence_length)
            v_logits = raw_logits[v_indices]
            q_logits = raw_logits[q_indices]
        else:
            batch_indices = torch.arange(raw_logits.shape[0], device=raw_logits.device)
            v_logits = raw_logits[batch_indices, v_positions]
            q_logits = raw_logits[batch_indices, q_positions]

        positive_id, negative_id = self.arq_token_ids
        v_output = value_outputs_from_logits(
            v_logits[:, positive_id],
            v_logits[:, negative_id],
            value_temperature=float(self.arq_config.value_temperature),
            reward_max=float(self.arq_config.reward_max),
        )
        q_output = value_outputs_from_logits(
            q_logits[:, positive_id],
            q_logits[:, negative_id],
            value_temperature=float(self.arq_config.value_temperature),
            reward_max=float(self.arq_config.reward_max),
        )
        return {
            "v_probs": v_output["probs"],
            "q_probs": q_output["probs"],
            "v_values": v_output["values"],
            "q_values": q_output["values"],
            "v_logit_gaps": v_output["logit_gaps"],
            "q_logit_gaps": q_output["logit_gaps"],
        }

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, calculate_nov=False, calculate_arq=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            action_novelty = None
            arq_outputs = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad, # prompt+response
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    if self.arq_enabled and calculate_arq:
                        arq_outputs = self._extract_arq_outputs(
                            raw_logits=logits_rmpad,
                            attention_mask=attention_mask,
                            response_length=response_length,
                            unpad_indices=indices,
                        )
                    # V/Q were gathered from raw logits above.  Reuse the same
                    # storage for policy-temperature scaling to avoid another
                    # [tokens, vocab] allocation.
                    policy_logits_rmpad = logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    if self.arq_enabled:
                        log_probs, entropy_rmpad = masked_policy_logprobs_and_entropy(
                            logits=policy_logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            excluded_token_ids=self.arq_token_ids,
                            calculate_entropy=calculate_entropy,
                        )
                    else:
                        inplace_backward = not calculate_entropy
                        log_probs = logprobs_from_logits(
                            logits=policy_logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                        # compute entropy
                        if calculate_entropy:
                            entropy_rmpad = self.compute_entropy_from_logits(policy_logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length) align
                #response_logits = full_logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                #response_ids = input_ids[:, -response_length - 1: -1]  # (bsz, response_length)
                #response_logit = torch.gather(
                #     response_logits,           # (bsz, response_length, vocab)
                #     dim=2,
                #     index=response_ids.unsqueeze(-1)  # (bsz, response_length, 1)
                # ).squeeze(-1)                  # (bsz, response_length)
                #response_certainty = torch.logsumexp(response_logits, dim=-1) - response_logits.mean(dim=-1) #(bsz, response_length)
                #response_mask = attention_mask[:, -response_length:]
                action_novelty = None
                # if calculate_nov:
                #     action_mask = create_action_mask(batch=micro_batch) # (bsz, response_length)
                #     action_probs_sum = (action_mask * log_probs).sum(dim=-1) # (bsz, response_length)
                #     #action_probs_sum = (action_mask * response_logit).sum(dim=-1) # (bsz, response_length)
                #     #action_probs_sum = (action_mask * response_certainty).sum(dim=-1) # (bsz, response_length)
                #     #action_probs_sum = (response_mask * response_certainty).sum(dim=-1) # (bsz, response_length)

                #     #token_count = response_mask.sum(dim=-1) # (bsz, response_length)
                #     token_count = action_mask.sum(dim=-1) # (bsz, response_length)

                #     action_probs_mean = action_probs_sum / token_count.clamp(min=1e-6)
                #     action_probs_mean = action_probs_mean.masked_fill(token_count != -10000, 0.0) # 0.0 -> reward hacking
                #     # action_novelty = (action_probs_mean * self.novelty_coef).clamp(min=-self.novelty_max, max=self.novelty_max) # 0.001 0.05
                #     action_novelty = action_probs_mean.unsqueeze(-1).detach()
                #     #print(action_novelty)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    raw_logits = output.logits
                    if self.arq_enabled and calculate_arq:
                        arq_outputs = self._extract_arq_outputs(
                            raw_logits=raw_logits,
                            attention_mask=attention_mask,
                            response_length=response_length,
                        )
                    # Boundary values have already been copied out of the raw
                    # tensor, so policy scaling can safely reuse logits storage.
                    policy_logits = raw_logits.div_(temperature)
                    response_logits = policy_logits[:, -response_length - 1 : -1, :]
                    if self.arq_enabled:
                        log_probs, entropy = masked_policy_logprobs_and_entropy(
                            logits=response_logits,
                            labels=micro_batch["responses"],
                            excluded_token_ids=self.arq_token_ids,
                            calculate_entropy=calculate_entropy,
                        )
                    else:
                        log_probs = logprobs_from_logits(response_logits, micro_batch["responses"])
                        if calculate_entropy:
                            entropy = verl_F.entropy_from_logits(response_logits)  # (bsz, response_length)

            if calculate_arq:
                return entropy, log_probs, action_novelty, arq_outputs
            return entropy, log_probs, action_novelty

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, calculate_nov=False, calculate_arq=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        action_novelty_lst = []
        arq_output_lists = {
            "v_probs": [],
            "q_probs": [],
            "v_values": [],
            "q_values": [],
            "v_logit_gaps": [],
            "q_logit_gaps": [],
        }
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                forward_outputs = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    calculate_nov=calculate_nov,
                    calculate_arq=calculate_arq,
                )
                if calculate_arq:
                    entropy, log_probs, action_novelty, arq_outputs = forward_outputs
                else:
                    entropy, log_probs, action_novelty = forward_outputs
                    arq_outputs = None
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if calculate_nov:
                action_novelty_lst.append(action_novelty)
            if calculate_arq:
                if arq_outputs is None:
                    raise RuntimeError("ARQ values were requested but the actor did not return them")
                for key, value in arq_outputs.items():
                    arq_output_lists[key].append(value)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        action_novelties = None
        if calculate_nov:
            action_novelties = torch.concat(action_novelty_lst, dim=0)
        concatenated_arq_outputs = None
        if calculate_arq:
            concatenated_arq_outputs = {
                key: torch.concat(values, dim=0) for key, values in arq_output_lists.items()
            }
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        if calculate_arq:
            return log_probs, entropys, action_novelties, concatenated_arq_outputs
        return log_probs, entropys, action_novelties

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.arq_enabled:
            select_keys.extend(["old_v_probs", "old_q_probs", "mc_return"])
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    forward_outputs = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            calculate_nov=False,
                            calculate_arq=self.arq_enabled,
                        )
                    if self.arq_enabled:
                        entropy, log_prob, _, arq_outputs = forward_outputs
                    else:
                        entropy, log_prob, _ = forward_outputs
                        arq_outputs = None
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.arq_enabled:
                        if arq_outputs is None:
                            raise RuntimeError("ARQ actor update requires current V/Q outputs")
                        target_probs = torch.clamp(
                            data["mc_return"].to(torch.float32) / float(self.arq_config.reward_max),
                            min=0.0,
                            max=1.0,
                        )
                        value_loss, value_clipfrac = compute_arq_value_loss(
                            current_probs=arq_outputs["v_probs"],
                            old_probs=data["old_v_probs"],
                            target_probs=target_probs,
                            cliprange=float(self.arq_config.value_cliprange),
                        )
                        q_loss, q_clipfrac = compute_arq_value_loss(
                            current_probs=arq_outputs["q_probs"],
                            old_probs=data["old_q_probs"],
                            target_probs=target_probs,
                            cliprange=float(self.arq_config.q_cliprange),
                        )
                        policy_loss = (
                            policy_loss
                            + float(self.arq_config.value_loss_coef) * value_loss
                            + float(self.arq_config.q_loss_coef) * q_loss
                        )
                        append_to_dict(
                            metrics,
                            {
                                "arq/value_loss": value_loss.detach().item(),
                                "arq/q_loss": q_loss.detach().item(),
                                "arq/value_clipfrac": value_clipfrac.detach().item(),
                                "arq/q_clipfrac": q_clipfrac.detach().item(),
                                "arq/current_v_prob_mean": arq_outputs["v_probs"].detach().mean().item(),
                                "arq/current_q_prob_mean": arq_outputs["q_probs"].detach().mean().item(),
                                "arq/current_q_minus_v_mean": (
                                    arq_outputs["q_values"] - arq_outputs["v_values"]
                                ).detach().mean().item(),
                                "arq/v_logit_gap_mean": arq_outputs["v_logit_gaps"].detach().mean().item(),
                                "arq/q_logit_gap_mean": arq_outputs["q_logit_gaps"].detach().mean().item(),
                            },
                        )

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
