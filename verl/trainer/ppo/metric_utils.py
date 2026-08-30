# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated

@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.
    
    This is an internal helper function that extracts masks and lengths for prompts and responses.
    
    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        
    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    grpo_advantages = batch.batch["grpo_advantages"]
    len_advantages = batch.batch["step_len_advantages"]
    token_len_advantages = batch.batch["token_len_advantages"]

    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()
    ppo_turn_unique_mask = batch.batch.get("ppo_turn_unique_mask")
    if ppo_turn_unique_mask is not None:
        ppo_turn_unique_mask = ppo_turn_unique_mask.bool()
        policy_metric_mask = batch.batch["ppo_turn_policy_mask"].bool()
        value_metric_mask = batch.batch["ppo_turn_value_mask"].bool()
        # Report the reward actually used by the new estimator, rather than
        # the unrelated reward-manager tensors retained for other methods.
        sequence_reward = batch.batch["env_reward"].to(torch.float32)
        sequence_score = sequence_score[ppo_turn_unique_mask]
        sequence_reward = sequence_reward[ppo_turn_unique_mask]
    else:
        policy_metric_mask = response_mask
        value_metric_mask = response_mask

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]
    metric_prompt_length = (
        prompt_length[ppo_turn_unique_mask]
        if ppo_turn_unique_mask is not None
        else prompt_length
    )
    metric_response_length = (
        response_length[ppo_turn_unique_mask]
        if ppo_turn_unique_mask is not None
        else response_length
    )

    valid_adv = torch.masked_select(advantages, policy_metric_mask)
    valid_grpo_adv = torch.masked_select(grpo_advantages, policy_metric_mask)
    valid_len_adv = torch.masked_select(len_advantages, policy_metric_mask)
    valid_token_len_adv = torch.masked_select(token_len_advantages, policy_metric_mask)

    valid_returns = torch.masked_select(returns, value_metric_mask)
    unique_traj_uid, unique_idx = np.unique(batch.non_tensor_batch['traj_uid'], return_index=True)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, value_metric_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # adv
        "critic/grpo_advantages/mean": torch.mean(valid_grpo_adv).detach().item(),
        "critic/grpo_advantages/max": torch.max(valid_grpo_adv).detach().item(),
        "critic/grpo_advantages/min": torch.min(valid_grpo_adv).detach().item(),
        # adv
        "critic/step_len_advantages/mean": torch.mean(valid_len_adv).detach().item(),
        "critic/step_len_advantages/max": torch.max(valid_len_adv).detach().item(),
        "critic/step_len_advantages/min": torch.min(valid_len_adv).detach().item(),
        # adv
        "critic/token_len_advantages/mean": torch.mean(valid_token_len_adv).detach().item(),
        "critic/token_len_advantages/max": torch.max(valid_token_len_adv).detach().item(),
        "critic/token_len_advantages/min": torch.min(valid_token_len_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(metric_response_length).detach().item(),
        "response_length/max": torch.max(metric_response_length).detach().item(),
        "response_length/min": torch.min(metric_response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(metric_response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(metric_prompt_length).detach().item(),
        "prompt_length/max": torch.max(metric_prompt_length).detach().item(),
        "prompt_length/min": torch.min(metric_prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(metric_prompt_length, max_prompt_length).float()).detach().item(),
        # episode
        "episode/reward/mean": 
            batch.non_tensor_batch["episode_rewards"][unique_idx].mean().item(),
        "episode/reward/max": 
            batch.non_tensor_batch["episode_rewards"][unique_idx].max().item(),
        "episode/reward/min": 
            batch.non_tensor_batch["episode_rewards"][unique_idx].min().item(),
        "episode/length/mean": 
            batch.non_tensor_batch["episode_lengths"][unique_idx].mean().item(),
        "episode/length/max":
            batch.non_tensor_batch["episode_lengths"][unique_idx].max().item(),
        "episode/length/min": 
            batch.non_tensor_batch["episode_lengths"][unique_idx].min().item(),
        "episode/tool_call_count/mean": 
            batch.non_tensor_batch["tool_callings"][unique_idx].mean().item(),
        # "episode/tool_call_count/max":
        #     batch.non_tensor_batch["tool_callings"][unique_idx].max().item(),
        # "episode/tool_call_count/min":
        #     batch.non_tensor_batch["tool_callings"][unique_idx].min().item(),
        **({f"episode/{k}": v[0].item() for k, v in batch.non_tensor_batch.items() if "success_rate" in k}),
    }

    arq_metric_keys = {
        "old_v_values",
        "old_q_values",
        "old_v_probs",
        "old_q_probs",
        "mc_return",
        "arq_v_targets",
        "arq_q_targets",
        "arq_v_target_probs",
        "arq_q_target_probs",
        "arq_gae_advantages",
        "arq_td_errors",
    }
    if arq_metric_keys.issubset(batch.batch.keys()):
        unique_turn_mask = batch.batch.get(
            "arq_unique_turn_mask",
            torch.ones_like(batch.batch["old_v_values"], dtype=torch.bool),
        ).bool()
        old_v = batch.batch["old_v_values"][unique_turn_mask].to(torch.float32)
        old_q = batch.batch["old_q_values"][unique_turn_mask].to(torch.float32)
        old_v_prob = batch.batch["old_v_probs"][unique_turn_mask].to(torch.float32)
        old_q_prob = batch.batch["old_q_probs"][unique_turn_mask].to(torch.float32)
        mc_return = batch.batch["mc_return"][unique_turn_mask].to(torch.float32)
        v_target = batch.batch["arq_v_targets"][unique_turn_mask].to(torch.float32)
        q_target = batch.batch["arq_q_targets"][unique_turn_mask].to(torch.float32)
        v_target_prob = batch.batch["arq_v_target_probs"][unique_turn_mask].to(torch.float32)
        q_target_prob = batch.batch["arq_q_target_probs"][unique_turn_mask].to(torch.float32)
        q_target_advantage = q_target - old_v
        gae_advantage = batch.batch["arq_gae_advantages"][unique_turn_mask].to(torch.float32)
        td_error = batch.batch["arq_td_errors"][unique_turn_mask].to(torch.float32)
        q_minus_v = old_q - old_v

        def correlation(lhs, rhs):
            centered_lhs = lhs - lhs.mean()
            centered_rhs = rhs - rhs.mean()
            return (centered_lhs * centered_rhs).mean() / (
                centered_lhs.std(unbiased=False) * centered_rhs.std(unbiased=False) + 1e-6
            )

        v_target_variance = torch.var(v_target, unbiased=False)
        q_target_variance = torch.var(q_target, unbiased=False)
        mc_return_variance = torch.var(mc_return, unbiased=False)
        v_residual_variance = torch.var(v_target - old_v, unbiased=False)
        q_residual_variance = torch.var(q_target - old_q, unbiased=False)
        v_mc_residual_variance = torch.var(mc_return - old_v, unbiased=False)
        q_mc_residual_variance = torch.var(mc_return - old_q, unbiased=False)
        q_mse = (old_q - q_target).square().mean()
        v_as_q_mse = (old_v - q_target).square().mean()
        q_advantage_std = q_minus_v.std(unbiased=False)
        td_error_std = td_error.std(unbiased=False)
        q_target_advantage_std = q_target_advantage.std(unbiased=False)
        metrics.update(
            {
                "arq/old_v_value_mean": old_v.mean().detach().item(),
                "arq/old_v_value_std": old_v.std(unbiased=False).detach().item(),
                "arq/old_q_value_mean": old_q.mean().detach().item(),
                "arq/old_q_value_std": old_q.std(unbiased=False).detach().item(),
                "arq/old_v_prob_min": old_v_prob.min().detach().item(),
                "arq/old_v_prob_max": old_v_prob.max().detach().item(),
                "arq/old_q_prob_min": old_q_prob.min().detach().item(),
                "arq/old_q_prob_max": old_q_prob.max().detach().item(),
                "arq/q_minus_v_mean": q_minus_v.mean().detach().item(),
                "arq/q_minus_v_std": q_minus_v.std(unbiased=False).detach().item(),
                "arq/q_v_near_collapse_ratio": (q_minus_v.abs() < 1e-3).float().mean().detach().item(),
                "arq/v_explained_variance": (1.0 - v_residual_variance / (v_target_variance + 1e-5)).detach().item(),
                "arq/q_explained_variance": (1.0 - q_residual_variance / (q_target_variance + 1e-5)).detach().item(),
                "arq/v_target_correlation": correlation(old_v, v_target).detach().item(),
                "arq/q_target_correlation": correlation(old_q, q_target).detach().item(),
                "arq/v_mc_correlation": correlation(old_v, mc_return).detach().item(),
                "arq/q_mc_correlation": correlation(old_q, mc_return).detach().item(),
                "arq/v_mc_explained_variance": (
                    1.0 - v_mc_residual_variance / (mc_return_variance + 1e-5)
                ).detach().item(),
                "arq/q_mc_explained_variance": (
                    1.0 - q_mc_residual_variance / (mc_return_variance + 1e-5)
                ).detach().item(),
                "arq/v_target_mse": (old_v_prob - v_target_prob).square().mean().detach().item(),
                "arq/q_target_mse": (old_q_prob - q_target_prob).square().mean().detach().item(),
                "arq/q_incremental_mse_gain": (v_as_q_mse - q_mse).detach().item(),
                "arq/q_advantage_td_correlation": correlation(q_minus_v, td_error).detach().item(),
                "arq/q_advantage_signal_ratio": (q_advantage_std / (td_error_std + 1e-6)).detach().item(),
                "arq/q_advantage_target_correlation": correlation(
                    q_minus_v, q_target_advantage
                ).detach().item(),
                "arq/q_advantage_target_mse": (
                    q_minus_v - q_target_advantage
                ).square().mean().detach().item(),
                "arq/q_advantage_target_signal_ratio": (
                    q_advantage_std / (q_target_advantage_std + 1e-6)
                ).detach().item(),
                "arq/gae_advantage_mean": gae_advantage.mean().detach().item(),
                "arq/gae_advantage_std": gae_advantage.std(unbiased=False).detach().item(),
                "arq/td_error_mean": td_error.mean().detach().item(),
                "arq/td_error_std": td_error_std.detach().item(),
                "arq/v_target_mean": v_target.mean().detach().item(),
                "arq/q_target_mean": q_target.mean().detach().item(),
                "arq/mc_return_mean": mc_return.mean().detach().item(),
            }
        )
        if "arq_truncated" in batch.batch:
            metrics["arq/horizon_truncation_ratio"] = (
                batch.batch["arq_truncated"][unique_turn_mask].float().mean().detach().item()
            )
        if "arq_v_target_clipped" in batch.batch:
            metrics["arq/v_target_clip_ratio"] = (
                batch.batch["arq_v_target_clipped"][unique_turn_mask].float().mean().detach().item()
            )
        if "arq_q_target_clipped" in batch.batch:
            metrics["arq/q_target_clip_ratio"] = (
                batch.batch["arq_q_target_clipped"][unique_turn_mask].float().mean().detach().item()
            )

    ppo_turn_metric_keys = {
        "ppo_turn_old_values",
        "ppo_turn_value_targets",
        "ppo_turn_gae_advantages",
        "ppo_turn_raw_advantages",
        "ppo_turn_advantages",
        "ppo_turn_td_errors",
        "ppo_turn_unique_mask",
        "mc_return",
        "env_reward",
    }
    if ppo_turn_metric_keys.issubset(batch.batch.keys()):
        unique_turn_mask = batch.batch["ppo_turn_unique_mask"].bool()
        old_values = batch.batch["ppo_turn_old_values"][unique_turn_mask].to(torch.float32)
        value_targets = batch.batch["ppo_turn_value_targets"][unique_turn_mask].to(torch.float32)
        gae_advantages = batch.batch["ppo_turn_gae_advantages"][unique_turn_mask].to(torch.float32)
        raw_advantages = batch.batch["ppo_turn_raw_advantages"][unique_turn_mask].to(torch.float32)
        normalized_advantages = batch.batch["ppo_turn_advantages"][unique_turn_mask].to(torch.float32)
        td_errors = batch.batch["ppo_turn_td_errors"][unique_turn_mask].to(torch.float32)
        mc_returns = batch.batch["mc_return"][unique_turn_mask].to(torch.float32)
        env_rewards = batch.batch["env_reward"][unique_turn_mask].to(torch.float32)

        def ppo_turn_correlation(lhs, rhs):
            centered_lhs = lhs - lhs.mean()
            centered_rhs = rhs - rhs.mean()
            return (centered_lhs * centered_rhs).mean() / (
                centered_lhs.std(unbiased=False) * centered_rhs.std(unbiased=False) + 1e-6
            )

        target_variance = value_targets.var(unbiased=False)
        mc_variance = mc_returns.var(unbiased=False)
        metrics.update(
            {
                "ppo_turn/old_v_mean": old_values.mean().detach().item(),
                "ppo_turn/old_v_std": old_values.std(unbiased=False).detach().item(),
                "ppo_turn/value_target_mean": value_targets.mean().detach().item(),
                "ppo_turn/value_target_std": value_targets.std(unbiased=False).detach().item(),
                "ppo_turn/gae_advantage_mean": gae_advantages.mean().detach().item(),
                "ppo_turn/gae_advantage_std": gae_advantages.std(unbiased=False).detach().item(),
                "ppo_turn/raw_policy_advantage_mean": raw_advantages.mean().detach().item(),
                "ppo_turn/raw_policy_advantage_std": raw_advantages.std(unbiased=False).detach().item(),
                "ppo_turn/normalized_advantage_mean": normalized_advantages.mean().detach().item(),
                "ppo_turn/normalized_advantage_std": normalized_advantages.std(unbiased=False).detach().item(),
                "ppo_turn/td_error_mean": td_errors.mean().detach().item(),
                "ppo_turn/td_error_std": td_errors.std(unbiased=False).detach().item(),
                "ppo_turn/env_reward_mean": env_rewards.mean().detach().item(),
                "ppo_turn/mc_return_mean": mc_returns.mean().detach().item(),
                "ppo_turn/v_target_correlation": ppo_turn_correlation(old_values, value_targets).detach().item(),
                "ppo_turn/v_mc_correlation": ppo_turn_correlation(old_values, mc_returns).detach().item(),
                "ppo_turn/v_target_mse": (old_values - value_targets).square().mean().detach().item(),
                "ppo_turn/v_target_explained_variance": (
                    1.0
                    - (value_targets - old_values).var(unbiased=False)
                    / (target_variance + 1e-5)
                ).detach().item(),
                "ppo_turn/v_mc_explained_variance": (
                    1.0
                    - (mc_returns - old_values).var(unbiased=False)
                    / (mc_variance + 1e-5)
                ).detach().item(),
                "ppo_turn/duplicate_ratio": (
                    1.0 - unique_turn_mask.float().mean()
                ).detach().item(),
            }
        )
        if "ppo_turn_truncated" in batch.batch:
            metrics["ppo_turn/horizon_truncation_ratio"] = (
                batch.batch["ppo_turn_truncated"][unique_turn_mask]
                .float()
                .mean()
                .detach()
                .item()
            )
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.
    
    This function calculates both raw timing metrics (in seconds) and per-token timing metrics 
    (in milliseconds) for various processing stages like generation, reference computation, 
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.
    
    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).
    
    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.
        
    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU
            
    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.
    
    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.
    
    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }
        
        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)
        
    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
