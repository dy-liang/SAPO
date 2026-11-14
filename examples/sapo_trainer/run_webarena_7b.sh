#!/usr/bin/env bash
set -euo pipefail
set -x

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

DATA_DIR=${AGENTGYM_DATA_DIR:-$HOME/data/verl-agent/agentgym}
ENV_ADDR=${AGENTGYM_ENV_ADDR:-http://127.0.0.1:8000}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
N_GPUS=${N_GPUS:-2}
ARQ_POSITIVE_TOKEN=${ARQ_POSITIVE_TOKEN:-'<|vision_start|>'}
ARQ_NEGATIVE_TOKEN=${ARQ_NEGATIVE_TOKEN:-'<|vision_end|>'}

python3 -m examples.data_preprocess.prepare_agentgym \
    --environment webarena \
    --local_dir "$DATA_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=arq_ppo \
    algorithm.gamma=0.95 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=False \
    algorithm.arq.enable=True \
    "algorithm.arq.positive_token='${ARQ_POSITIVE_TOKEN}'" \
    "algorithm.arq.negative_token='${ARQ_NEGATIVE_TOKEN}'" \
    algorithm.arq.reward_mode=bounded_outcome \
    algorithm.arq.reward_max=10.0 \
    algorithm.arq.value_temperature=1.0 \
    algorithm.arq.q_target=sarsa_td0 \
    algorithm.arq.q_loss_mode=absolute \
    algorithm.arq.advantage_mode=gae \
    algorithm.arq.normalize_turn_advantage=True \
    algorithm.arq.exclude_value_tokens_from_policy=True \
    data.train_files="$DATA_DIR/webarena/train.parquet" \
    data.val_files="$DATA_DIR/webarena/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length=750 \
    data.max_response_length=14098 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.novelty_module.enable=False \
    env.env_name=webarena \
    env.agentgym.env_addr="$ENV_ADDR" \
    env.agentgym.timeout=600 \
    env.agentgym.reward_scale=10.0 \
    env.seed=3 \
    env.history_length=15 \
    env.agentgym.observation_max_prompt_length=32256 \
    env.max_steps=15 \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus=0.02 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=verl_agent_webarena \
    trainer.experiment_name=qwen2.5_7b_SAPO \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=25 \
    trainer.val_before_train=True \
    "${@:2}"
