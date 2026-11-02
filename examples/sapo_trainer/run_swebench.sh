#!/usr/bin/env bash
set -xeuo pipefail

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

: "${TRAIN_FILE:?Set TRAIN_FILE to a SWE-bench-compatible parquet produced by prepare_swebench.py}"
: "${VAL_FILE:?Set VAL_FILE to the SWE-bench Verified parquet produced by prepare_swebench.py}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-100}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-28672}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-1}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=arq_ppo \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=False \
    algorithm.arq.enable=True \
    "algorithm.arq.positive_token='<|vision_start|>'" \
    "algorithm.arq.negative_token='<|vision_end|>'" \
    algorithm.arq.reward_mode=bounded_outcome \
    algorithm.arq.reward_max=10.0 \
    algorithm.arq.value_temperature=1.0 \
    algorithm.arq.value_cliprange=0.2 \
    algorithm.arq.q_cliprange=0.2 \
    algorithm.arq.value_loss_coef=0.1 \
    algorithm.arq.q_loss_coef=0.1 \
    algorithm.arq.advantage_mode=gae_qv_mix \
    algorithm.arq.q_advantage_mix_coef=0.2 \
    algorithm.arq.q_target=sarsa_td0 \
    algorithm.arq.q_loss_mode=absolute \
    algorithm.arq.normalize_turn_advantage=True \
    algorithm.arq.exclude_value_tokens_from_policy=True \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=1.0 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.novelty_module.enable=False \
    env.env_name=swebench \
    env.seed=0 \
    env.max_steps="$MAX_STEPS" \
    env.history_length=6 \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus="$NUM_CPUS_PER_ENV_WORKER" \
    env.resources_per_worker.num_gpus=0 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=sapo_swebench \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen_sapo_swebench}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-1}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.save_freq="${SAVE_FREQ:-20}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-10}" \
    trainer.val_before_train=True \
    "$@"
