#!/usr/bin/env bash
set -x

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

model_path=${MODEL_PATH:-$HOME/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct}
train_data_size=${TRAIN_DATA_SIZE:-128}
val_data_size=${VAL_DATA_SIZE:-128}
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.02}
invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF:-1.0}

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=shared_head_ppo \
    algorithm.gamma=0.95 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=False \
    algorithm.shared_head_ppo.enable=True \
    algorithm.shared_head_ppo.scale_value_target=False \
    algorithm.shared_head_ppo.value_cliprange=0.5 \
    algorithm.shared_head_ppo.value_loss_coef=0.1 \
    algorithm.shared_head_ppo.normalize_turn_advantage=True \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_data_size" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef="$invalid_action_penalty_coef" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.novelty_module.enable=False \
    env.env_name=Webshop \
    env.seed=3 \
    env.max_steps=15 \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus="$num_cpus_per_env_worker" \
    trainer.critic_warmup=0 \
    trainer.logger="['console','swanlab']" \
    trainer.project_name=verl_agent_webshop \
    trainer.experiment_name=qwen2.5_7b_shared_head_ppo_noscale \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    "${@:2}"
