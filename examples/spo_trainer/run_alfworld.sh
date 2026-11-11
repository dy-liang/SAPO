#!/usr/bin/env bash
set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.02}
train_data_size=${TRAIN_DATA_SIZE:-128}
val_data_size=${VAL_DATA_SIZE:-128}
spo_offline_values=${SPO_OFFLINE_VALUES:-null}
spo_require_offline_values=${SPO_REQUIRE_OFFLINE_VALUES:-False}
spo_rho_type=${SPO_RHO_TYPE:-kl}
if [[ "$spo_offline_values" == "null" ]]; then
    default_experiment_name=qwen2.5_1.5b_SPO_uniform_no_init
else
    default_experiment_name=qwen2.5_1.5b_SPO_uniform
fi
experiment_name=${EXPERIMENT_NAME:-$default_experiment_name}

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size"

# ALFWorld's bundled vector environment samples gamefiles internally, so this
# configuration is SPO-uniform. It still uses canonical SPO Beta advantages and
# KL-adaptive forgetting; enable=False prevents a misleading priority claim.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=spo \
    algorithm.use_kl_in_reward=False \
    algorithm.spo.enable=True \
    algorithm.spo.offline_values="$spo_offline_values" \
    algorithm.spo.offline_n=8 \
    algorithm.spo.require_offline_values="$spo_require_offline_values" \
    algorithm.spo.rho.type="$spo_rho_type" \
    algorithm.spo.rho.value=0.875 \
    algorithm.spo.rho.clip_lower=0.875 \
    algorithm.spo.rho.clip_upper=0.96 \
    algorithm.spo.rho.d_half=0.06 \
    algorithm.spo.task_sampling.enable=False \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_data_size" \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$HOME/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.novelty_module.enable=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus="$num_cpus_per_env_worker" \
    trainer.critic_warmup=0 \
    trainer.logger="['console','swanlab']" \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    "$@"
