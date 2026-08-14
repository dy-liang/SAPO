#!/usr/bin/env bash
pid=1443651
echo "进程 $pid 运行中..."
while kill -0 "$pid" 2>/dev/null; do
    sleep 5
done
echo "进程 $pid 已结束"
pkill -f "keepgpu"

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}
train_data_size=${TRAIN_DATA_SIZE:-128}
val_data_size=${VAL_DATA_SIZE:-2}
# "td0": r_t+\gamma V_{\rm old}(s_{t+1}) = \delta_t
# "lambda_return": G_t^{\lambda_Q}
# "monte_carlo": G_t
# "sarsa_td0": r_t+\gamma Q_{old}(s_{t+1},a_{t+1})-V_{old}(s_t)
arq_q_target=${ARQ_Q_TARGET:-lambda_return}
arq_q_loss_mode=${ARQ_Q_LOSS_MODE:-advantage_residual} # train Q or Q-V
arq_advantage_mode=${ARQ_ADVANTAGE_MODE:-gae} #{"gae", "qv", "gae_qv_mix"}
arq_q_advantage_mix_coef=${ARQ_Q_ADVANTAGE_MIX_COEF:-0.2}
arq_q_lambda=${ARQ_Q_LAMBDA:-null}

# nohup bash examples/arqppo_trainer/run_alfworld.sh > ./log/test_arqppo_QV_gae_ap1_gamlam9595_wQ_init0_t1_fixPros-rsid+lam.log 2>&1 &
arq_positive_token=${ARQ_POSITIVE_TOKEN:-'<|vision_start|>'}
arq_negative_token=${ARQ_NEGATIVE_TOKEN:-'<|vision_end|>'}

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size"

# Keep the inner quotes in argv so Hydra treats delimiters such as
# <|vision_start|> as strings instead of trying to lex their pipe symbols.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=arq_ppo \
    algorithm.gamma=0.95 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=False \
    algorithm.arq.enable=True \
    "algorithm.arq.positive_token='${arq_positive_token}'" \
    "algorithm.arq.negative_token='${arq_negative_token}'" \
    algorithm.arq.reward_mode=bounded_outcome \
    algorithm.arq.reward_max=10.0 \
    algorithm.arq.value_temperature=1.0 \
    algorithm.arq.value_cliprange=0.2 \
    algorithm.arq.q_cliprange=0.2 \
    algorithm.arq.value_loss_coef=0.1 \
    algorithm.arq.q_loss_coef=0.1 \
    algorithm.arq.advantage_mode="$arq_advantage_mode" \
    algorithm.arq.q_advantage_mix_coef="$arq_q_advantage_mix_coef" \
    algorithm.arq.q_target="$arq_q_target" \
    algorithm.arq.q_lambda="$arq_q_lambda" \
    algorithm.arq.q_loss_mode="$arq_q_loss_mode" \
    algorithm.arq.normalize_turn_advantage=True \
    algorithm.arq.exclude_value_tokens_from_policy=True \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
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
    actor_rollout_ref.actor.invalid_action_penalty_coef=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.novelty_module.enable=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=3 \
    env.max_steps=50 \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus="$num_cpus_per_env_worker" \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=verl_agent \
    trainer.experiment_name=qwen2.5_1.5b_SAPO_Gae_QV_Ap1_gamlam9595_wQ_init0_t1_Lam+Resid \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=-1 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False \
    "$@"

nohup python /home/heliyuan/project_code/keepgpu.py --gpu 0 --util 25 --memory 1000 > keep_gpu.log 2>&1 &
echo "GPU ok!!"
