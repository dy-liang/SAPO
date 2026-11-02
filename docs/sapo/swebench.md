# SWE-bench support for SAPO

This integration preserves the existing turn-level SAPO/ARQ-PPO trainer.  Each
dataset row identifies one repository image, one issue, and host-side evaluator
metadata.  A Ray environment worker starts the corresponding Docker image, runs
model-generated shell commands in `/testbed`, and executes the official
SWE-bench tests after `submit` or the rollout step limit.

## Install optional dependencies

```bash
pip install -r requirements-swebench.txt
docker version
```

The training process must be able to access a Docker daemon.  Do not expose an
unauthenticated remote Docker socket.

## Prepare validation data

SWE-bench Verified should be used for validation rather than training:

```bash
python -m examples.data_preprocess.prepare_swebench \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --output ~/data/sapo-swe/swe_bench_verified.parquet
```

Prepare a SWE-bench-compatible training split with the same command.  The
source dataset must expose `instance_id`, `repo`, `version`, `base_commit`,
`problem_statement`, `test_patch`, `FAIL_TO_PASS`, and `PASS_TO_PASS`.
`image`, `docker_image`, or `image_name` may override the canonical SWE-bench
image name.  The installed `swebench` package must know the repository/version
test specification.  SWE-Gym, SWE-reBench, and R2E-Gym can be used only when
their rows and images satisfy this contract; otherwise add a suite-specific
image/evaluator adapter first.

## Validate the environment before training

The empty repository state should be unresolved:

```bash
python -m examples.sapo_trainer.swebench_smoke \
  --parquet ~/data/sapo-swe/swe_bench_verified.parquet \
  --index 0
```

The gold patch should resolve the same instance:

```bash
python -m examples.sapo_trainer.swebench_smoke \
  --parquet ~/data/sapo-swe/swe_bench_verified.parquet \
  --index 0 \
  --gold
```

Only proceed when both checks pass.  A failure here means the Docker image,
dataset metadata, or evaluator version is inconsistent; training would turn
that infrastructure failure into a misleading zero reward.

## Train

```bash
export TRAIN_FILE=~/data/sapo-swe/swe_bench_train.parquet
export VAL_FILE=~/data/sapo-swe/swe_bench_verified.parquet
export MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct
bash examples/sapo_trainer/run_swebench.sh
```

Start with `TRAIN_BATCH_SIZE=1` and a handful of rows.  Every concurrent sample
uses a separate container and may run repository tests for several minutes.

## Security and evaluation integrity

- Network access is disabled inside agent containers by default.
- Git history is rebuilt from the visible base snapshot, preventing the policy
  from recovering a later gold commit through refs, reflogs, or unreachable
  objects.  The evaluator is updated to use the sanitized base commit.
- Gold patches and F2P/P2P metadata remain in host-side `env_kwargs`; they are
  not included in the model prompt or written into the container before grading.
- Do not mount the host Docker socket or training source tree inside an agent
  container.
- Use a trusted image registry and isolate the Docker daemon from unrelated
  workloads.  Model-generated shell commands are untrusted code.
- The current implementation is synchronous at each environment step.  This is
  suitable for correctness experiments; large-scale training will need an
  asynchronous rollout scheduler to avoid stragglers.
