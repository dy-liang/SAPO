# Environment Setup

## Table of Contents
- [1. ALFWorld](#1-alfworld)  
- [2. WebShop](#2-webshop)  
- [3. Sokoban](#3-sokoban)  
- [4. Gym Cards](#4-gym-cards)  
- [5. AppWorld (Experimental)](#5-appworld-experimental)  
- [6. AgentGym SciWorld and WebArena](#6-agentgym-sciworld-and-webarena)

## 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
pip install vllm==0.8.5
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:
```bash
alfworld-play-tw
```
---

## 2. WebShop
WebShop requires Python <=3.10, so begin by creating a new `verl-agent-webshop` environment
```bash
conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop
```

Install WebShop
```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

Note: If you encounter issues with gdown, you may need visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `.cache/gdown/cookies.txt`.
Or you may need to manually download the files.


Verify that WebShop was installed correctly by running:
```bash
python run_web_agent_text_env.py
```

After WebShop is installed, return to the root directory of the repository and install the verl package in `verl-agent`:
```bash
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
```
The warnings can be safely ignored.

---
## 3. Sokoban
```bash
pip install matplotlib
pip install gym==0.26.2
pip install gym_sokoban==0.0.6
```
---
## 4. Gym Cards

```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```
---
### 5. AppWorld (Experimental)
Install AppWorld package
```bash
cd repo_root/
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
pip install -e .
pip install vllm==0.8.5
```
You can ignore the warning of incompatiblity for appworld, because we don't run appworld in `verl-agent` environment.

Create a dedicated conda environment `appworld` for the AppWorld server:
```bash
conda create -n appworld python=3.12 -y
conda activate appworld
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
appworld download data
```

---
## 6. AgentGym SciWorld and WebArena

SAPO uses the official AgentGym HTTP services but does not install
AgentGym-RL's older `verl` fork. Use three isolated environments: the normal
SAPO environment for training, one server environment for SciWorld, and one
server environment for WebArena. The environment servers may run on the same
machine as training or on separate CPU/server machines.

### 6.1 Clone AgentGym

Clone all AgentGym submodules outside the SAPO source tree:

```bash
git clone --recursive https://github.com/WooooDyy/AgentGym.git
cd AgentGym
git submodule update --init --recursive
```

The instructions below assume that this checkout is available at
`/path/to/AgentGym` and SAPO is available at `/path/to/SAPO`.

### 6.2 Install and launch SciWorld

SciWorld requires Java 8 or newer. Check Java before creating its Python 3.8
environment:

```bash
java -version
conda create -n agentenv-sciworld python=3.8 -y
conda activate agentenv-sciworld
cd /path/to/AgentGym/agentenv-sciworld
pip install -e .
```

Start the service in a dedicated terminal. Keep this process running throughout
data collection, training, and evaluation:

```bash
conda activate agentenv-sciworld
cd /path/to/AgentGym/agentenv-sciworld
sciworld --host 0.0.0.0 --port 36001
```

From the training machine, verify that the HTTP endpoint is reachable:

```bash
curl --fail http://127.0.0.1:36001/
```

Replace `127.0.0.1` with the server hostname when SciWorld is remote. The
default SciWorld training scripts use port `36001`.

### 6.3 Install and launch WebArena

WebArena is not a standalone text package: its Reddit, GitLab, Shopping,
Shopping Admin, Map, Wikipedia, and Homepage services must already be deployed
and reachable from the WebArena server. Follow the upstream
[WebArena website deployment guide](https://github.com/web-arena-x/webarena/tree/main/environment_docker)
before installing the AgentGym wrapper.

AgentGym's WebArena package requires Python 3.10.13:

```bash
conda create -n agentenv-webarena python=3.10.13 -y
conda activate agentenv-webarena
cd /path/to/AgentGym/agentenv-webarena
```

Before running `setup.sh`, replace the example service URLs in both
`setup.sh` and `agentenv_webarena/__init__.py` with the URLs of your deployment.
Set `OPENAI_API_KEY` and, when needed, `OPENAI_BASE_URL` for evaluator tasks that
use an LLM judge. Do not commit real credentials to either file.

Then install WebArena, Playwright/browser dependencies, generate task configs,
and create the authentication state:

```bash
source ./setup.sh
```

`setup.sh` runs `playwright install-deps`, which may require system package
installation privileges. It also executes `generate_test_data.py` and
`browser_env/auto_login.py`; all configured websites must be online at this
point. If credentials or URLs later change, rerun the generation and login
steps before restarting the service.

Launch WebArena from the `agentenv-webarena` directory in a dedicated terminal:

```bash
conda activate agentenv-webarena
cd /path/to/AgentGym/agentenv-webarena
webarena --host 0.0.0.0 --port 8000
```

Check connectivity from the SAPO training machine:

```bash
curl --fail http://127.0.0.1:8000/
```

The endpoint should return `ok`. Replace `127.0.0.1` with the remote hostname
when necessary and allow ports `36001`/`8000` through the relevant firewall.

### 6.4 Obtain the official task splits

Activate the normal SAPO training environment and run the included converter
from the SAPO repository root:

```bash
conda activate sapo
cd /path/to/SAPO

export AGENTGYM_DATA_DIR=/path/to/agentgym-data

python3 -m examples.data_preprocess.prepare_agentgym \
  --environment sciworld \
  --local_dir "$AGENTGYM_DATA_DIR"

python3 -m examples.data_preprocess.prepare_agentgym \
  --environment webarena \
  --local_dir "$AGENTGYM_DATA_DIR"
```

The converter downloads the official task IDs from the
[AgentGym-RL-Data-ID dataset](https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID)
and creates:

```text
/path/to/agentgym-data/
├── sciworld/
│   ├── train.parquet
│   └── test.parquet
└── webarena/
    ├── train.parquet
    └── test.parquet
```

Each row stores the official ID in `env_kwargs.task_id`. GRPO copies the same
task ID across its rollout group while allocating an independent server-side
environment for each rollout. For a small installation smoke test, pass
`--train_data_size 8 --val_data_size 2`; omit these options for real training.
Local copies of the official JSON files can be supplied with
`--train_id_file` and `--val_id_file`.

The six launch scripts call this converter automatically, so the manual step is
mainly useful for checking data access before allocating GPUs.

### 6.5 Configure a training run

Training runs in the normal SAPO environment. At minimum, set the model, data
directory, and address of the corresponding environment server:

```bash
conda activate sapo
cd /path/to/SAPO

export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export AGENTGYM_DATA_DIR=/path/to/agentgym-data
export N_GPUS=2
```

For SciWorld:

```bash
export AGENTGYM_ENV_ADDR=http://127.0.0.1:36001
```

For WebArena:

```bash
export AGENTGYM_ENV_ADDR=http://127.0.0.1:8000
```

`MODEL_PATH` may be either a local checkpoint directory or a Hugging Face model
ID. When the server is remote, use an address reachable from every training
node; `0.0.0.0` is only a server bind address and must not be used as the client
address.

### 6.6 Launch SAPO, GRPO, or PPO

All commands below use the vLLM rollout backend. Start exactly one command after
exporting the matching SciWorld or WebArena server address.

SAPO:

```bash
bash examples/sapo_trainer/run_sciworld_7b.sh vllm
bash examples/sapo_trainer/run_webarena_7b.sh vllm
```

GRPO:

```bash
bash examples/grpo_trainer/run_sciworld_7b.sh vllm
bash examples/grpo_trainer/run_webarena_7b.sh vllm
```

PPO with a separate critic:

```bash
bash examples/ppo_trainer/run_sciworld_7b.sh vllm
bash examples/ppo_trainer/run_webarena_7b.sh vllm
```

Environment variables can reduce concurrency without editing a script. This is
particularly important for a local WebArena deployment:

```bash
TRAIN_BATCH_SIZE=4 VAL_BATCH_SIZE=1 \
  bash examples/sapo_trainer/run_webarena_7b.sh vllm

TRAIN_BATCH_SIZE=2 VAL_BATCH_SIZE=1 GROUP_SIZE=2 \
  bash examples/grpo_trainer/run_webarena_7b.sh vllm
```

Additional Hydra overrides may be appended after `vllm`, for example:

```bash
bash examples/sapo_trainer/run_sciworld_7b.sh vllm \
  trainer.total_epochs=1 \
  "trainer.logger=['console']"
```

### 6.7 Evaluation and metrics

The training scripts use the official `test.parquet` split for validation and
enable `trainer.val_before_train=True`. Evaluation then runs every
`trainer.test_freq=5` training steps by default; no separate evaluator process
is required. The SciWorld and WebArena servers must remain available during
validation.

Reported `success_rate` uses the terminal benchmark result. The additional
`sciworld_task_score (not success_rate)` and
`webarena_task_score (not success_rate)` metrics remain in the original
benchmark scale even though optimization rewards are scaled.

### 6.8 Action, reward, and context conventions

Both environments use the same model-facing action protocol as the existing
SAPO environments: `<think>...</think><action>...</action>`. ReAct-style
colon labels and bare triple-backtick actions are marked invalid. For WebArena,
the adapter converts only the extracted `<action>` content to the fenced format
required by the upstream server.

Reward handling intentionally follows AgentGym-RL:

- SciWorld exposes a cumulative score. The adapter first converts it to score
  deltas so it is never double-counted, then the supplied 7B scripts apply a
  `0.1` training scale (raw 0–100 becomes 0–10).
- WebArena emits zero on non-terminal steps and the evaluator reward only on
  termination. The supplied scripts scale its raw 0/1 evaluator result to 0/10.

Scaling affects optimization rewards only. The logged `sciworld_task_score` and
`webarena_task_score` remain in their original benchmark units, so evaluation
results stay directly comparable with AgentGym-RL. Override
`env.agentgym.reward_scale` if raw training rewards are desired.

The scripts retain AgentGym-RL's dataset limits (`1024/4096` for SciWorld and
`750/14098` for WebArena), its per-turn generation limits (`200/512`), and its
`32768` model context. Because SAPO rebuilds the prompt for each environment
turn, `env.agentgym.observation_max_prompt_length` is set to model context minus
the per-turn generation limit (`32568/32256`). This preserves the official full
history behavior without letting prompt plus generation exceed the model limit.

AgentGym-RL keeps the complete chat history. The supplied scripts therefore use
`history_length=20` for SciWorld and `history_length=15` for WebArena, matching
their official rollout round counts. SciWorld follows the upstream reset
behavior exactly: `task_description` is prepended to the first observation once
and remains available through that full history; it is not re-injected on every
turn. WebArena likewise retains every model response and server observation,
including the objective and server-generated action history.

The WebArena server is memory intensive (AgentGym recommends roughly 1 GB per
concurrent browser environment). Reduce `TRAIN_BATCH_SIZE`, `GROUP_SIZE`, or
`VAL_BATCH_SIZE` through environment variables when the server host is smaller.

### 6.9 Troubleshooting

- `Connection refused`: confirm the server is running, use the server's
  reachable hostname in `AGENTGYM_ENV_ADDR`, and check the firewall.
- WebArena reset returns `TimeoutError`: verify every configured website URL,
  regenerate task files, refresh `.auth` with `browser_env/auto_login.py`, and
  restart the WebArena service.
- WebArena host runs out of memory: lower `TRAIN_BATCH_SIZE`, `VAL_BATCH_SIZE`,
  and GRPO's `GROUP_SIZE`. AgentGym reports about 1 GB per active browser
  environment, with additional fixed server overhead.
- Data download fails: check access to Hugging Face or download the two official
  JSON split files and pass them through `--train_id_file`/`--val_id_file`.
- Do not install AgentGym-RL's bundled `verl` into the SAPO environment. Only
  the standalone AgentGym servers are required by this integration.
