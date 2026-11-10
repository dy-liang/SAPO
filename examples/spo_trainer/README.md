# SPO in SAPO

This integration implements the official Single-stream Policy Optimization
Beta tracker and PPO advantage, adapted to SAPO's multi-turn trajectory format.
It is disabled unless both `algorithm.adv_estimator=spo` and
`algorithm.spo.enable=true` are set.

## Offline values

Canonical SPO uses eight initial-policy rollouts per training task. The launch
scripts now default to `algorithm.spo.initialization.mode=collect`: before the
first optimizer step, the frozen initial policy is run eight times on every
training task and the binary outcomes initialize the Beta tracker. The results
are cached as `spo_initial_values.json` under the experiment checkpoint folder
and reused on restart.

To reuse a separately generated JSON instead, set `SPO_OFFLINE_VALUES`. Its keys
must exactly match the stable environment task IDs and its values must be rollout
outcome lists:

```json
{
  "500": [0, 1, 0, 0, 1, 0, 0, 1],
  "/path/to/alfworld/game.tw-pddl": [1, 0, 0, 1, 1, 0, 0, 1]
}
```

WebShop keys are session indices; ALFWorld keys are `extra.gamefile` strings.
For a canonical run, set:

```bash
SPO_OFFLINE_VALUES=/path/to/offline_values.json \
SPO_TASK_SAMPLING=True \
bash examples/spo_trainer/run_webshop_7b.sh
```

The `SPO-no-init` ablation must be requested explicitly:

```bash
SPO_INIT_MODE=none bash examples/spo_trainer/run_alfworld.sh
```

In canonical `collect` or `offline` mode, missing task values emit a warning and
fall back to Beta(0.5, 0.5). Set `SPO_STRICT_TASK_COVERAGE=True` to make a
coverage mismatch fatal.

WebShop supports the paper's uncertainty-prioritized task sampling. The bundled
ALFWorld vector environment now uses explicit gamefile assignment only during
the init-8 warm-up. Its subsequent online training still samples uniformly, so
the script deliberately reports `SPO-uniform` in the experiment name.

The scripts without a model-size suffix use Qwen2.5-1.5B-Instruct; the
`run_*_7b.sh` scripts use Qwen2.5-7B-Instruct.

For a controlled comparison with SAPO, all launch scripts enable the same
invalid-action penalty with coefficient 1.0. In SPO this penalty is applied only
to the per-turn policy advantage after trajectory-level normalization; the
binary success outcome and Beta tracker remain unchanged.
