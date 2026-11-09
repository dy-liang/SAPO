# SPO in SAPO

This integration implements the official Single-stream Policy Optimization
Beta tracker and PPO advantage, adapted to SAPO's multi-turn trajectory format.
It is disabled unless both `algorithm.adv_estimator=spo` and
`algorithm.spo.enable=true` are set.

## Offline values

Canonical SPO uses eight initial-policy rollouts per training task. Pass a JSON
file whose keys exactly match the stable environment task IDs and whose values
are rollout outcome lists:

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
SPO_REQUIRE_OFFLINE_VALUES=True \
SPO_TASK_SAMPLING=True \
bash examples/spo_trainer/run_webshop_7b.sh
```

Without `SPO_OFFLINE_VALUES`, the tracker starts from Beta(0.5, 0.5); this is
the `SPO-no-init` ablation, not the paper's main configuration.

WebShop supports the paper's uncertainty-prioritized task sampling. The bundled
ALFWorld vector environment does not expose a safe reset-to-gamefile API, so its
script deliberately runs `SPO-uniform` and says so in the experiment name.

The scripts without a model-size suffix use Qwen2.5-1.5B-Instruct; the
`run_*_7b.sh` scripts use Qwen2.5-7B-Instruct.

For a controlled comparison with SAPO, all launch scripts enable the same
invalid-action penalty with coefficient 1.0. In SPO this penalty is applied only
to the per-turn policy advantage after trajectory-level normalization; the
binary success outcome and Beta tracker remain unchanged.
