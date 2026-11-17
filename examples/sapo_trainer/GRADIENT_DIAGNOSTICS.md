# SAPO policy/value/Q gradient diagnostics

The diagnostics are disabled by default and do not change the optimizer
gradient. On a scheduled step, the first micro-batch is recomputed in up to
three independent FSDP-safe forward/backward passes for the actor, V, and Q
objectives. Diagnostic gradients are cleared, the RNG state is restored, and
the normal joint backward pass then runs unchanged. Local shard dot products
and squared norms are summed across ranks.

Enable the default ALFWorld schedule with:

```bash
GRADIENT_DIAGNOSTICS_ENABLE=true \
GRADIENT_DIAGNOSTICS_INTERVAL=25 \
bash examples/sapo_trainer/run_alfworld.sh
```

For direct Hydra launches, set:

```text
actor_rollout_ref.actor.gradient_diagnostics.enable=True
actor_rollout_ref.actor.gradient_diagnostics.start_step=1
actor_rollout_ref.actor.gradient_diagnostics.interval=25
actor_rollout_ref.actor.gradient_diagnostics.max_micro_batches_per_update=1
actor_rollout_ref.actor.gradient_diagnostics.offload_gradient_snapshots=True
```

`max_micro_batches_per_update=1` is currently required. Gradient snapshots are
offloaded to CPU by default to avoid accelerator-memory spikes. Setting
`offload_gradient_snapshots=False` is faster but retains up to two
additional FP32 local gradient shards on the accelerator while V and Q are
measured.

The measurements describe the first micro-batch of each scheduled actor update,
not the fully accumulated optimizer gradient. Keep the micro-batch construction
and schedule identical across methods, and aggregate many scheduled points and
seeds rather than interpreting a single observation.

## Logged metrics

Metrics are emitted under `gradient_diagnostics/{group}/...`, where `group` is
`all` and, when recognizable from FSDP parameter names, `embedding`, `lower`,
`middle`, `upper`, `lm_head`, or `other`.

All gradient quantities are measured before global gradient clipping. The
`*_gradient_present` flags distinguish a true zero/unused gradient from a
well-defined zero cosine.

- `policy_value_cosine`, `policy_q_cosine`, `value_q_cosine`: raw pairwise
  gradient cosine.
- `policy_value_conflict`, `policy_q_conflict`, `value_q_conflict`: 1 when the
  corresponding dot product is negative, otherwise 0. Average this metric
  across diagnostic steps and seeds to obtain the conflict rate.
- `weighted_value_to_policy_norm`, `weighted_q_to_policy_norm`: effective
  auxiliary/policy gradient-norm ratios after applying the configured SAPO loss
  coefficients.
- `weighted_value_policy_projection`, `weighted_q_policy_projection`: signed
  auxiliary projection onto the policy gradient, normalized by squared policy
  gradient norm.
- `total_policy_alignment`: first-order alignment of the joint gradient with
  the policy gradient,

  ```text
  1 + c_V <g_policy,g_V>/||g_policy||^2
    + c_Q <g_policy,g_Q>/||g_policy||^2.
  ```

  A positive value means the joint gradient remains a local descent direction
  for the policy objective. `total_is_policy_descent` is its binary form.
- `joint_aux_to_policy_norm`: exact norm ratio
  `||c_V g_V + c_Q g_Q|| / ||g_policy||`, including V/Q cancellation or
  reinforcement.
- `joint_grad_to_policy_norm`: norm of the actual joint direction relative to
  the policy-only gradient.
- `aux_to_policy_norm_upper_bound`: sum of both effective norm ratios; this is a
  conservative upper bound because it does not assume V/Q alignment.

The policy objective includes PPO, reference KL when enabled, and entropy
regularization. Advantages and value targets retain their training-time
stop-gradient semantics.

## Suggested paper aggregation

Use the same checkpoints for every seed (for example 1, 25, 50, 75, 100, 125,
and 150). Report median cosine with an interquartile band, mean conflict rate,
effective norm ratios, and the fraction of checkpoints for which
`total_policy_alignment > 0`. A near-zero mean cosine alone is not evidence of
no conflict; report the signed distribution and layer groups together.

