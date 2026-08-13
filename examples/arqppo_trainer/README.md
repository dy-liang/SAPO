# ARQ-PPO Q-target and advantage experiments

The default configuration preserves the original behavior:

```text
q_target=td0
q_loss_mode=absolute
advantage_mode=gae
```

These three axes can now be varied independently without affecting PPO/GRPO.

## Q return target

- `td0`: `r_t + gamma * V_old(s_{t+1})`.
- `lambda_return`: V-bootstrapped lambda-return. `q_lambda=null` reuses
  `algorithm.lam`; `q_lambda=0` equals TD(0), and `q_lambda=1` equals the
  finite-trajectory Monte Carlo return.
- `monte_carlo`: discounted return from the collected finite trajectory.
- `sarsa_td0`: `r_t + gamma * Q_old(s_{t+1}, a_{t+1})`. This is experimental
  because it bootstraps Q from itself.

## Q regression loss

- `absolute`: regress `Q(s_t,a_t)` to the selected Q return target.
- `advantage_residual`: regress
  `Q(s_t,a_t) - stopgrad(V(s_t))` to `Q_target - V_old(s_t)`. This focuses the
  post-action readout on information added by the sampled action. Q loss has no
  direct gradient through the current V output, although both still share the
  Transformer parameters.

Setting `algorithm.arq.q_loss_coef=0.0` skips the Q-loss branch completely. Q is
still evaluated for diagnostics and can drift as the shared model is updated.

## Policy advantage

- `gae`: standard turn-level GAE; predicted Q is not used by the actor.
- `qv`: use `Q_old(s_t,a_t) - V_old(s_t)`.
- `gae_qv_mix`: use
  `(1-beta) * GAE + beta * (Q_old-V_old)`, where
  `beta=algorithm.arq.q_advantage_mix_coef`.

Invalid-action shaping is applied after selecting/mixing the raw advantage, and
the result is normalized once over unique turns.

## Example

```bash
ARQ_Q_TARGET=lambda_return \
ARQ_Q_LAMBDA=0.95 \
ARQ_Q_LOSS_MODE=advantage_residual \
ARQ_ADVANTAGE_MODE=gae \
bash examples/arqppo_trainer/run_webshop.sh \
  algorithm.arq.q_loss_coef=0.01
```

After Q becomes reliable, test a conservative direct policy contribution:

```bash
ARQ_Q_TARGET=lambda_return \
ARQ_Q_LOSS_MODE=advantage_residual \
ARQ_ADVANTAGE_MODE=gae_qv_mix \
ARQ_Q_ADVANTAGE_MIX_COEF=0.1 \
bash examples/arqppo_trainer/run_webshop.sh \
  algorithm.arq.q_loss_coef=0.01
```

Before enabling `qv` or `gae_qv_mix`, check that
`arq/q_incremental_mse_gain` is positive and
`arq/q_advantage_target_correlation` is consistently positive. The latter is
computed against the currently selected Q target, whereas the existing
`arq/q_advantage_td_correlation` always remains a TD(0) diagnostic.
