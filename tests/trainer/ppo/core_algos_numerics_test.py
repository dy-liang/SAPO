# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty


def test_vanilla_ppo_extreme_log_ratio_has_finite_backward():
    # exp(100) overflows in FP32. PPO clipping makes the forward loss finite,
    # but without a pre-exp clamp its backward pass still produces NaN.
    log_prob = torch.tensor([100.0, 100.0], requires_grad=True)
    old_log_prob = torch.zeros_like(log_prob)
    advantages = torch.tensor([1.0, -1.0])
    response_mask = torch.ones_like(log_prob)

    pg_loss, _, _, _ = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )
    pg_loss.backward()

    assert torch.isfinite(pg_loss)
    assert torch.isfinite(log_prob.grad).all()


def test_low_var_kl_extreme_log_ratio_has_finite_backward():
    # ref - current = 100 would overflow inside exp before the final kld clamp.
    log_prob = torch.tensor([-100.0], requires_grad=True)
    ref_log_prob = torch.zeros_like(log_prob)

    loss = kl_penalty(log_prob, ref_log_prob, "low_var_kl").sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(log_prob.grad).all()
