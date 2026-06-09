---
layout: post
title: "DPO and Safety Fine Tuning"
date: 2026-05-29
tags: transformers attention deep-learning
categories: machine-learning
math: true
mermaid: true
---

# Introduction

As seen in the previous post, RLHF has two main components.

First, we train a reward model

$$
r_\phi(u,y)
$$

from human preference data. Then we optimize the language model policy using this learned reward. This improves over SFT because the model no longer learns only from demonstrations. It also learns from comparisons between answers. However, RLHF also introduces new difficulties. The reward model is trained on a finite preference dataset. Therefore, it may generalize poorly outside the region covered by the data. If the policy is optimized too aggressively against this imperfect reward model, it may find responses that receive high reward but are not actually preferred by humans. This is the problem of **reward hacking**.

Moreover, the policy optimization step is technically complex. It usually relies on reinforcement learning algorithms such as PPO. These methods require sampling from the current policy, estimating values or advantages, controlling the KL divergence to a reference model, and carefully tuning the optimization procedure.

This motivates a natural question:

$$
\text{Can we use preference data without explicitly training a reward model and running RL?}
$$

This is the idea behind Direct Preference Optimization, or DPO. At a high level, DPO replaces the RLHF pipeline

$$
\text{train a reward model}
\quad + \quad
\text{run RL policy optimization}
$$

with a direct objective:

$$
\text{train the policy directly on preference pairs}.
$$

So DPO keeps the preference data, but removes the explicit reward-modeling and reinforcement-learning steps.

# The Bridge Between RLHF and DPO

The starting point of DPO is the KL-regularized RLHF objective. For a fixed prompt $u$, recall the optimization problem

$$
\max_{p\in \Delta(\mathcal{Y})}
\left\{
\mathbb{E}_{y\sim p(\cdot\mid u)}
\left[
r(u,y)
\right]
- \beta
D_{\mathrm{KL}}
\left(
p(\cdot\mid u)
\|
p_{\mathrm{ref}}(\cdot\mid u)
\right)
\right\}.
$$

Here $p_{\mathrm{ref}}$ is the reference policy, usually the SFT model, and $\beta>0$ controls the strength of the KL penalty.

As we saw in the previous post, the solution of this problem is

$$
p^*(y\mid u)
= \frac{
p_{\mathrm{ref}}(y\mid u)
\exp\left(r(u,y)/\beta\right)
}{
\sum_{y'}
p_{\mathrm{ref}}(y'\mid u)
\exp\left(r(u,y')/\beta\right)
}.
$$

This equation says that the optimal policy is obtained by tilting the reference policy toward high-reward responses. The key idea of DPO starts from this equation. Instead of learning a reward model separately, DPO asks:

$$
\text{Can we express the reward directly in terms of the optimal policy?}
$$

The answer is yes.

Starting from the previous formula, we can rearrange terms. We get

$$
r(u,y)
= \beta
\log
\frac{
p^*(y\mid u)
}{
p_{\mathrm{ref}}(y\mid u)
}
+
c(u),
$$

where

$$
c(u) =
\beta
\log
\sum_{y'}
p_{\mathrm{ref}}(y'\mid u)
\exp\left(r(u,y')/\beta\right).
$$

Therefore, for a fixed prompt, the reward is equivalent up to an additive constant to the log-ratio

$$
\log
\frac{
p^*(y\mid u)
}{
p_{\mathrm{ref}}(y\mid u)
}.
$$

This is the bridge between RLHF and DPO.

RLHF learns a reward model and then optimizes a policy against it. DPO instead uses the fact that, under the KL-regularized objective, the reward can be represented through the optimal policy itself. So rather than learning

$$
r_\phi(u,y)
$$

explicitly, DPO directly trains a policy

$$
p_\theta(y\mid u)
$$

whose log-ratio against the reference policy behaves like a reward.

# The Bradley--Terry Preference Model Revisited

In RLHF, preferences are usually modeled through the Bradley--Terry model:

$$
\mathbb{P}
\left(
y^+ \succ y^- \mid u
\right)
=
\sigma
\left(
r(u,y^+)-r(u,y^-)
\right),
$$

where $y^+$ is the preferred response and $y^-$ is the rejected response. In RLHF $r$ is learned separatly using preference data, but DPO instead replaces $r$ by,

$$r(u,y)= \beta \log\frac{p_{\theta}(y\mid u)}{p_{\text{ref}}(y\mid u)} +c(u)$$

Thus, the preference probability becomes: 

$$\mathbb{P}_{\theta}(y^{+}\succ y^{-} \mid u) = \sigma \left(\beta \left[ \log \frac{p_{\theta}(y^{+}\mid u)}{p_{\text{ref}}(y^{+}\mid u)}-\log \frac{p_{\theta}(y^{-}\mid u)}{p_{\text{ref}}(y^{-}\mid u)} \right] \right)$$


