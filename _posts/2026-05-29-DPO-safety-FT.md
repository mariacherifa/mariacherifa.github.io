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

Substituting this expression into the Bradley--Terry model gives

$$\mathbb{P}_{\theta}(y^{+}\succ y^{-} \mid u) = \sigma \left(\beta \left[ \log \frac{p_{\theta}(y^{+}\mid u)}{p_{\text{ref}}(y^{+}\mid u)}-\log \frac{p_{\theta}(y^{-}\mid u)}{p_{\text{ref}}(y^{-}\mid u)} \right] \right)$$

So DPO trains the policy so that the preferred response has a larger policy-to-reference log-ratio than the rejected response:

$$
\log
\frac{
p_\theta(y^+\mid u)
}{
p_{\mathrm{ref}}(y^+\mid u)
}
>
\log
\frac{
p_\theta(y^-\mid u)
}{
p_{\mathrm{ref}}(y^-\mid u)
}.
$$

This is the central idea of DPO. The model does not need an explicit reward model. The quantity

$$
\beta
\log
\frac{
p_\theta(y\mid u)
}{
p_{\mathrm{ref}}(y\mid u)
}
$$

plays the role of an implicit reward.

# DPO as Maximum Likelihood Estimation

Let

$$
\mathcal{D}_{\mathrm{pref}}
$$

denote the distribution of preference data. A sample from this distribution is a triple

$$
(u,y^+,y^-)\sim \mathcal{D}_{\mathrm{pref}},
$$

where \(y^+\) is preferred to \(y^-\).

Under the DPO preference model, the probability of observing this preference is

$$
\mathbb{P}_{\theta}
\left(
y^+ \succ y^- \mid u
\right)
=
\sigma
\left(
\beta
\left[
\log
\frac{
p_\theta(y^+\mid u)
}{
p_{\mathrm{ref}}(y^+\mid u)
}
-
\log
\frac{
p_\theta(y^-\mid u)
}{
p_{\mathrm{ref}}(y^-\mid u)
}
\right]
\right).
$$

Therefore, DPO can be trained by maximum likelihood on preference pairs. Equivalently, we minimize the negative log-likelihood of the observed preferences:

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
-
\mathbb{E}_{(u,y^+,y^-)\sim \mathcal{D}_{\mathrm{pref}}}
\left[
\log
\sigma
\left(
\beta
\left[
\log
\frac{
p_\theta(y^+\mid u)
}{
p_{\mathrm{ref}}(y^+\mid u)
}
-
\log
\frac{
p_\theta(y^-\mid u)
}{
p_{\mathrm{ref}}(y^-\mid u)
}
\right]
\right)
\right].
$$

This is the DPO objective.

The key point is that DPO compares the preferred and rejected responses through their policy-to-reference ratios, not only through their raw probabilities under the policy. More precisely, it compares

$$
\frac{
p_\theta(y^+\mid u)
}{
p_{\mathrm{ref}}(y^+\mid u)
}
\qquad
\text{and}
\qquad
\frac{
p_\theta(y^-\mid u)
}{
p_{\mathrm{ref}}(y^-\mid u)
}.
$$

Thus, DPO encourages the preferred response to become more likely relative to the reference model, and the rejected response to become less likely relative to the reference model. The reference policy acts as an anchor: the model is not only learning which response is better, but also how to move away from the supervised model in a controlled way.

# Toward Safety Fine-Tuning

DPO gives a direct way to train a language model from preference pairs. Until now, we have mostly interpreted these preferences as helpfulness preferences: one answer may be clearer, more relevant, or more useful than another.

However, preference data can also encode safety. For a given prompt $u$, we may compare a safer response $y^+$ with an unsafe or less appropriate response $y^-$. The preference label

$$
y^+ \succ y^-
$$

then tells the model that $y^+$ should be preferred to $y^-$.

In the DPO objective, this means increasing the policy-to-reference ratio of $y^+$, while decreasing the policy-to-reference ratio of $y^-$. Therefore, the model is not only learning which response is more helpful. It can also learn which response is more appropriate under safety constraints. This gives a natural bridge from DPO to safety fine-tuning.

In safety fine-tuning, post-training is no longer only about making the model more helpful. It is about making the model helpful within a constrained response space.

In the next post, we make this idea more explicit by introducing admissible response sets, safety costs, and constrained conditional generation.
