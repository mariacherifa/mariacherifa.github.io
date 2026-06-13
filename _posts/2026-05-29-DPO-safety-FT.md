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

# From Preference Optimization to Safety Fine-Tuning

So far, post-training has been about shaping the conditional distribution

$$
p_\theta(y\mid u).
$$

SFT does this by imitating desired responses. RLHF and DPO do it by using preference data, so that preferred responses become more likely than rejected ones.

Until now, the main question was:

$$
\text{Which response should the model prefer?}
$$

For safety, we need a slightly different question. We do not only want to rank responses by quality. We also want to distinguish responses that are acceptable from responses that should not be produced.

For a given prompt $u$, let

$$
\mathcal{Y}
$$

be the set of possible responses. In a safety-sensitive setting, we can introduce a subset

$$
\mathcal{Y}_{\mathrm{safe}}(u)
\subseteq
\mathcal{Y},
$$

which represents the set of acceptable responses for this prompt. Its complement is

$$
\mathcal{Y}_{\mathrm{unsafe}}(u)= \mathcal{Y}\setminus \mathcal{Y}_{\mathrm{safe}}(u).
$$

contains responses that should not be produced for the prompt $u$.

For example, if $u$ is a standard mathematical question, then a direct explanation may belong to

$$
\mathcal{Y}_{\mathrm{safe}}(u).
$$

But if $u$ asks for dangerous instructions, private information, or harmful advice, then a detailed answer may belong to

$$
\mathcal{Y}_{\mathrm{unsafe}}(u).
$$

In that case, a safe response may instead be a refusal, a redirection, or high-level non-actionable information.

So safety is not the opposite of helpfulness. Rather, safety restricts the set of responses among which the model should be helpful.

Ideally, for each prompt $u$, we would like the model to assign almost all its probability mass to safe responses:

$$
p_\theta
\left(
\mathcal{Y}_{\mathrm{safe}}(u)
\mid u
\right)
\approx 1.
$$

Equivalently, we would like the probability of unsafe responses to be small:

$$
p_\theta
\left(
\mathcal{Y}_{\mathrm{unsafe}}(u)
\mid u
\right)
\approx 0.
$$

This gives a natural constrained optimization view of safety fine-tuning.

Let

$$
\mathcal{L}_{\mathrm{task}}(\theta)
$$

denote the post-training objective we would optimize without the safety constraint. Depending on the method, this could be an SFT loss, an RLHF objective, or a DPO loss.

Safety fine-tuning adds a constraint on the probability mass assigned to unsafe responses:

$$
\mathbb{E}_{u\sim \mathcal{D}_{U}}
\left[
p_\theta
\left(
\mathcal{Y}_{\mathrm{unsafe}}(u)
\mid u
\right)
\right]
\leq
\varepsilon.
$$

Thus, we can write the constrained problem as

$$
\min_\theta
\mathcal{L}_{\mathrm{task}}(\theta)
$$

subject to

$$
\mathbb{E}_{u\sim \mathcal{D}_{U}}
\left[
p_\theta
\left(
\mathcal{Y}_{\mathrm{unsafe}}(u)
\mid u
\right)
\right]
\leq
\varepsilon.
$$

This formulation says:

$$
\text{optimize the post-training objective, but keep unsafe behavior rare.}
$$

The parameter

$$
\varepsilon
$$

controls the strength of the safety requirement. A smaller value of (\varepsilon) means that the model is allowed to put less probability mass on unsafe responses.

As usual in constrained optimization, we can pass from the constrained problem to a regularized objective. Introducing a multiplier

$$
\lambda>0,
$$

we obtain

$$
\min_\theta
\mathcal{L}{\mathrm{task}}(\theta)
+
\lambda
\mathbb{E}_{u\sim \mathcal{D}_{U}}
\left[
p\theta
\left(
\mathcal{Y}_{\mathrm{unsafe}}(u)
\mid u
\right)
\right].
$$

The first term optimizes the usual post-training objective. The second term penalizes probability mass assigned to unsafe responses.

So, from this point of view, safety fine-tuning is a regularized post-training problem:

$$
\text{post-training objective}
\quad
+
\quad
\text{safety penalty}.
$$

This formulation is useful conceptually. It says that safety is not only about adding refusal examples. It is about changing the distribution

$$
p_\theta(y\mid u)
$$

so that unsafe regions of the response space receive small probability.

Of course, in practice, we do not know the full set

$$
\mathcal{Y}_{\mathrm{unsafe}}(u)
$$

for every prompt. The set of all possible unsafe responses is too large to enumerate. Therefore, the safety constraint has to be approximated from data.

In supervised safety fine-tuning, the data consists of examples

$$
(u_i,y_i),
$$

where (y_i) is the desired safe response to the prompt (u_i). If the prompt is safe, (y_i) may be a direct answer. If the prompt is unsafe, (y_i) may be a refusal, a redirection, or a safer alternative.

The training objective is still a negative log-likelihood term,

$$
-\log p_\theta(y_i\mid u_i),
$$

but the target response now encodes safety behavior.

In preference-based safety fine-tuning, the data consists of triples

$$
(u_i,y_i^+,y_i^-),
$$

where (y_i^+) is safer or more appropriate than (y_i^-). Ideally,

$$
y_i^+ \in \mathcal{Y}_{\mathrm{safe}}(u_i),
$$

while

$$
y_i^- \in \mathcal{Y}_{\mathrm{unsafe}}(u_i)
$$

or is at least less safe. The preference label says

$$
y_i^+ \succ y_i^-.
$$

This fits naturally with DPO. The DPO objective can increase the policy-to-reference ratio of the safer response and decrease the policy-to-reference ratio of the unsafe response.

Thus, safety fine-tuning can be understood as constrained behavior shaping. We are still modifying the same conditional distribution

$$
p_\theta(y\mid u),
$$

but the objective now includes a safety constraint: unsafe regions of the response space should receive small probability.

The main idea is therefore

$$
\text{post-training for helpfulness}
\quad
\longrightarrow
\quad
\text{post-training under safety constraints}.
$$
