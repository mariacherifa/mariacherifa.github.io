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
\text{Which response is better?}
$$

But for safety, the question changes slightly. We are not only comparing responses by quality. We also need to ask whether a response is acceptable at all.

For a given prompt $u$, let

$$
\mathcal{Y}
$$

denote the set of possible responses. In a safety-sensitive setting, we can think of this set as being divided into two regions:

$$
\mathcal{Y}_{\mathrm{safe}}(u)
\subseteq
\mathcal{Y},
$$

and

$$
\mathcal{Y}\setminus \mathcal{Y}_{\mathrm{safe}}(u).
$$

Here $\mathcal{Y}{\mathrm{safe}}(u)$ is the set of responses that are acceptable for the prompt $u$, while $\mathcal{Y}{\mathrm{unsafe}}(u)$ contains responses that should not be produced.

For example, if the prompt is a normal mathematical question, then a direct answer belongs to $\mathcal{Y}{\mathrm{safe}}(u)$. But if the prompt asks for dangerous instructions, private information, or harmful advice, then a detailed answer may belong to $\mathcal{Y}{\mathrm{unsafe}}(u)$. In that case, a safe response may instead be a refusal, a redirection, or high-level non-actionable information.

So safety is not simply about making the model less helpful. It is about restricting the space of acceptable responses depending on the prompt.

Ideally, we would like the model to put most of its probability mass on safe responses:

$$
p_\theta
\left(
\mathcal{Y}_{\mathrm{safe}}(u)
\mid u
\right)
\approx 1.
$$

Equivalently, we want

$$
p_\theta
\left(
\mathcal{Y}_{\mathrm{unsafe}}(u)
\mid u
\right)
\approx 0.
$$

This gives a natural constrained optimization view. We still want the model to produce useful responses, but we want the probability of unsafe responses to remain small.

Let

$$
\mathcal{L}_{\mathrm{help}}(\theta)
$$

denote a helpfulness-oriented training loss. This could be an SFT loss, an RLHF objective, or a DPO loss, depending on the post-training method we are using.

Safety fine-tuning adds a constraint of the form

$$
\mathbb{E}{u\sim \mathcal{D}{U}}
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

So the constrained problem becomes

$$
\min_\theta
\mathcal{L}_{\mathrm{help}}(\theta)
$$

subject to

$$
\mathbb{E}{u\sim \mathcal{D}{U}}
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
\text{optimize helpfulness, but keep unsafe behavior rare.}
$$

The parameter $\varepsilon$ controls how much unsafe behavior we are willing to tolerate. A smaller $\varepsilon$ corresponds to a stricter safety constraint.

As usual in constrained optimization, we can replace the constraint by a regularization term. This gives an objective of the form

$$
\min_\theta
\mathcal{L}{\mathrm{help}}(\theta)
+
\lambda
,
\mathbb{E}{u\sim \mathcal{D}{U}}
\left[
p\theta
\left(
\mathcal{Y}_{\mathrm{unsafe}}(u)
\mid u
\right)
\right],
$$

where

$$
\lambda>0
$$

controls the strength of the safety penalty.

This is the regularized view of safety fine-tuning. The first term trains the model to be useful. The second term penalizes probability mass assigned to unsafe responses.

In practice, we do not know the full set

$$
\mathcal{Y}_{\mathrm{unsafe}}(u)
$$

for every prompt $u$. We only observe examples. This is why safety fine-tuning is implemented through data.

In a supervised safety dataset, we may have examples

$$
(u_i,y_i),
$$

where $y_i$ is the desired safe response. If the prompt is safe, $y_i$ can be a direct answer. If the prompt is unsafe, $y_i$ can be a refusal or a safe redirection.

Then the training loss is still a negative log-likelihood,

$$
-\log p_\theta(y_i\mid u_i),
$$

but the target response now encodes the safety behavior we want.

In a preference-based safety dataset, we may instead have triples

$$
(u_i,y_i^+,y_i^-),
$$

where

$$
y_i^+ \in \mathcal{Y}_{\mathrm{safe}}(u_i)
$$

is the safer response, and

$$
y_i^- \in \mathcal{Y}_{\mathrm{unsafe}}(u_i)
$$

is the unsafe or less appropriate response.

The preference label says

$$
y_i^+ \succ y_i^-.
$$

This fits directly into DPO. The DPO objective will increase the relative probability of the safer response and decrease the relative probability of the unsafe response.

Thus, safety fine-tuning can be understood as constrained behavior shaping. We are still modifying

$$
p_\theta(y\mid u),
$$

but now we are not only asking which responses are preferred. We are also imposing that certain regions of the response space should receive very small probability.

The main idea is therefore:

$$
\text{post-training for helpfulness}
\quad
\longrightarrow
\quad
\text{post-training under safety constraints}.
$$
