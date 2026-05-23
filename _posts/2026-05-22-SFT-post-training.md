---
layout: post
title: "Supervised fine tuning"
date: 2026-05-22
tags: transformers post-training deep-learning
categories: machine-learning
math: true
mermaid: true
---
# Introduction 
In the previous post, we described the decoder-only Transformer architecture and the next-token prediction objective. We saw that an autoregressive language model takes a sequence of tokens and, at each position, outputs a probability distribution over the next token. This raises a natural question. If a model is trained only to predict the next token, why does it behave like an assistant?

A pretrained language model learns to continue text. But an assistant is expected to do something more specific: follow instructions, answer clearly, adapt to a user, refuse some unsafe requests, and maintain a conversational format. These behaviors are not guaranteed by next-token prediction alone. This is the role of **post-training**.

Post-training refers to the set of training procedures applied after pretraining in order to shape the behavior of a language model. In this post, we focus on the first and most basic post-training step: supervised fine-tuning, usually abbreviated as SFT.
# The Starting Point: A Pretrained Autoregressive Model
Let

$$
\mathcal{V}=\lbrace v_1,\dots,v_V\rbrace
$$

be a vocabulary of size $V$, and let

$$
x=(x_1,\dots,x_T)\in \lbrace 1,\dots,V\rbrace^T
$$

be a tokenized sequence. A decoder-only autoregressive Transformer defines, for every position $t$, a conditional probability distribution

$$
p_\theta(x_{t+1}\mid x_{\leq t}).
$$

After the Transformer stack has computed the hidden representation $h_t$ at position $t$, a final linear layer maps this representation to a vector of logits

$$
\ell_t = W_{\text{out}}h_t + b_{\text{out}} \in \mathbb{R}^V.
$$

The coordinate $\ell_{t,j}$ is the score assigned to token $j$ in the vocabulary. Applying the softmax function gives the next-token distribution:

$$
p_\theta(x_{t+1}=j \mid x_{\leq t})
=
\frac{\exp(\ell_{t,j})}
{\sum_{k=1}^{V}\exp(\ell_{t,k})}.
$$

During pretraining, the parameters $\theta$ are optimized so that the model assigns high probability to the actual next token in large text corpora. The corresponding loss is the negative log-likelihood

$$
\mathcal{L}_{\mathrm{pretrain}}(\theta)
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t}).
$$

Thus, pretraining teaches the model to continue text. The training data may contain books, websites, code, papers, forums, documentation, and many other kinds of text. As a result, the model learns a broad distribution over language: grammar, facts, reasoning patterns, styles, and common ways in which text continues.

But **a good text continuation model is not necessarily a good assistant**. To see why, suppose the prompt is

$$
u = \text{Explain the spectral theorem.}
$$

A pretrained language model treats this prompt as a prefix and tries to continue it in a statistically plausible way. Many continuations may be reasonable under the pretraining distribution, for example:

-  This question was asked in an exam ...
-  See Chapter 7 for a proof ...
- The spectral theorem is one of the most important results in linear algebra ...
  
All of these may be plausible continuations of text. But if the model is used as an assistant, we want a more specific behavior: we want the model to recognize the prompt as a request and produce a clear, useful answer. This is the central motivation for supervised fine-tuning. Pretraining learns a broad distribution over text. SFT modifies the model so that user prompts are more often followed by assistant-like responses.

# Prompt-Response Data
Supervised fine-tuning uses a dataset of prompt-response pairs. We write this dataset as

$$\{(u_i,y_i)\}_{i=1}^n.$$

Here, 

$$u_i= \text{prompt, instruction, or conversation context},$$

and

$$y_i= \text{desired assistant response.}$$

The important point is that the response $y_i$ is not arbitrary text. It is intended to represent the kind of behavior we want from the assistant: clear, useful, correct, and adapted to the prompt. Let

$$y_{i}=(y_{i,1},\dots,y_{i,m_{i}})$$

be a tokenized response associated with prompt $u_i$. Since the model is still autoregressive, the conditional probability of the full response is factorized token by token: 

$$p_{\theta}(y_i\mid u_i)=\prod_{t=1}^{m_{i}} p_{\theta}(y_{i,t}\mid u_i,y_{i,<t})$$

where 

$$y_{i,<t}=(y_{i,1},\dots,y_{i,t-1})$$

is the previous response tokens to the prompt $u_i$. 

This is the key point. SFT does not change the autoregressive nature of the model. At every step, the model still predicts the next token. What changes is the training distribution. During pretraining, the model predicts tokens in generic text. During SFT, the model predicts answer tokens conditioned on prompts.


# The SFT objective 
For one example $(u_i,y_i)$, the negative log-likelihood loss is:

$$
\mathcal{L}_{\mathrm{SFT}}^{(i)}(\theta)
=
-
\sum_{t=1}^{m_i}
\log
p_\theta
\left(
y_{i,t}
\mid
u_i, y_{i,<t}
\right).
$$

Averaging over the dataset gives the supervised fine-tuning objective

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\sum_{t=1}^{m_i}
\log
p_\theta
\left(
y_{i,t}
\mid
u_i, y_{i,<t}
\right).
$$

Thus, SFT is maximum likelihood training on curated prompt-response examples. The model is encouraged to assign high probability to the target response tokens. If the model assigns low probability to a token appearing in the desired answer, the loss increases. Training then adjusts the parameters so that similar responses become more likely in similar contexts.

The parameters $\theta$ are initialized from the pretrained model: 
$$
\theta \leftarrow \theta_0,
$$

where \(\theta_0\) denotes the parameters obtained after pretraining. SFT then continues gradient-based optimization, but now using the supervised dataset \(\mathcal D_{\mathrm{SFT}}\).

# What changes during SFT?
It is important to notice what does **not** change. The Transformer architecture is the same. The model still uses token embeddings, positional information, causal self-attention, MLP blocks, residual connections, layer normalization, and a final softmax over the vocabulary. The next-token prediction mechanism is also the same. What changes is the data distribution and therefore the behavior encouraged by the loss. During pretraining, the model sees sequences sampled from a broad text distribution, which we can denote informally by
$$x \sim \mathcal{D}_{\text{pretrain}}.$$
It learns to predict, 
$$x_t \text{from } x_{<t}$$
During SFT, the model sees formatted prompt-response examples
$$z_i=(u_i,y_i)$$
where $u_i$ is the prompt and $y_i$ is the target response. For response tokens, the prediction problem becomes
$$y_{i,t} \text{from } (u_i,y_{i,<t})$$
Thus, SFT changes the model from a distribution $p_{\theta_{0}}(\cdot\mid \text{generic prefix})$ toward a distribution $p_{\theta_{\text{SFT}}}(\cdot\mid \text{user prompt})$ that assigns more mass to assistant-like continuations.

# SFT as behavior cloning 
One useful interpretation of SFT is behavior cloning. In reinforcement learning language, a policy maps a context to a distribution over actions. For a language model, the context is a prompt and the action is a sequence of tokens. We can therefore write the language model as a policy
$$
\pi_\theta(y\mid u)
=
p_\theta(y\mid u).
$$

The SFT dataset contains demonstrations of the behavior we want: 
$$(u_i,y_i) \in \mathcal{D}_{\text{SFT}}$$

The model is trained to imitate these demonstrations. Given prompt $u_i$, it should assign high probability to the demonstrated response $y_i$.

Since $y_i$ is a sequence, the imitation happens token by token:

$$
\pi_\theta(y_i\mid u_i)
=
\prod_{t=1}^{m_i}
\pi_\theta(y_{i,t}\mid u_i,y_{i,<t}).
$$

This is why SFT can be understood as behavior cloning for language models. We provide examples of the desired assistant behavior, and the model learns to reproduce that behavior.

# SFT as Empirical Risk Minimization

A second viewpoint is the standard learning theory one. Assume that prompt-response pairs are sampled from some unknown distribution

$$
(u,y)\sim \mathcal D_{\mathrm{SFT}}.
$$

This distribution represents the behavior we want the assistant to learn. It may contain helpful answers, polite refusals, mathematical explanations, code solutions, summaries, or multi-turn conversations.

Ideally, we would like to minimize the population risk

$$
\mathcal R(\theta)= \mathbb E_{(u,y)\sim \mathcal D_{\mathrm{SFT}}}[-log(p_{\theta}(y|u))].
$$

Using the autoregressive factorization, this becomes

$$
\mathcal{R}(\theta)
=
\mathbb{E}_{(u,y)\sim\mathcal{D}_{\mathrm{SFT}}}
\left[
-
\sum_{t=1}^{m}
\log p_\theta\left(y_t \mid u, y_{<t}\right)
\right].
$$


The true distribution \(\mathcal D_{\mathrm{SFT}}\) is unknown. We only observe a finite dataset $\{(u_i,y_i)\}_{i=1}^n$. Therefore, we replace the expectation by an empirical average. This gives

$$
\widehat{\mathcal R}(\theta)
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\sum_{t=1}^{m_i}
\log p_\theta\left(y_{i,t}\mid u_i, y_{i,<t}\right).
$$

Thus,

$$
\theta_{\mathrm{SFT}}
\in
\arg\min_\theta
\widehat{\mathcal R}(\theta).
$$

This shows that SFT is empirical risk minimization with the token-level cross-entropy loss. This viewpoint is useful because it also reveals a limitation: SFT does not directly optimize abstract properties such as truthfulness, harmlessness, or usefulness. It minimizes a supervised loss on a finite set of demonstrations. If the demonstrations are high quality, the model can learn good behavior. If they are narrow, inconsistent, or biased, the model can inherit these limitations.

# Masked Loss on response tokens 

So far, we have described SFT at the level of prompt-response pairs and empirical risk minimization. However, the Transformer itself does not directly take a pair $(u_i,y_i)$ as two separate mathematical objects. It takes a single sequence of tokens as input. Therefore, in practice, each prompt-response pair is formatted as one sequence:
$$
z_i = (u_i,y_i).
$$

The prompt is used as context, but only the response is treated as the target. We encode this with a mask:

$$
M_{i,t}
=
\begin{cases}
0, & \text{if token } z_{i,t} \text{ belongs to the prompt},\\
1, & \text{if token } z_{i,t} \text{ belongs to the response}.
\end{cases}
$$

A minibatch loss can then be written as

$$
\mathcal L_B(\theta)
=
-
\frac{1}{\sum_{i\in B}\sum_t M_{i,t}}
\sum_{i\in B}
\sum_t
M_{i,t}
\log p_\theta(z_{i,t+1}\mid z_{i,\leq t}).
$$
The prompt tokens are used as context, but the model is not penalized for failing to reproduce the prompt. The model is trained to generate the response conditioned on the prompt. This distinction is important. SFT is not asking the model to copy the user. It is asking the model to produce the assistant side of the conversation.

# Minimal SFT Algorithm
We can summarize supervised fine-tuning (SFT) as follows.

Given:

$$
\theta_0 = \text{pretrained model parameters},
$$

and
$$
\mathcal D_{\mathrm{SFT}}
=
\{(u_i,y_i)\}_{i=1}^n.
$$

Initialize:

$$
\theta \leftarrow \theta_0.
$$

For minibatches \(B\subset \mathcal D_{\mathrm{SFT}}\):

1. Format each prompt-response pair as a single token sequence $z_i = (u_i,y_i).$
2. Build a mask \(M_i\).
3. Run the transformer on \(z_i\) and compute logits at every position.

4. Compute the masked cross-entropy loss

$$
\mathcal L_B(\theta)
=
-
\frac{1}{\sum_{i\in B}\sum_t M_{i,t}}
\sum_{i\in B}
\sum_t
M_{i,t}
\log p_\theta(z_{i,t+1}\mid z_{i,\leq t}).
$$

5. Update the parameters using gradient descent or one of its variants:

$$
\theta
\leftarrow
\theta
-
\eta \nabla_\theta \mathcal L_B(\theta).
$$

Return:

$$
\theta_{\mathrm{SFT}}.
$$


# Why Does SFT Work Well?

SFT works well because it starts from a model that already understands a lot about language.

During pretraining, the model has learned grammar, factual associations, code patterns, mathematical language, and many common ways in which text is structured. SFT does not need to teach all of this from scratch. Instead, it teaches the model how to use these abilities in an assistant-like setting.

Before SFT, a prompt is mainly treated as text to continue. After SFT, the same prompt is more likely to be treated as an instruction to answer.

This is why SFT can improve behaviors such as following instructions, answering directly, formatting responses clearly, maintaining a consistent assistant style, and adapting to multi-turn conversations.


# Limitations of SFT

However, SFT also has important limitations.

The first limitation is that SFT only learns from positive demonstrations. For each prompt \(u_i\), the dataset usually contains one desired answer \(y_i\). The loss tells the model:

$$
\text{increase the probability of } y_i.
$$

But it does not explicitly compare \(y_i\) with other possible answers.

Suppose we have two responses to the same prompt:

$$
y^+ = \text{a clear, correct, helpful answer},
$$

and

$$
y^- = \text{an answer that is plausible but misleading}.
$$

SFT trains on \(y^+\), but it does not directly say:

$$
y^+ \succ y^-.
$$

It only increases the likelihood of the demonstrated answer.

The second limitation is that the SFT objective is token-level, while the qualities we care about are often sequence-level.

The loss is
$$
-
\sum_t
\log p_\theta(y_t\mid u,y_{<t}).
$$

This evaluates the response one token at a time. But human preferences are usually about the whole answer:

$$
\text{Was it helpful?}
$$

$$
\text{Was it truthful?}
$$

$$
\text{Was it safe?}
$$

$$
\text{Was it too verbose, too vague, or too confident?}
$$

These are not naturally token-level properties.

The third limitation is that there may be many good answers to the same prompt. SFT usually treats one demonstration as the target, even though several responses could be equally good or better. This can make the objective too narrow.

The fourth limitation is distribution shift. The model may behave well on prompts similar to the SFT dataset, but users can ask unusual, adversarial, ambiguous, or highly specialized questions. In those cases, imitation of the SFT data may not be enough.

Thus, SFT is a strong first post-training step, but it does not fully solve the problem of aligning model behavior with human preferences.

# From demonstrations to preferences
The limitations of SFT motivate the next stage of post-training. Instead of only showing the model examples of good answers, we can compare answers. Given a prompt $u$, suppose we have two candidate responses:

$$
y^+ = \text{preferred response},
$$

and

$$
y^- = \text{rejected response}.
$$

The supervision is now a preference relation

$$
y^+ \succ y^-.
$$

This kind of data contains information that SFT does not directly use. It tells the model not only what a good answer may look like, but also which answer is better when several possibilities are available.

This leads to preference-based post-training methods such as reward modeling, reinforcement learning from human feedback, and direct preference optimization.

These will be the topic of the next post.

