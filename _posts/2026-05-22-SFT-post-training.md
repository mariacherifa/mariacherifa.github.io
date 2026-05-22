---
layout: post
title: "Supervised fine tuning"
date: 2026-05-22
tags: transformers post training deep-learning
categories: machine-learning
math: true
mermaid: true
---

# The Starting Point: A Pretrained Autoregressive Model

In the previous post, we saw how a decoder-only Transformer takes a sequence of tokens and produces, at each position, a probability distribution over the next token. Let

$$
\mathcal{V}=\lbrace v_1,\dots,v_V\rbrace
$$

be a vocabulary of size $V$, and let

$$
x=(x_1,\dots,x_T)\in \lbrace 1,\dots,V\rbrace^T
$$

be a tokenized sequence. For each prefix $x_{\leq t}=(x_1,\dots,x_t)$, an autoregressive language model defines a conditional distribution

$$
p_\theta(x_{t+1}\mid x_{\leq t}).
$$

The model predicts the next token, given the tokens that came before it.

After the Transformer stack has computed the hidden representation $h_t$ at position $t$, a final linear layer maps this representation to a vector of logits

$$
\ell_t \in \mathbb{R}^V.
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
u = \text{``Explain the spectral theorem.''}
$$

A pretrained language model treats this prompt as a prefix and tries to continue it in a statistically plausible way. Many continuations may be reasonable under the pretraining distribution, for example:

- ``This question was asked in an exam ...''
- ``See Chapter 7 for a proof ...''
- ``The spectral theorem is one of the most important results in linear algebra ...''
  
All of these are plausible continuations of the prompt. However, if the model is being used as an assistant, we want something more specific. We do not only want a continuation that is likely under the distribution of internet text. We want a continuation that is helpful, clear, and adapted to the user’s request.

This is the motivation for post-training. Pretraining gives the model broad linguistic and conceptual competence. Post-training tries to shape this competence into a more useful behavior. In particular, it changes the model so that prompts are treated less like arbitrary prefixes to continue, and more like instructions to answer.

The first and simplest step in this direction is supervised fine-tuning, usually abbreviated as SFT.
