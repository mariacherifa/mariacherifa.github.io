---
layout: post
title: "Understanding Transformers with math and intuitions"
date: 2026-05-21
tags: transformers attention deep-learning
categories: machine-learning
math: true
---

# Introduction

The goal of this blog post is to understand the main components of Transformers, while providing both mathematical insights and intuition.

# From Text to Tokens

To treat text as a mathematical object, we first need to define two important notions: a **vocabulary** and a **sequence of tokens**.

A vocabulary is a finite set of symbols used to represent text. These symbols can be characters, subwords, words, numbers, punctuation marks, or special tokens. We denote the vocabulary by

$$
\mathcal{V} = \lbrace v_1, \dots, v_V \rbrace,
$$

where $V = \lvert \mathcal{V}\rvert$ is the size of the vocabulary.

We define an indexing function,  $\mathrm{idx} : \mathcal{V} \to \{1,\dots,V\}$ by, 

$$
\mathrm{idx}(v_i)=i.
$$

This function maps each element in the vocabulary to its integer index. Conversely, we define a decoding function $\mathrm{tok}: \lbrace 1,\dots,V \rbrace \to \mathcal{V}$ by

$$
\mathrm{tok}(i)=v_i.
$$

Once a vocabulary and an indexing function are fixed, a text can be represented as a discrete sequence of token indices:

$$
(x_1, \dots, x_T) \in \lbrace 1,\dots,V
\rbrace^T.
$$

Here, $T$ denotes the length of the tokenized text, and each integer $x_t$ corresponds to the token

$$
\mathrm{tok}(x_t)=v_{x_t}.
$$

In other words, $x_t$ is not the token itself, but the index of a token in the vocabulary.

# From Tokens to Embeddings

At this stage, a text is represented as a sequence of integers, $(x_1,\dots,x_T) \in \lbrace 1,\dots,V \rbrace^T$. Each integer $x_t$ identifies one token in the vocabulary. However, an integer index is not a very meaningful object for a neural network. For example, if the token `"cat"` has index $42$ and the token `"dog"` has index $43$, this does not mean that `"dog"` is mathematically larger than `"cat"` or that they are close simply because their indices are close. Therefore, we need to transform each discrete token index into a vector representation. This is the role of **the embedding layer**.

A first way to represent a token $x_t$ is through a one-hot vector. The one-hot vector associated with $x_t$ is denoted by $e_{x_t}$ and belongs to $\mathbb{R}^V$:

$$
e_{x_t} = (0,\dots,0,1,0,\dots,0) \in \mathbb{R}^V.
$$

The only nonzero coordinate is the one corresponding to the index $x_t$. This representation is useful because it identifies each token uniquely. However, one-hot vectors are very high-dimensional, sparse, and do not encode semantic similarity between tokens. To obtain a dense and learnable representation, we introduce an embedding matrix

$$
E \in \mathbb{R}^{V \times d}.
$$

Each row of $E$ corresponds to one token in the vocabulary. The integer $V$ is the vocabulary size, while $d$ is the embedding dimension. The vector representation of token $x_t$ is obtained by selecting the row of $E$ corresponding to $x_t$:

$$
z_t = e_{x_t}^{\top}E \in \mathbb{R}^d.
$$

Equivalently, $z_t$ is simply the row of $E$ associated with the token index $x_t$.

At this point, $z_t$ contains information about the identity of the token. However, it does not contain information about the position of the token in the sequence. This is an important issue for Transformers. Unlike recurrent neural networks, Transformers process all tokens in parallel, so they do not naturally know whether a token appears at the beginning, in the middle, or at the end of the sentence.

For example, the sentences `"the cat chased the dog"` and `"the dog chased the cat"` contain almost the same tokens, but their meanings are different because the order of the tokens is different. Therefore, we need to add positional information to the token embeddings.

To do this, we introduce a positional embedding vector $p_t \in \mathbb{R}^d$ for each position $t$. The initial representation of token $x_t$ is then obtained by adding its token embedding and its positional embedding:

$$
h_t^{(0)} = z_t + p_t.
$$

Since both $z_t$ and $p_t$ belong to $\mathbb{R}^d$, their sum is also a vector in $\mathbb{R}^d$:

$$
h_t^{(0)} \in \mathbb{R}^d.
$$

In matrix form, if we define

$$
Z =
\begin{pmatrix}
z_1 \\
z_2 \\
\vdots \\
z_T
\end{pmatrix}
\in \mathbb{R}^{T \times d}
$$

and

$$
P =
\begin{pmatrix}
p_1 \\
p_2 \\
\vdots \\
p_T
\end{pmatrix}
\in \mathbb{R}^{T \times d},
$$

then the input matrix to the Transformer is

$$
H^{(0)} = Z + P \in \mathbb{R}^{T \times d}.
$$

The notation $h_t^{(0)}$ means the initial hidden representation of the token at position $t$. The superscript $(0)$ indicates that this vector is the input representation before applying any Transformer layer. After the first Transformer block, it will become $h_t^{(1)}$; after the second block, $h_t^{(2)}$; and so on.

So the embedding step maps each token and its position to a dense vector:

$$
(x_t,t) \longmapsto h_t^{(0)} \in \mathbb{R}^d.
$$

The matrix $E$ is not chosen manually. It is a learnable parameter of the model. At the beginning of training, its entries are initialized randomly. During training, the model updates $E$ using gradient descent, so that tokens used in similar contexts tend to acquire useful vector representations.

The positional vectors $p_t$ can be defined in different ways. In some Transformers, they are also learned parameters, just like the token embeddings. In the original Transformer architecture, they were defined using sinusoidal functions, which gives the model a structured way to represent positions. In both cases, their role is the same: they inject information about the order of the tokens into the model.

The dimension $d$ is a hyperparameter. It controls the size of the vector used to represent each token and each position. A small value of $d$ gives a compact representation, but may not be expressive enough. A large value of $d$ allows the model to store richer information, but increases the number of parameters and the computational cost. In practice, $d$ is chosen depending on the size of the model, the amount of data, and the available computational resources.

# Self-Attention

At this stage, the text is represented by a matrix

$$
H =
\begin{pmatrix}
h_1 \\
h_2 \\
\vdots \\
h_T
\end{pmatrix}
\in \mathbb{R}^{T \times d},
$$

where each row $h_t \in \mathbb{R}^d$ is the embedding representation of the token $x_t$.

This gives us a vector representation of each token. However, these vectors are still not **fully contextual**. The representation of a token should not depend only on the token itself, but also on the other tokens around it. For example, the meaning of a word can change depending on the sentence in which it appears. Therefore, if we want to predict the next token or understand the structure of a sequence, we need representations that take the whole context into account.

This is the main intuition behind self-attention: **self-attention allows each token to build a new representation by looking at the other tokens in the same sequence**.

Mathematically, self-attention starts by constructing three matrices from $H$:

$$
Q = HW_Q,
\qquad
W_Q \in \mathbb{R}^{d \times d_k},
$$

$$
K = HW_K,
\qquad
W_K \in \mathbb{R}^{d \times d_k},
$$

$$
V = HW_V,
\qquad
W_V \in \mathbb{R}^{d \times d_v}.
$$

Therefore,

$$
Q \in \mathbb{R}^{T \times d_k},
\qquad
K \in \mathbb{R}^{T \times d_k},
\qquad
V \in \mathbb{R}^{T \times d_v}.
$$

The rows of these matrices are denoted by

$$
Q =
\begin{pmatrix}
q_1 \\
q_2 \\
\vdots \\
q_T
\end{pmatrix},
\qquad
K =
\begin{pmatrix}
k_1 \\
k_2 \\
\vdots \\
k_T
\end{pmatrix},
\qquad
V =
\begin{pmatrix}
v_1 \\
v_2 \\
\vdots \\
v_T
\end{pmatrix}.
$$

The vectors $q_t$, $k_t$, and $v_t$ are called the **query**, **key**, and **value** vectors associated with token $x_t$.

Intuitively, for a token $x_t$ represented by $h_t$:

- $q_t$ is the query: it represents what token $x_t$ is looking for in the other tokens.
- $k_s$ is the key of another token $x_s$: it represents what token $x_s$ offers, or how it can be matched.
- $v_s$ is the value of token $x_s$: it represents the information that token $x_s$ can contribute if it is considered relevant.

So the self-attention mechanism answers the following question:

> For a given token $x_t$, which other tokens $x_s$ should it pay attention to in order to build a better contextual representation?

To measure the relevance of token $x_s$ to token $x_t$, we compare the query $q_t$ with the key $k_s$. This comparison is done using a dot product:

$$
q_t^\top k_s.
$$

If this quantity is large, then token $x_s$ is highly relevant to token $x_t$. If it is small, then token $x_s$ is less relevant to token $x_t$.

Thus, we define the attention score between token $t$ and token $s$ by

$$
a_{t,s}
=
\frac{q_t^\top k_s}{\sqrt{d_k}}.
$$

**Remark.** The normalization factor $\sqrt{d_k}$ is important when the dimension of the key and query vectors is large. Indeed, the dot product $q_t^\top k_s$ is obtained by summing over $d_k$ coordinates, so its magnitude tends to grow with $d_k$. Without this normalization, the scores could become too large in magnitude. This would make the softmax very sharp, meaning that most of the probability mass would be assigned to only a few tokens. Dividing by $\sqrt{d_k}$ keeps the scores on a more stable scale, leading to smoother attention weights and more stable optimization.

For a fixed token $x_t$, we compute one score with every token in the sequence:

$$
a_{t,1}, a_{t,2}, \dots, a_{t,T}.
$$

These scores are then transformed into probabilities using the softmax function:

$$
\alpha_{t,s}
=
\frac{\exp(a_{t,s})}
{\sum_{r=1}^{T} \exp(a_{t,r})}.
$$

The coefficients $\alpha_{t,s}$ are called **attention weights**. For a fixed $t$, they satisfy

$$
\alpha_{t,s} \geq 0, \qquad \sum_{s=1}^{T} \alpha_{t,s}=1.
$$

Therefore, for each token $x_t$, the attention weights define a probability distribution over all tokens in the sequence. The quantity $\alpha_{t,s}$ tells us how much token $x_t$ attends to token $x_s$.

Once we have these attention weights, we compute the new representation of token $x_t$ as a weighted average of the value vectors:

$$
\widetilde{h}_t
=
\sum_{s=1}^{T} \alpha_{t,s} v_s.
$$

This vector $\widetilde{h}_t$ is the contextual representation of token $x_t$. It is no longer built only from the token itself. Instead, it is built from all tokens in the sequence, weighted by their relevance to $x_t$.

**Remark.** We can think of self-attention as producing a kernel-like representation of tokens, because each output vector is a weighted combination of value vectors. However, there is an important difference with classical kernel methods: here, the weights are not fixed by a predefined kernel. They are learned through the matrices $W_Q$, $W_K$, and $W_V$, and they also depend on the input sequence itself.

In matrix form, all attention scores can be computed at once as

$$
A
=
\frac{QK^\top}{\sqrt{d_k}}.
$$

Here,

$$
A \in \mathbb{R}^{T \times T},
$$

and the entry $A_{t,s}$ is the attention score between token $x_t$ and token $x_s$.

Then, we apply the softmax row by row:

$$
P
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right).
$$

Each row of $P$ contains the attention weights for one token. In other words,

$$
P_{t,s} = \alpha_{t,s}.
$$

Finally, the output of self-attention is

$$
\widetilde{H}
=
PV.
$$

Equivalently,

$$
\widetilde{H}
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
V.
$$

This is the **scaled dot-product self-attention** formula.

The important point is that each row $\widetilde{h}_t$ of $\widetilde{H}$ is a new representation of token $x_t$ that depends on the entire sequence. This is why self-attention is a mechanism for building context-dependent representations.
