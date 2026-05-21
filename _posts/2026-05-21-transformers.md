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

where $V = |\mathcal{V}|$ is the size of the vocabulary.

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

At this stage, a text is represented as a sequence of integers,$(x_1,\dots,x_T) \in \lbrace 1,\dots,V \rbrace^T$. Each integer $x_t$ identifies one token in the vocabulary. However, an integer index is not a very meaningful object for a neural network. For example, if the token `"cat"` has index $42$ and the token `"dog"` has index $43$, this does not mean that `"dog"` is mathematically larger than `"cat"` or that they are close simply because their indices are close. Therefore, we need to transform each discrete token index into a vector representation. This is the role of \textbf{the embedding layer}.

A first way to represent a token $x_t$ is through a one-hot vector. The one-hot vector associated with $x_t$ is denoted by $e_{x_t}$ and belongs to $\mathbb{R}^V$:

$$
e_{x_t} = (0,\dots,0,1,0,\dots,0) \in \mathbb{R}^V.
$$

The only nonzero coordinate is the one corresponding to the index $x_t$. This representation is useful because it identifies each token uniquely. However, One-hot vectors are very high-dimensional, sparse, and do not encode semantic similarity between tokens. To obtain a dense and learnable representation, we introduce an embedding matrix,

$$
E \in \mathbb{R}^{V \times d}.
$$

Each row of $E$ corresponds to one token in the vocabulary. $V$ is the vocabulary size, while $d$ is the embedding dimension. The vector representation of token $x_t$ is obtained by selecting the row of $E$ corresponding to $x_t$:

$$
h_t^{(0)} = e_{x_t}^{\top} E \in \mathbb{R}^d.
$$

The notation $h_t^{(0)}$ means the initial hidden representation of the token at position $t$. The superscript $(0)$ indicates that this vector is the input representation before applying any Transformer layer. After the first Transformer block, it will become $h_t^{(1)}$; after the second block, $h_t^{(2)}$; and so on.

So the embedding layer maps each token index to a dense vector:

$$
x_t \longmapsto h_t^{(0)} \in \mathbb{R}^d.
$$

The matrix $E$ is not chosen manually. It is a learnable parameter of the model. At the beginning of training, its entries are initialized randomly. During training, the model updates $E$ using gradient descent, so that tokens used in similar contexts tend to acquire useful vector representations. The dimension $d$ is a hyperparameter. It controls the size of the vector used to represent each token. A small value of $d$ gives a compact representation, but may not be expressive enough. A large value of $d$ allows the model to store richer information, but increases the number of parameters and the computational cost. In practice, $d$ is chosen depending on the size of the model, the amount of data, and the available computational resources.
