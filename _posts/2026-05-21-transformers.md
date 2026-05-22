---
layout: post
title: "Notes on Transformers"
date: 2026-05-21
tags: transformers attention deep-learning
categories: machine-learning
math: true
mermaid: true
---

# Introduction

Transformers are at the heart of modern language models, but their inner workings can feel intimidating at first. The goal of this post is to make the main components of Transformers easier to understand, blending mathematical insights with intuition along the way.

To keep things focused, we will look at **the decoder-only Transformer architecture**: the architecture behind autoregressive language models such as GPT-style models. The original Transformer was introduced with an encoder-decoder structure, but most modern language models for next-token prediction use a decoder-only design, where each layer gradually refines the representation of the input sequence.

We will walk through the full pipeline step by step: starting from raw tokens, turning them into rich contextual representations through self-attention and MLP layers, and finally using those representations to predict the next token.

# From Text to Tokens

To treat text as a mathematical object, we first need to define two fundamental notions: a **vocabulary** and a **sequence of tokens**.

A vocabulary is a finite set of symbols used to represent text. These symbols can be characters, subwords, words, numbers, punctuation marks, or special tokens. We denote the vocabulary by

$$
\mathcal{V} = \lbrace v_1, \dots, v_V \rbrace,
$$

where $V = \lvert \mathcal{V}\rvert$ is the size of the vocabulary.

We define an indexing function,  $\mathrm{idx} : \mathcal{V} \to \lbrace 1,\dots,V\rbrace$ by, 

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

At this stage, a text is represented as a sequence of integers, $(x_1,\dots,x_T) \in \lbrace 1,\dots,V \rbrace^T$. However, an integer index is not a very meaningful object for a neural network. For example, if the token `"cat"` has index $42$ and the token `"dog"` has index $43$, this does not mean that `"dog"` is mathematically larger than `"cat"` or that they are close simply because their indices are close. Therefore, we need to transform each discrete token index into a vector representation. This is the role of **the embedding layer**.

A first way to represent a token $x_t$ is through a one-hot vector. The one-hot vector associated with $x_t$ is denoted by $e_{x_t}$ and belongs to $\mathbb{R}^V$:

$$
e_{x_t} = (0,\dots,0,1,0,\dots,0) \in \mathbb{R}^V.
$$

The only nonzero coordinate is the one corresponding to the index $x_t$. This representation is useful because it identifies each token uniquely. However, one-hot vectors are high-dimensional, sparse, and do not encode semantic similarity between tokens. To obtain a dense and learnable representation, we introduce an embedding matrix:

$$
E \in \mathbb{R}^{V \times d}.
$$

Each row of $E$ corresponds to one token in the vocabulary and $d$ is the embedding dimension. The vector representation of token $x_t$ is obtained by selecting the row of $E$ corresponding to $x_t$:

$$
z_t = e_{x_t}^{\top}E \in \mathbb{R}^d.
$$

Equivalently, $z_t$ is simply the row of $E$ associated with the token index $x_t$.

At this point, $z_t$ contains information about the identity of the token. However, it does not contain information about the position of the token in the sequence. This is an important issue for Transformers. Unlike recurrent neural networks, Transformers process all tokens in parallel, so they do not naturally know whether a token appears at the beginning, in the middle, or at the end of the sentence. For example, the sentences `"the cat chased the dog"` and `"the dog chased the cat"` contain almost the same tokens, but their meanings are different because the order of the tokens is different. Therefore, we need to add positional information to the token embeddings.

To do this, we introduce a positional embedding vector $p_t \in \mathbb{R}^d$ for each position $t$. The initial representation of token $x_t$ is then obtained by adding its token embedding and its positional embedding:

$$
h_t^{(0)} = z_t + p_t,  \qquad h_t^{(0)}\in \mathbb{R}^d
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
\in \mathbb{R}^{T \times d}, \qquad P =
\begin{pmatrix}
p_1 \\
p_2 \\
\vdots \\
p_T
\end{pmatrix}
\in \mathbb{R}^{T \times d}
$$

then the input matrix to the Transformer is:

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

At this stage, the text is represented by a matrix:

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

where each row $h_t \in \mathbb{R}^d$ is the embedding representation of the token $x_t$. For simplicity, in this section we write \(H\) for the current hidden representation, which may be the initial embedding matrix \(H^{(0)}\) or the output of a previous Transformer block.

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

Thus, we define the attention score between token $t$ and token $s$ by:

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

In matrix form, all attention scores can be computed at once by taking the product between the query matrix and the transpose of the key matrix:

$$
A
=
\frac{QK^\top}{\sqrt{d_k}}
\in \mathbb{R}^{T \times T}.
$$

The entry $A_{t,s}$ measures how much token $x_t$ is related to token $x_s$ before normalization. In other words, it is the raw attention score between the query of token $x_t$ and the key of token $x_s$.

We then apply a softmax row by row to transform these scores into attention weights. Finally, these weights are used to take a weighted average of the value vectors:

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

The important point is that each row $\widetilde{h}_t$ of $\widetilde{H}$ is a new representation of token $x_t$. It is built using information from the sequence, with more weight given to the tokens that are more relevant to $x_t$.

So far, this attention mechanism allows every token to attend to every other token in the sequence. This is useful when the whole sequence is available, but it creates a problem for autoregressive language modeling. Indeed, in next-token prediction, the model should predict the future from the past, not by looking directly at the future. When predicting token $x_{t+1}$ from $x_1,\dots,x_t$, the representation at position $t$ should not use information from positions after $t$.

This is why we need **causal self-attention**.

## Causal Self-Attention

Causal self-attention is the same mechanism as self-attention, but with one important constraint: when computing the representation of token $x_t$, the model can only attend to tokens at positions $s \leq t$.

To impose this constraint directly inside the attention computation, we modify the attention scores before applying the softmax. Recall that the attention score matrix is:

$$
A
=
\frac{QK^\top}{\sqrt{d_k}}.
$$

We introduce a mask matrix $M \in \mathbb{R}^{T \times T}$ defined by:

$$
M_{t,s}
=
\begin{cases}
0 & \text{if } s \leq t,\\
-\infty & \text{if } s > t.
\end{cases}
$$

After applying the softmax, the entries corresponding to future positions receive attention weight zero. Therefore, the causal self-attention formula becomes:

$$
\widetilde{H}
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)
V.
$$

Thus, the representation of token $x_t$ is built only from $x_1,\dots,x_t$.

## Multi-Head Attention

So far, we described self-attention as if the model had only one attention mechanism. In practice, Transformers use several attention mechanisms in parallel. This is called **multi-head attention**. The intuition is that one attention head may focus on one type of relationship between tokens, while another head may focus on a different type of relationship. For example, one head may learn to connect a pronoun to the noun it refers to, while another head may focus on local syntactic relations or long-range dependencies.

Instead of computing a single set of queries, keys, and values, we compute several of them. For head $j$, we define

$$
Q^{(j)} = HW_Q^{(j)}, \qquad
K^{(j)} = HW_K^{(j)}, \qquad
V^{(j)} = HW_V^{(j)}.
$$

Each head then computes its own attention output:

$$
O^{(j)}
=
\mathrm{softmax}
\left(
\frac{Q^{(j)}(K^{(j)})^\top}{\sqrt{d_k}}
\right)
V^{(j)}.
$$

If we have $m$ heads, we obtain $m$ different outputs:

$$
O^{(1)}, O^{(2)}, \dots, O^{(m)}.
$$

These outputs are then concatenated and projected back to dimension $d$:

$$
\mathrm{MultiHead}(H)
=
\mathrm{Concat}
\left(
O^{(1)},\dots,O^{(m)}
\right)
W_O.
$$

The role of $W_O$ is to project the concatenated heads back to dimension $d$. Intuitively, it also mixes the information coming from the different heads into a single representation. Therefore,

$$
\mathrm{MultiHead}(H) \in \mathbb{R}^{T \times d}.
$$

The main idea is that multi-head attention allows the model to look at the sequence from several perspectives at the same time. Each head can learn a different way of comparing tokens and extracting contextual information.

For the rest of the Transformer block, we denote the output of multi-head attention by:

$$
\widetilde{H}
=
\mathrm{MultiHead}(H).
$$

## Residual Connection and Layer Normalization

The multi-head attention mechanism gives us a new contextual representation $\widetilde{H}$. However, in practice, Transformers do not simply replace $H$ by $\widetilde{H}$. Instead, they use two important operations: a **residual connection** and **layer normalization**.

### Why Do We Need a Residual Connection?

The residual connection consists of adding the input representation back to the output of the attention layer:

$$
H + \widetilde{H}.
$$

The idea is simple but very important. The attention layer computes new information from the context, but we do not want the model to lose the previous representation of the tokens. By adding $H$ back, the model keeps direct access to what it already knew before attention.

This is especially useful in deep networks. Without residual connections, information could gradually be distorted or lost as it passes through many layers. Residual connections create a direct path for information to flow from earlier layers to later layers. They also make optimization easier: instead of forcing each layer to learn a completely new representation from scratch, the layer only needs to learn a correction, or an update, to the previous representation.

### Why Do We Need Layer Normalization?

After the residual connection, we obtain

$$
H_{\mathrm{res}} = H + \widetilde{H}.
$$

This matrix contains one vector representation for each token. Since a Transformer is made of many successive blocks, these representations are transformed again and again by attention layers, MLPs, and residual additions. As a result, their numerical scale can drift during training. This can make optimization harder: the next layer has to process vectors whose coordinates may become too large, too small, or too variable from one layer to another. Layer normalization helps by keeping each token representation on a more controlled scale.

The key idea is simple: for each token, we normalize its hidden vector across the feature dimension. So if

$$
h_t =
(h_{t,1},\dots,h_{t,d})
\in \mathbb{R}^d,
$$

layer normalization first computes the mean and variance of the coordinates of this vector:

$$
\mu_t
=
\frac{1}{d}
\sum_{i=1}^{d} h_{t,i},
\qquad
\sigma_t^2
=
\frac{1}{d}
\sum_{i=1}^{d}
(h_{t,i}-\mu_t)^2.
$$

Then each coordinate is centered and rescaled:

$$
\widehat{h}_{t,i}
=
\frac{h_{t,i}-\mu_t}{\sqrt{\sigma_t^2+\varepsilon}}.
$$

The small constant $\varepsilon$ is added only for numerical stability, to avoid division by zero.

At this point, the normalized vector has a more stable scale. However, we do not want to force all hidden representations to always have the exact same scale and center. The model should still be able to choose the scale and shift that are useful for the task. For this reason, layer normalization introduces two learnable parameters:

$$
\gamma \in \mathbb{R}^d
\qquad
\text{and}
\qquad
\beta \in \mathbb{R}^d.
$$

The final layer-normalized vector is

$$
\mathrm{LayerNorm}(h_t)
=
\gamma
\odot
\frac{h_t-\mu_t}{\sqrt{\sigma_t^2+\varepsilon}}
+
\beta.
$$

Here, $\odot$ denotes coordinate-wise multiplication.

The important point is that layer normalization is applied independently to each token representation. It normalizes the coordinates of a single hidden vector $h_t$, not the whole sequence at once.

Intuitively, layer normalization says: before sending this representation to the next part of the Transformer, let us recenter and rescale it so that it is easier to process, while still allowing the model to learn the most useful scale through $\gamma$ and $\beta$.

After the residual connection and layer normalization, the output of the attention sublayer is

$$
H_{\mathrm{attn}}
=
\mathrm{LayerNorm}
\left(
H + \widetilde{H}
\right).
$$

## The MLP Block

After the attention sublayer, we obtain a contextual representation:

$$
H_{\mathrm{attn}} \in \mathbb{R}^{T \times d}.
$$

Each row of this matrix now contains information from the whole sequence. However, attention mainly allows tokens to exchange information with each other. Once this contextual information has been collected, we still need to process it and transform it in **a richer way**. This is the role of the MLP block.

The MLP is applied independently to each token representation. If

$$
H_{\mathrm{attn}}
=
\begin{pmatrix}
h^{\mathrm{attn}}_1 \\
h^{\mathrm{attn}}_2 \\
\vdots \\
h^{\mathrm{attn}}_T
\end{pmatrix},
$$

then the same MLP is applied to every row $h^{\mathrm{attn}}_t$. A standard Transformer MLP has the form:

$$
\mathrm{MLP}(h)
=
W_2 \, \phi(W_1 h + b_1) + b_2.
$$

Here,

$$
W_1 \in \mathbb{R}^{d_{\mathrm{ff}} \times d},
\qquad
b_1 \in \mathbb{R}^{d_{\mathrm{ff}}},
$$

and

$$
W_2 \in \mathbb{R}^{d \times d_{\mathrm{ff}}},
\qquad
b_2 \in \mathbb{R}^{d}.
$$

The dimension $d_{\mathrm{ff}}$ is usually larger than $d$. For example, in many Transformer architectures, one takes $d_{\mathrm{ff}} = 4d$. So the MLP first expands the representation to a higher-dimensional space, applies a nonlinearity, and then projects it back to dimension $d$.

The function $\phi$ is a nonlinear activation function, such as ReLU or GELU. This nonlinearity is important because without it, the MLP would only be a linear transformation. So for each token $x_t$, we compute

$$
m_t
=
\mathrm{MLP}(h_t^{\mathrm{attn}}).
$$

In matrix form, this can be written as

$$
M
=
\mathrm{MLP}(H_{\mathrm{attn}}),
$$

where the MLP is applied row by row.

Intuitively, attention answers the question:

> Which tokens should I look at?

The MLP answers a different question:

> Once I have collected contextual information, how should I transform it?

So attention mixes information across tokens, while the MLP enriches the representation of each token independently.

As before, Transformers add a residual connection and layer normalization around the MLP block. Therefore, the output of one Transformer block can be written as

$$
H_{\mathrm{out}}
=
\mathrm{LayerNorm}
\left(
H_{\mathrm{attn}} + \mathrm{MLP}(H_{\mathrm{attn}})
\right).
$$

This gives us a new matrix

$$
H_{\mathrm{out}} \in \mathbb{R}^{T \times d},
$$

where each row is a richer contextual representation of the corresponding token.

If we stack several Transformer blocks, this process is repeated. Each layer refines the representation by first allowing tokens to communicate through attention, and then transforming each token representation through an MLP.

The whole point of the Transformer is therefore **to build good contextual representations of the sequence**.


# Next-Token Prediction and Training Objective

Until now, we have described how a Transformer takes a sequence of tokens and builds contextual representations. The main idea is that each token representation is progressively updated through self-attention, residual connections, layer normalization, and MLP blocks.

But why do we build these representations?

For language modeling, the goal is to predict the next token.

Assume we have a sequence of tokens

$$
x_1, x_2, \dots, x_T,
$$

where each token belongs to a vocabulary of size $V$. The autoregressive modeling objective is to learn, for each position $t$, the conditional probability

$$
p_\theta(x_{t+1} \mid x_1,\dots,x_t).
$$

Intuitively, the model tries to answer the following question:

> Given the previous tokens $x_1,\dots,x_t$, what is the probability distribution of the next token $x_{t+1}$?

This is why the final hidden representations are important. After passing the sequence through the Transformer, we obtain contextual representations:

$$
h_1, h_2, \dots, h_T.
$$

The vector $h_t$ summarizes the information available at position $t$. In an autoregressive Transformer, it is used to predict the next token $x_{t+1}$.

To do this, we map $h_t$ to a vector of logits over the vocabulary:

$$
\ell_t = W_{\mathrm{out}}h_t + b_{\mathrm{out}},
$$

where

$$
\ell_t \in \mathbb{R}^{V}.
$$

Each coordinate $\ell_{t,i}$ is a score associated with token $i$ in the vocabulary. These scores are not probabilities yet: they can be negative, and they do not sum to one. To transform them into probabilities, we apply the softmax function:

$$
p_\theta(x_{t+1}=i \mid x_1,\dots,x_t)
=
\frac{\exp(\ell_{t,i})}
{\sum_{j=1}^{V}\exp(\ell_{t,j})}.
$$

Therefore, for each position $t$, the Transformer outputs a probability distribution over the whole vocabulary:

$$
p_\theta(\cdot \mid x_1,\dots,x_t) \in \Delta^{V-1},
$$

where $\Delta^{V-1}$ is the probability simplex over the vocabulary:

$$
\Delta^{V-1}
=
\left\{
p \in \mathbb{R}^{V}
:
p_i \geq 0
\quad \text{and} \quad
\sum_{i=1}^{V} p_i = 1
\right\}.
$$

So mathematically, we can think of the Transformer as a parameterized map

$$
(x_1,\dots,x_t)
\longmapsto
p_\theta(\cdot \mid x_1,\dots,x_t)
\in \Delta^{V-1}.
$$

The output is not directly a single token. It is first a probability distribution over all possible next tokens.

## Factorization of the Sequence Probability

For a full sequence $x_1,\dots,x_T$, an autoregressive language model factorizes the joint probability as

$$
p_\theta(x_1,\dots,x_T)
=
\prod_{t=1}^{T}
p_\theta(x_t \mid x_1,\dots,x_{t-1}) =
\prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t}). 
$$

This means that the model assigns a probability to the whole sequence by predicting each token from the tokens that came before it. So next-token prediction is not only a generation rule. It also defines a probability model over entire sequences.

## Training Objective

During training, we already know the true next token. For each position $t$, the model predicts a probability distribution

$$
p_\theta(\cdot \mid x_{<t}),
$$

and we want the model to assign high probability to the true token $x_t$. Therefore, we maximize the likelihood of the training sequence:

$$
\prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t}).
$$

Equivalently, we minimize the negative log-likelihood:

$$
\mathcal{L}(\theta)
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t}).
$$

This is the usual cross-entropy loss used to train language models.

Intuitively, if the true token is $x_t$, then the loss penalizes the model when it assigns a small probability to $x_t$. Training pushes the model to put more probability mass on the correct next token.

## What Are the Parameters $\theta$?

The notation $\theta$ denotes all the trainable parameters of the Transformer. More precisely, it includes:

- the token embedding matrix $E$;
- the positional embeddings, if they are learned;
- the attention matrices $W_Q$, $W_K$, and $W_V$ in each layer;
- the output projection matrices used after attention;
- the MLP weights and biases in each Transformer block;
- the layer normalization parameters $\gamma$ and $\beta$;
- the final output matrix $W_{\mathrm{out}}$ and bias $b_{\mathrm{out}}$.

So $\theta$ represents everything the model learns during training.

The role of training is to adjust these parameters so that the probability distribution

$$
p_\theta(\cdot \mid x_1,\dots,x_t)
$$

becomes good at predicting the next token.

This is the central idea behind autoregressive language modeling: the Transformer builds a contextual representation of the previous tokens, then uses this representation to produce a probability distribution over the next token.

The overall pipeline can be summarized as follows:

<div style="text-align:center; line-height:2; font-size:1.05em; margin: 2em 0;">
  <div><strong>Tokens</strong></div>
  <div>↓</div>
  <div><strong>Token embeddings + positional embeddings</strong></div>
  <div>↓</div>
  <div><strong>Causal self-attention</strong></div>
  <div>↓</div>
  <div><strong>Residual connection + layer normalization</strong></div>
  <div>↓</div>
  <div><strong>MLP</strong></div>
  <div>↓</div>
  <div><strong>Residual connection + layer normalization</strong></div>
  <div>↓</div>
  <div><strong>Logits over vocabulary</strong></div>
  <div>↓</div>
  <div><strong>Softmax</strong></div>
  <div>↓</div>
  <div><strong>Next-token distribution</strong></div>
</div>

# What comes next?

In this post, we described the transformer architecture and the next-token prediction objective used to train language models.

However, pretraining alone does not fully explain why modern LLMs behave like assistants. A pretrained model learns to continue text; an assistant must follow instructions, answer helpfully, refuse unsafe requests, and adapt to user intent.

This gap is addressed by post-training methods such as supervised fine-tuning, preference optimization, RLHF, and safety tuning. These will be the topic of the next post.
