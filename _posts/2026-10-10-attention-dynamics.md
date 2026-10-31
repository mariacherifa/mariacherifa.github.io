---
layout: post
title: "Attention Dynamics and Differential Inclusions"
date: 2026-10-15
tags: transformers attention deep-learning
categories: machine-learning
math: true
mermaid: true
---

# Understanding Sharp Attention Dynamics: From ODEs to Differential Inclusions

## Introduction

Attention is one of the central mechanisms behind Transformers. Given a collection of token representations, attention allows each token to assign different weights to other tokens and aggregate their information.

Mathematically, these weights are computed using a softmax function. An important parameter in this function is the **temperature** $\tau>0$, which controls how concentrated the attention weights are.

When $\tau$ is large, attention distributes its weights relatively smoothly across tokens. When $\tau$ becomes small, attention becomes increasingly selective, concentrating on the tokens with the largest attention scores.

In the limiting regime

$$
\tau\to 0,
$$

softmax attention approaches a hard selection mechanism, commonly called **hardmax attention**.

This raises an interesting mathematical question: how can we describe the evolution of token representations when attention becomes increasingly sharp?

To understand this, we can view attention through the lens of dynamical systems.

# Modeling Attention Dynamics

Consider a sequence of $n$ tokens represented by

$$
X=(x_1,\ldots,x_n),\qquad x_i\in\mathbb R^d.
$$

For simplicity, assume that the query, key, and value matrices $Q,K,V$ are fixed.

The attention score between tokens $i$ and $j$ is

$$
s_{ij}(X)=\frac{\langle Qx_i,Kx_j\rangle}{\sqrt{d_k}}.
$$

The corresponding attention weight is

$$
a_{ij}^{\tau}(X)
=
\frac{\exp(s_{ij}(X)/\tau)}
{\sum_{\ell=1}^{n}\exp(s_{i\ell}(X)/\tau)}.
$$

The attention output for token $i$ is then

$$
F_{\tau,i}(X)
=
\sum_{j=1}^{n}a_{ij}^{\tau}(X)Vx_j.
$$

Collecting these outputs gives a vector field

$$
F_\tau(X)=\left(F_{\tau,1}(X),\ldots,F_{\tau,n}(X)\right).
$$

Now consider a simplified residual attention network:

$$
X_{k+1}^{h,\tau}
=
X_k^{h,\tau}
+
hF_\tau(X_k^{h,\tau}),
$$

where $h>0$ controls the size of each update.

This equation describes how token representations evolve from one layer to the next.

An interesting observation is that this update has the structure of an **Euler discretization of a differential equation**.

# From Transformers to Differential Equations

Suppose that the temperature $\tau>0$ is fixed.

As the step size $h$ becomes smaller, the number of layers needed to cover a fixed depth interval $[0,T]$ increases:

$$
L\approx \frac{T}{h}.
$$

Under suitable regularity assumptions, as

$$
h\to 0,
$$

the discrete representation dynamics converge to the solution of the ODE

$$
\boxed{
\frac{dX^\tau(t)}{dt}
=
F_\tau(X^\tau(t)).
}
$$

This gives a continuous-depth interpretation of the Transformer.

Instead of viewing a Transformer as a sequence of discrete transformations, we can think of the token representations as trajectories evolving continuously over depth.

For fixed positive temperature, the attention vector field is smooth, and standard ODE theory provides a natural framework for studying its dynamics.

But what happens when attention becomes increasingly sharp?

# What Happens When Attention Becomes Sharp?

Recall that the attention weights are defined by

$$
a_{ij}^{\tau}(X)
=
\frac{\exp(s_{ij}(X)/\tau)}
{\sum_{\ell=1}^{n}\exp(s_{i\ell}(X)/\tau)}.
$$

For a fixed representation $X$, as $\tau\to0$, the attention weights concentrate on the indices maximizing the attention score.

Let

$$
M_i(X)=\arg\max_j s_{ij}(X).
$$

If there is a unique maximizer $j^*$, then

$$
a_{ij}^{\tau}(X)\longrightarrow
\begin{cases}
1,&j=j^*,\\
0,&j\neq j^*.
\end{cases}
$$

Consequently,

$$
F_{\tau,i}(X)\longrightarrow Vx_{j^*}.
$$

The attention mechanism no longer averages information smoothly across tokens. Instead, it selects the value associated with the highest score.

However, an interesting difficulty appears when several tokens have the same maximal score.

**What happens when two or more tokens are equally relevant according to the attention mechanism?**

At an exact tie, softmax distributes its limiting weight uniformly among the maximizers. But an arbitrarily small perturbation of the representation can change which token is selected.

To see this, consider two scores

$$
s_1(z)=z,\qquad s_2(z)=0.
$$

The softmax weight associated with the first score is

$$
a_1^\tau(z)=\frac{e^{z/\tau}}{e^{z/\tau}+1}.
$$

As $\tau\to0$,

$$
a_1^\tau(z)\longrightarrow
\begin{cases}
1,&z>0,\\
\frac12,&z=0,\\
0,&z<0.
\end{cases}
$$

Thus, the limiting attention mechanism switches abruptly at $z=0$.

This illustrates an important phenomenon: **the limiting dynamics may be discontinuous, even though softmax attention is smooth for every positive temperature.**

# Why Differential Inclusions Appear

Ordinary differential equations describe the evolution of a system through a single-valued vector field:

$$
\dot X(t)=F(X(t)).
$$

At every state $X$, the vector field determines a velocity.

But when attention becomes sharp, the limiting vector field may become discontinuous at configurations where multiple tokens maximize the attention score.

This is where differential inclusions become relevant.

A differential inclusion generalizes an ordinary differential equation by allowing the velocity to belong to a set:

$$
\boxed{
\dot X(t)\in G(X(t)).
}
$$

Here,

$$
G:\mathbb R^{n\times d}\rightrightarrows\mathbb R^{n\times d}
$$

is a set-valued map.

Rather than assigning a single velocity to each representation, it associates a set of possible velocities.

A classical way to describe discontinuous dynamics is to consider convex combinations of nearby limiting velocities, as in the theory of Filippov differential inclusions.

This provides a mathematical framework for describing systems whose dynamics involve abrupt changes or switching mechanisms.

In the context of sharp attention, the transition from smoothly weighted combinations to hard selection makes this viewpoint particularly natural. The broader connection is therefore between :

$$
\boxed{
\text{Attention}
\quad\longleftrightarrow\quad
\text{Dynamical systems}
\quad\longleftrightarrow\quad
\text{Differential inclusions}.
}
$$

This perspective offers a different way to think about attention: not only as a mechanism that computes weighted representations, but also as a dynamical system whose mathematical structure changes when attention becomes increasingly selective.
