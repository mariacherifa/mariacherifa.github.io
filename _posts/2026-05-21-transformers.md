---
layout: post
title: "Understanding Transformers with math and intuitions"
date: 2026-05-21
tags: transformers attention deep-learning
categories: machine-learning
---

# Introduction

The goal of this blog post is to understand the main components of Transformers, while providing both mathematical insights and intuition.

# From Text to Tokens

To treat text as a mathematical object, we first need to define two important notions: a **vocabulary** and a **sequence of tokens**.

A vocabulary is a finite set of symbols used to represent text. These symbols can be characters, subwords, words, numbers, punctuation marks, or special tokens. We denote the vocabulary by

$$
\mathcal{V} = \{v_1, \dots, v_V\},
$$

where $ V = |\mathcal{V}|$ is the size of the vocabulary.

We define an indexing function,  $\mathrm{idx} : \mathcal{V} \to \{1,\dots,V\}$ by, 

$$
\mathrm{idx}(v_i)=i.
$$

This function maps each element in the vocabulary to its integer index. Conversely, we define a decoding function $\mathrm{tok} : \{1,\dots,V\} \to \mathcal{V}$ by

$$
\mathrm{tok}(i)=v_i.
$$

Once a vocabulary and an indexing function are fixed, a text can be represented as a discrete sequence of token indices:

$$
(x_1, \dots, x_T) \in \{1,\dots,V\}^T.
$$

Here, \(T\) denotes the length of the tokenized text, and each integer \(x_t\) corresponds to the token

$$
\mathrm{tok}(x_t)=v_{x_t}.
$$

In other words, \(x_t\) is not the token itself, but the index of a token in the vocabulary.
