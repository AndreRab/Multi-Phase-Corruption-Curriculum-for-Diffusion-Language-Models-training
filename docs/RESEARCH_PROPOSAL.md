# Overview of Popular Areas in Diffusion Language Models

## How Diffusion Language Models Work in General

A diffusion language model usually keeps the main Transformer backbone, but changes the **input representation**, the **output prediction target**, and the **generation procedure**.

In a standard autoregressive language model:

- the input is a left-to-right prefix,
- the model uses a causal mask,
- and the output head predicts the **next token**.

In a diffusion language model:

- the input is usually a **corrupted full sequence** rather than a clean prefix,
- the model typically uses **bidirectional attention** instead of a causal mask,
- an additional **timestep / noise-step embedding** is provided,
- and the output head predicts either:
  - the original clean tokens,
  - a less noisy token distribution,
  - or a transition toward the next intermediate state.

So the main architectural idea is not to rebuild the whole model from scratch, but to keep the Transformer core and introduce **new input and output layers or prediction heads** adapted to denoising. Instead of generating text strictly token by token, the model iteratively improves a noisy sequence over several refinement steps.

---

# Main Directions in Diffusion Language Models

## 1) Masked Diffusion Language Models

This is the most intuitive family. A full sequence is given as input, some tokens are corrupted with `[MASK]`, and the model iteratively restores the corrupted positions. This is currently one of the most widely studied and practically stable branches among discrete diffusion approaches for language. Representative strong models include **MDLM / Simple and Effective Masked Diffusion Language Models**, **LLaDA**, and **Dream 7B**.

## 2) Discrete Diffusion over Tokens

This is a more general class. Tokens are not corrupted only by `[MASK]`; instead, they can evolve through more general discrete transition processes over the vocabulary. This includes **absorbing-state diffusion** (masking) as one special case, but also **uniform-state diffusion** and **interpolating discrete diffusion**. This area is especially promising because it gives much more room to experiment with the corruption process and the sampler. Strong reference points include **Duo / The Diffusion Duality**, **Eso-LMs**, **GIDD**, and scaling studies comparing masked, uniform, and interpolating diffusion under matched budgets.

## 3) Flow Matching / Discrete Flow Matching for Text

Here the model is trained not only to recover tokens, but to predict the transition step between intermediate states. The main motivation is **few-step generation**: reaching good quality with very few refinement iterations. A strong recent example is **FS-DFM**, where the number of generation steps is made an explicit part of training, and the model is optimized to remain stable even with only a small number of steps.

## 4) Continuous / Embedding-Space Diffusion for Text

In this family, noise and denoising are defined not over discrete token IDs, but over continuous embeddings or hidden states. This is mathematically attractive and fits naturally with ODE/flow-style formulations, but the main difficulty is mapping the denoised continuous state back into high-quality discrete tokens. This direction has become active again, with works such as **LangFlow** and **CoDAR** showing that continuous diffusion for text is becoming more competitive.

---

# Typical Pipeline for Discrete Diffusion over Tokens

## 1. Data Preparation

Start from a clean text sequence:

`x0 = [x1, x2, ..., xn]`

## 2. Forward Corruption

Create a corrupted sequence `xt` using a chosen corruption process:

- replace some tokens with `[MASK]` in an absorbing-state setup,
- replace tokens with random tokens in a uniform-state setup,
- or use an interpolating family between these extremes.

## 3. Denoiser / Transformer

A bidirectional Transformer receives:

- the corrupted sequence `xt`,
- a timestep embedding `t`,
- optionally condition/target indicators.

The model predicts either:

- the clean token distribution,
- or a transition / score toward a less noisy state.

## 4. Training Objective

The objective is usually a discrete denoising objective:

- cross-entropy on corrupted positions,
- or an equivalent variational / diffusion / flow-style objective.

For masked diffusion in particular, the Rao-Blackwellized masked cross-entropy objective from **MDLM** is a strong reference.

## 5. Sampling

At inference time, generation moves from a noisy state to a cleaner state:

- ancestral decoding,
- block sampling,
- consistency / few-step distillation,
- refinement-based decoding.

Strong recent references include:

- **Duo** for few-step uniform diffusion,
- **Eso-LM** for efficient block / KV-cached decoding,
- **FS-DFM** for few-step flow-style generation.

---

# Research Goal

The goal of this work is to investigate whether the corruption strategy itself can improve diffusion language model training. Instead of proposing a new Transformer architecture, the focus is on designing and evaluating a new training-time corruption curriculum for discrete diffusion language models.

---

# Proposed Research Direction: Multi-Phase Corruption Curriculum

For the first experimental stage, I would start with a relatively small model, such as **GPT-2 small**, and adapt it toward a discrete diffusion-style denoising setup instead of immediately training a large model. The goal is not to compete with large-scale diffusion LMs at the beginning, but to test whether the proposed corruption pipeline gives a measurable advantage over modern classical baselines.

The initial comparison should include several strong baseline corruption strategies:

- standard **mask-only diffusion**,
- **uniform random token replacement**,
- an **interpolating discrete diffusion** setup between masking and random replacement,
- and, if possible, an MDLM-style masked diffusion baseline.

The proposed method is a **multi-phase corruption curriculum**. Instead of training the model with one fixed corruption type from the beginning, the training process gradually changes the kind and difficulty of corruption:

1. **Semantic-near replacement phase**  
   Tokens are replaced with semantically similar but still incorrect alternatives. This forces the model to pay attention to grammar, context, and exact lexical choice rather than simply detecting obviously wrong tokens.

2. **Masking phase**  
   Tokens are replaced with `[MASK]`, so the model learns the classic denoising task of reconstructing missing information from context.

3. **Random replacement phase**  
   Tokens are replaced with random vocabulary tokens, making the task harder because the model must identify and correct visible but incorrect tokens.

4. **Progressive random corruption phase**  
   The percentage of randomly replaced tokens is gradually increased, so the model is exposed to increasingly difficult denoising conditions.

The main hypothesis is that starting from semantically close corruption teaches the model fine-grained contextual correction first, while later masking and random replacement phases make it robust to stronger and less realistic noise. In other words, the corruption process moves from **almost correct text** to **hidden text** and finally to **heavily corrupted text**.

This direction can be described as a **semantic-distance-controlled multi-phase corruption curriculum** for diffusion language models. The expected contribution is not a new Transformer backbone, but a new training-time corruption policy that can be compared directly against standard masked and random discrete diffusion approaches.

---

# Method Comparison and Open Questions

The comparison should focus on **corruption methods**, not on comparing different published models directly. Large models such as LLaDA, Dream 7B, MDLM, or Duo should be used mainly as references for existing approaches. A fair experiment should use the **same backbone**, for example GPT-2 small or the same small Transformer, and change only the corruption strategy.

---

## Method Comparison

| Method | Reference papers | Short description | Controlled baseline |
|---|---|---|---|
| **Mask-only corruption** | DiffusionBERT: https://arxiv.org/abs/2211.15029<br>MDLM: https://arxiv.org/abs/2406.07524<br>LLaDA: https://arxiv.org/abs/2502.09992 | Tokens are replaced with `[MASK]`, and the model reconstructs the original clean tokens. | Train the same backbone with mask-only corruption. |
| **Uniform random replacement** | D3PM: https://arxiv.org/abs/2107.03006<br>SEDD: https://arxiv.org/abs/2310.16834<br>Duo: https://arxiv.org/abs/2506.10892 | Tokens are replaced with random vocabulary tokens instead of `[MASK]`. | Train the same backbone with random token replacement. |
| **Static mask + random mixture** | GIDD: https://arxiv.org/abs/2503.04482<br>Eso-LM: https://arxiv.org/abs/2506.01928 | Some tokens are replaced with `[MASK]`, others with random tokens, using a fixed ratio. | Train the same backbone with a fixed mask/random mixture. |
| **Token-dependent noise** | DiffusionBERT: https://arxiv.org/abs/2211.15029<br>Dream 7B: https://arxiv.org/abs/2508.15487 | The corruption probability depends on token importance, frequency, context, or confidence. | Train the same backbone with a simplified token-aware corruption rule. |
| **Semantic-near replacement only** | Related: D3PM: https://arxiv.org/abs/2107.03006<br>Related: GIDD: https://arxiv.org/abs/2503.04482 | Tokens are replaced with semantically similar but incorrect alternatives. | Train the same backbone only with semantic-near replacement. |
| **Progressive random curriculum** | Related: Duo: https://arxiv.org/abs/2506.10892 | The amount of random replacement increases during training. | Train the same backbone with increasing random corruption ratio. |
| **Proposed method** | New proposed direction | Training moves through semantic-near replacement, then masking, then random replacement, then stronger random corruption. | Train the same backbone with the full semantic-near -> mask -> random curriculum. |

---

## Open Questions

1. **Which baselines should be reproduced under the same training budget, and which published results should be used only as literature context?**

2. **Which backbone should be used for the first fair experiment: GPT-2 small adapted to diffusion-style denoising, or a small MDLM-style bidirectional Transformer?**

3. **Which dataset should be used first: WikiText-103 for fast proof of concept, or OpenWebText / LM1B for a stronger comparison?**

4. **How should the model switch between corruption modes during training: hard phase boundaries, smooth probability schedule, or mixed curriculum from the beginning?**

5. **Should the semantic-near phase use nearest neighbors from static embeddings, contextual embeddings, or a separate small sentence-transformer model?**

6. **How large should the first experiment be before scaling to larger models?**