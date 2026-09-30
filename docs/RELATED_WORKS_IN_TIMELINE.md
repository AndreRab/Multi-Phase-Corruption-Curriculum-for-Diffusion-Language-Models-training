# Diffusion Language Models: Development Timeline and Research Positioning

This README organizes the development of diffusion language modeling as a chronological story rather than as a flat list of related papers. The goal is to track how the field moved from defining discrete diffusion formally, through simpler masked-denoising objectives and pretrained-model adaptation, to large-scale diffusion LLMs, curriculum strategies, self-correction, and modern scaling studies.

For consistency, **every work is described using the same template**:

- **Initialization** — pretrained model or trained from scratch.
- **Corruption** — how clean text is noised.
- **Training** — what input/target the model sees and what objective is optimized.
- **Denoising / inference** — how generation or reconstruction proceeds.
- **Key idea / feature** — the main technical contribution in a few words.
- **Why it matters in the timeline** — what question this paper answered and what it left open.

> Dates below refer to the first public paper/preprint unless otherwise stated.

---

## 2021 — D3PM: Discrete diffusion becomes a general framework

### Structured Denoising Diffusion Models in Discrete State-Spaces — Austin et al. (2021)

Paper: [arXiv:2107.03006](https://arxiv.org/abs/2107.03006)

- **Initialization:** Trained from scratch. The text denoiser is not initialized from a pretrained language model.
- **Corruption:** Explicit categorical transition kernels \(Q_t\). The paper studies several families, including **uniform replacement**, **absorbing-state masking**, nearest-neighbor transitions in embedding space, and discretized-Gaussian-style transitions for ordered states.
- **Training:** Sample a clean \(x_0\), choose a diffusion timestep \(t\), sample \(x_t\) from the known forward process, and train a model using the diffusion variational objective plus an auxiliary clean-token cross-entropy term. The explicit forward kernels make the diffusion posterior tractable.
- **Denoising / inference:** Start from the terminal noisy state and iteratively apply the learned reverse diffusion process \(x_t \rightarrow x_{t-1}\) until reaching \(x_0\).
- **Key idea / feature:** **The corruption kernel is a modeling decision.** Different discrete transition structures produce substantially different results.
- **Why it matters in the timeline:** D3PM establishes the mathematical basis for discrete diffusion and shows that **how text is corrupted matters**. It provides direct precedents for Random/Uniform, Mask/Absorbing, and Similar/Nearest-Neighbor corruption families.

### What changed after D3PM?

The field now had a formal answer to:

> **How can diffusion be defined directly over discrete tokens?**

The next questions became: should text diffusion remain discrete, can linguistic structure improve corruption, and do we really need to train the denoiser from scratch?

---

## 2022 — Diffusion-LM: A parallel continuous-space branch

### Diffusion-LM Improves Controllable Text Generation — Li et al. (2022)

Paper: [arXiv:2205.14217](https://arxiv.org/abs/2205.14217)

- **Initialization:** Trained as a new diffusion LM rather than initialized from a pretrained BERT/GPT checkpoint. It uses a BERT-base-style Transformer architecture, but the diffusion model itself is trained for the task.
- **Corruption:** **Continuous Gaussian noise** is applied to word embeddings rather than corrupting discrete token IDs directly. The paper introduces a text-oriented square-root noise schedule.
- **Training:** A Transformer receives noisy continuous representations \((x_t,t)\) and learns to reconstruct cleaner representations; the paper also studies direct \(x_0\)-prediction to reduce rounding errors.
- **Denoising / inference:** Begin from Gaussian noise and iteratively denoise continuous latent word vectors, finally mapping the resulting representations back to tokens. Gradient-based guidance can steer generation toward desired controls.
- **Key idea / feature:** **Move diffusion into embedding space** to retain the machinery of continuous Gaussian diffusion while generating text.
- **Why it matters in the timeline:** It creates the major alternative branch to discrete token diffusion. Later work increasingly returns to discrete/masked diffusion because it avoids embedding-to-token rounding and maps naturally to token prediction.

---

## 2023 — DiffusionBERT: Reuse pretrained language representations

### DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models — He et al. (ACL 2023)

Paper: [ACL Anthology](https://aclanthology.org/2023.acl-long.248/)

- **Initialization:** **Pretrained BERT** is used as initialization. This is a major shift from earlier text diffusion models trained as new denoisers.
- **Corruption:** Absorbing-state **masking**. Tokens progressively become `[MASK]`. The proposed **Spindle schedule** makes noise depend on both diffusion time and token information.
- **Training:** Corrupt clean text at a sampled diffusion level and adapt BERT to learn the reverse discrete diffusion process. The paper studies explicit timestep-conditioning designs and also a time-agnostic decoding variant.
- **Denoising / inference:** Iterative reverse denoising from highly masked text toward clean text. Multiple denoising passes progressively reconstruct the sequence.
- **Key idea / feature:** **A pretrained denoising LM can initialize a diffusion LM**, and the schedule determining which tokens are corrupted can materially affect performance.
- **Why it matters in the timeline:** It answers an important question: **diffusion language models do not necessarily need to learn linguistic representations from scratch**. However, BERT is already bidirectional and already pretrained with masking, so the adaptation problem is easier than converting an autoregressive GPT-style model.

---

## 2023 — Masked-Diffuse LM: Linguistically informed corruption

### A Cheaper and Better Diffusion Language Model with Soft-Masked Noise — Chen et al. (EMNLP 2023)

Paper: [ACL Anthology](https://aclanthology.org/2023.emnlp-main.289/)

- **Initialization:** The diffusion model is trained as a new model using a BERT-style architecture; the contribution is not pretrained-BERT initialization.
- **Corruption:** A **linguistically informed soft-masking forward process** instead of generic Gaussian noise.
- **Training:** The model directly predicts categorical token distributions with cross-entropy at diffusion steps, reducing the need to operate purely in a high-dimensional continuous latent space.
- **Denoising / inference:** Iterative diffusion-style reconstruction from linguistically corrupted inputs toward clean token sequences.
- **Key idea / feature:** **Corruption can exploit linguistic properties rather than treating every token identically.**
- **Why it matters in the timeline:** It strengthens the D3PM-era observation that corruption design is not merely implementation detail. The field starts asking not only *how much* noise to add, but *what kind of linguistic information* should be removed or preserved.

---

## 2024 — MDLM: Masked diffusion becomes much simpler

### Simple and Effective Masked Diffusion Language Models — Sahoo et al. (2024)

Paper: [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)

- **Initialization:** The paper trains masked diffusion language models directly rather than relying on pretrained autoregressive checkpoints.
- **Corruption:** **Masked / absorbing-state diffusion.** At a sampled noise level, a subset of tokens is replaced by `[MASK]`.
- **Training:** Sample a timestep/noise level, create one corrupted \(x_t\), run the model, and optimize a simplified Rao-Blackwellized objective that has the form of a mixture of classical masked-language-model cross-entropies predicting clean tokens.
- **Denoising / inference:** Iterative masked-token sampling progressively replaces masks with predicted tokens; efficient samplers can also generate text semi-autoregressively.
- **Key idea / feature:** **The training objective for masked discrete diffusion can be much simpler than early D3PM notation suggests.**
- **Why it matters in the timeline:** MDLM makes an important conceptual simplification: modern diffusion-LM training can look very close to ordinary denoising — corrupt once, predict \(x_0\), compute cross-entropy — while retaining a principled diffusion interpretation.

---

## 2024 — DiffuGPT / DiffuLLaMA: Convert pretrained autoregressive LMs

### Scaling Diffusion Language Models via Adaptation from Autoregressive Models — Gong et al. (2024)

Paper: [arXiv:2410.17891](https://arxiv.org/abs/2410.17891)

- **Initialization:** **Pretrained autoregressive GPT-2 and LLaMA checkpoints**, from 127M up to 7B parameters.
- **Corruption:** Masked discrete diffusion with a continuous-time masking process.
- **Training:** Continual pretraining converts the AR model into a denoiser. The method gradually adapts causal attention toward full-context attention, preserves/aligns the AR shift operation during conversion, and trains with a reweighted masked-token cross-entropy objective predicting clean tokens.
- **Denoising / inference:** Start from masked output positions and iteratively sample cleaner states. The sampler repeatedly predicts clean tokens and applies the reverse masking process; high-probability denoising can be used for conditional generation.
- **Key idea / feature:** **AR → diffusion conversion is practical.** Existing GPT/LLaMA weights can be reused instead of training a diffusion LM from scratch.
- **Why it matters in the timeline:** This removes another barrier: the field can inherit expensive linguistic knowledge from mature AR models. The question shifts from *can an AR model become a denoiser?* to **what is the best adaptation path?**

---

# 2025 — Diffusion language models become large models, not just proofs of concept

## 2025 — LLaDA: Large-scale masked diffusion from scratch

### Large Language Diffusion Models (LLaDA) — Nie et al. (2025)

Paper: [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)

- **Initialization:** **Trained from scratch.** LLaDA does not start from an autoregressive checkpoint.
- **Corruption:** Standard **mask-based forward diffusion**: clean tokens are independently replaced by `[MASK]` according to the sampled noise level.
- **Training:** Sample \(t\), sample masked \(x_t\), run one bidirectional Transformer pass, and predict the original clean tokens. Cross-entropy is computed on masked positions under the diffusion likelihood-bound formulation.
- **Denoising / inference:** Start with a masked answer region. At each iteration the model predicts all masked positions. A remasking strategy decides which predictions remain unresolved. In **low-confidence remasking**, high-confidence predictions are committed while low-confidence positions remain/remask for later iterations.
- **Key idea / feature:** **LLaDA-style decoding = predict all masked tokens → keep confident predictions → continue denoising uncertain positions.** The model scales masked diffusion to the multi-billion-parameter LLM regime.
- **Why it matters in the timeline:** It establishes confidence-guided iterative decoding as a practical modern masked-DLM pattern and shows that diffusion LMs can exhibit large-model capabilities when trained at scale.

---

## 2025 — GIDD: Masking alone cannot easily revise visible mistakes

### Generalized Interpolating Discrete Diffusion — von Rütte et al. (2025)

Paper: [arXiv:2503.04482](https://arxiv.org/abs/2503.04482)

- **Initialization:** Models are trained for the proposed discrete diffusion framework; pretrained AR conversion is not the central contribution.
- **Corruption:** Generalized interpolating diffusion that can combine **masking and uniform/random-state noise**.
- **Training:** Optimize a generalized diffusion ELBO under a more flexible noising process instead of committing to pure mask-only diffusion.
- **Denoising / inference:** Iterative discrete diffusion sampling. Because states are not restricted to irreversible mask→token transitions, the model can revise already visible token values.
- **Key idea / feature:** **Self-correction through editable visible tokens.** Mixing uniform noise with masking gives the model opportunities to change previous mistakes.
- **Why it matters in the timeline:** It exposes an important limitation of pure masked diffusion: once a token is committed, it can become difficult to revisit. This makes Random/Uniform corruption relevant not only as noise, but as a mechanism for teaching correction.

---

## 2025 — The Diffusion Duality (Duo): Uniform diffusion, self-correction, and curriculum

### The Diffusion Duality — Sahoo et al. (2025)

Paper: [arXiv:2506.10892](https://arxiv.org/abs/2506.10892)

- **Initialization:** The language diffusion models are trained within the proposed uniform-state diffusion framework; AR-checkpoint reuse is not the main contribution.
- **Corruption:** **Uniform-state discrete diffusion**, where tokens can transition among vocabulary states rather than only between token and `[MASK]`.
- **Training:** The paper connects uniform discrete diffusion to an underlying Gaussian diffusion and introduces a **Gaussian-guided curriculum** that reduces training variance and accelerates learning.
- **Denoising / inference:** Iterative uniform-state denoising with strong self-correction; the paper also introduces discrete consistency distillation for few-step generation.
- **Key idea / feature:** **Curriculum learning can improve diffusion-LM training, and uniform-state diffusion has useful self-correction behavior.**
- **Why it matters in the timeline:** Curriculum becomes an explicit part of diffusion language-model training. However, the curriculum operates *within one corruption family* rather than changing among semantically different families such as Similar → Mask → Random.

---

## 2025 — Seed Diffusion: Change the denoising task during training

### Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference — Song et al. (2025)

Paper: [arXiv:2508.02193](https://arxiv.org/abs/2508.02193)

- **Initialization:** Large-scale diffusion-model training; the paper does **not** present the method as conversion of a pretrained AR checkpoint.
- **Corruption:** A genuine **two-stage corruption curriculum**. For roughly the first 80% of diffusion training, it uses mask-based corruption. In the last 20%, it adds edit-based corruption using insertions, deletions, and substitutions.
- **Training:** Stage 1 trains mask-based density estimation. Stage 2 adds edit-based denoising so the model must re-evaluate visible content rather than assuming every unmasked token is correct. The objective still includes clean-sequence denoising terms.
- **Denoising / inference:** Parallel iterative denoising with additional trajectory-design and on-policy techniques aimed at fast code generation.
- **Key idea / feature:** **Mask → Edit curriculum.** The later stage explicitly teaches the model to correct existing content and reduces the harmful bias that visible tokens must already be correct.
- **Why it matters in the timeline:** This is one of the clearest precedents for **changing the nature of the corruption/denoising task as training progresses**. It means that the broad idea of multi-stage corruption is no longer unexplored, but the specific ordering and semantics of the stages remain open research questions.

---

## 2025 — Dream 7B: Large AR-initialized diffusion LLMs

### Dream 7B: Diffusion Large Language Models — Ye et al. (2025)

Paper: [arXiv:2508.15487](https://arxiv.org/abs/2508.15487)

- **Initialization:** **AR-based pretrained LLM initialization.** Dream explicitly reuses a large autoregressive model rather than starting from scratch.
- **Corruption:** Discrete diffusion with **context-adaptive token-level noise rescheduling**.
- **Training:** Continue training the AR-initialized model under a diffusion objective while adapting it to bidirectional, arbitrary-order denoising.
- **Denoising / inference:** Iterative parallel refinement with tunable quality/speed trade-offs, arbitrary-order generation, and infilling.
- **Key idea / feature:** **AR initialization works at 7B scale**, and the noise schedule can adapt to local context.
- **Why it matters in the timeline:** The pretrained-AR branch is no longer a small-model experiment. By 2025, converting existing large AR knowledge into a diffusion model becomes a serious scaling strategy.

---

## 2025 — Soft-Masked DLMs: Carry information from earlier denoising steps

### Soft-Masked Diffusion Language Models — Hersche et al. (2025)

Paper: [arXiv:2510.17206](https://arxiv.org/abs/2510.17206)

- **Initialization:** Adapts **pretrained masked diffusion models**; experiments include continued pretraining and fine-tuning of existing diffusion models such as Dream-7B / Dream-Coder-7B.
- **Corruption:** Masked diffusion remains the underlying family.
- **Training:** Continue training the masked diffusion model so retained mask positions can consume a **soft mixture of `[MASK]` and top-k predictions from the previous denoising step**.
- **Denoising / inference:** Iterative masked denoising, but unresolved positions carry soft information from previous predictions instead of discarding everything and returning to a pure binary `[MASK]` state.
- **Key idea / feature:** **Previous model predictions can be useful conditioning for the next denoising step.**
- **Why it matters in the timeline:** It moves the field toward explicit information flow between denoising iterations. This is conceptually close to the motivation for training a model on its own intermediate predictions, although the mechanism is soft conditioning rather than hard argmax replacement.

---

# 2026 — The field separates into scaling, better schedulers, better samplers, and better adaptation recipes

## 2026 — Scaling Beyond Masked Diffusion Language Models

### Scaling Beyond Masked Diffusion Language Models — Sahoo et al. (2026)

Paper: [arXiv:2602.15014](https://arxiv.org/abs/2602.15014)

- **Initialization:** The compared diffusion families are trained in a controlled scaling study; AR-checkpoint adaptation is not the focus.
- **Corruption:** Direct comparison of **masked, uniform-state, and interpolating discrete diffusion**.
- **Training:** Scale the different diffusion families under matched modern training recipes, including models up to 1.7B parameters; the paper also studies a simpler cross-entropy training objective for masked diffusion.
- **Denoising / inference:** Each family uses its corresponding iterative diffusion sampler, and the paper evaluates quality together with practical sampling speed.
- **Key idea / feature:** **Corruption family still matters at scale.** Perplexity can be misleading when comparing different diffusion families.
- **Why it matters in the timeline:** The original D3PM question from 2021 remains alive in 2026: Mask, Uniform, and interpolating processes have different trade-offs. There is no single universally dominant corruption family.

---

## 2026 — The Diffusion Duality, Chapter II: Better samplers and a cheaper curriculum

### The Diffusion Duality, Chapter II: Ψ-Samplers — Deschenaux et al. (2026)

Paper: [arXiv:2602.21185](https://arxiv.org/abs/2602.21185)

- **Initialization:** Extends the Duo uniform-state diffusion line; pretrained AR conversion is not the main contribution.
- **Corruption:** Primarily **uniform-state diffusion**, while the proposed predictor-corrector samplers generalize to arbitrary noise processes.
- **Training:** Introduces a **memory-efficient curriculum** for the Gaussian-relaxation training phase, reducing training cost while maintaining comparable performance.
- **Denoising / inference:** Predictor-corrector **Ψ-samplers** repeatedly refine the sequence and continue improving as more sampling steps are allowed.
- **Key idea / feature:** **Training curriculum and sampling algorithm are independent optimization axes.**
- **Why it matters in the timeline:** By 2026 the field is no longer debating only the existence of discrete diffusion; it is separately optimizing corruption, curriculum, and inference trajectory.

---

## 2026 — Information-density-driven masking

### Mask Is What DLLM Needs: A Masked Data Training Paradigm for Diffusion LLMs — Ma et al. (2026)

Paper: [arXiv:2603.15803](https://arxiv.org/abs/2603.15803)

- **Initialization:** The contribution modifies diffusion-LLM masked-data training rather than proposing a new AR→diffusion backbone conversion; initialization is not the core research variable.
- **Corruption:** Still **mask-based**, but masking is no longer uniform. The scheduler prioritizes information-dense tokens and constructs complementary reasoning-oriented and syntax-oriented masked examples.
- **Training:** Train on intelligently selected masked positions so optimization effort is concentrated on information-rich content rather than uniformly distributed noise.
- **Denoising / inference:** Standard diffusion-LLM denoising; the paper's main innovation is training-time noise allocation rather than a new decoding algorithm.
- **Key idea / feature:** **Which tokens receive noise matters, even within one corruption family.**
- **Why it matters in the timeline:** Corruption design remains an active topic in 2026. The field is optimizing both corruption *family* and corruption *allocation*.

---

## 2026 — iLLaDA: Scale masked diffusion from scratch even further

### Improved Large Language Diffusion Models (iLLaDA) — Nie et al. (2026)

Paper: [arXiv:2606.25331](https://arxiv.org/abs/2606.25331)

- **Initialization:** **Trained from scratch.** iLLaDA is an 8B model with fully bidirectional attention.
- **Corruption:** Masked discrete diffusion throughout pretraining and supervised fine-tuning.
- **Training:** Large-scale masked-diffusion pretraining on 12T tokens followed by large-scale instruction/SFT training, retaining the diffusion objective instead of reverting to autoregression.
- **Denoising / inference:** Iterative masked diffusion with variable-length generation; the work also introduces confidence-based scoring for multiple-choice evaluation.
- **Key idea / feature:** **Native diffusion training can scale to the full modern LLM regime.**
- **Why it matters in the timeline:** It represents the opposite strategy from AR conversion: instead of adapting GPT/LLaMA, simply train a very large bidirectional diffusion model from scratch. Both strategies are now viable research branches.

---

## 2026 — UNIFUSION: Pretrained GPT-2 + Uniform/Random diffusion

### UNIFUSION: Adapting Autoregressive Language Models into Discrete Diffusion under a Unified Reverse-Rate Objective — Jiang et al. (2026)

Paper: [arXiv:2607.24507](https://arxiv.org/abs/2607.24507)

- **Initialization:** **Pretrained GPT-2 checkpoints** (124M and 355M) are continually adapted.
- **Corruption:** **Uniform-noise diffusion**, where every token can remain editable; the paper also derives a shared \(x_0\) interface connecting mask and uniform kernels.
- **Training:** Continual pretraining under a unified reverse-rate objective. Clean-token \(x_0\) predictions are converted into the parameterizations needed by different diffusion formulations.
- **Denoising / inference:** Iterative uniform-state diffusion; more sampling steps progressively improve the generation quality/diversity trade-off.
- **Key idea / feature:** **AR→diffusion adaptation is not restricted to Mask corruption.** Pretrained GPT knowledge can also be transferred into uniform/random-state diffusion.
- **Why it matters in the timeline:** It is a direct modern reference for the **pretrained GPT + Random/Uniform corruption** corner of the design space.

---

## 2026 — PreDiff-LM: Study the causal→bidirectional conversion itself

### PreDiff-LM: Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention — Yao et al. (2026)

Paper: [arXiv:2607.25157](https://arxiv.org/abs/2607.25157)

- **Initialization:** **Pretrained GPT-2 Medium**.
- **Corruption:** Masked discrete diffusion.
- **Training:** Continue training GPT-2 as a denoiser while addressing attention mismatch. The proposed hybrid attention preserves causal attention over an observed prompt but allows bidirectional attention in the masked target. It can be combined with DiffuGPT-style objective adaptation.
- **Denoising / inference:** Iterative masked diffusion using the adapted GPT-2 denoiser.
- **Key idea / feature:** **The causal-attention → bidirectional-attention transition is itself an important adaptation problem.** Pretrained initialization massively accelerates convergence in the reported setup.
- **Why it matters in the timeline:** This is one of the closest modern references for converting a pretrained GPT backbone into a denoiser. It focuses on attention adaptation rather than a curriculum over different corruption families.

---

## 2026 — Zarya: Make the transition from AR behavior to diffusion behavior gradual

### Zarya: A Hybrid Autoregressive–Masked Diffusion Language Model with Flexible Training and Dual-Mode Inference — Sinev et al. (2026)

Paper: [arXiv:2609.19868](https://arxiv.org/abs/2609.19868)

- **Initialization:** Trained as a **hybrid AR + masked-diffusion model**; the paper is not framed as simply converting a fixed pretrained AR checkpoint into a diffusion model.
- **Corruption:** Masked-diffusion patterns inside variable-size slots, with configurable grouped noise patterns.
- **Training:** Jointly optimize autoregressive and masked-diffusion objectives. A curriculum gradually increases slot granularity, moving from fine-grained AR-like learning toward coarser diffusion-style learning.
- **Denoising / inference:** Supports both masked-diffusion sampling and a slotted speculative-decoding mode with KV-cache reuse.
- **Key idea / feature:** **The AR→diffusion behavioral transition itself can be curricular and gradual.**
- **Why it matters in the timeline:** This is a very recent indication that *training trajectory* — not just final architecture/objective — is becoming an explicit research variable in diffusion LMs.

---

# The evolution in one view

The history can be summarized as a sequence of questions:

1. **D3PM (2021):** How do we formally define diffusion over discrete states?
   - Answer: categorical transition kernels \(Q_t\).
   - Discovery: corruption family strongly affects results.

2. **Diffusion-LM (2022):** Can we instead diffuse continuous word representations?
   - Answer: yes, using Gaussian diffusion over embeddings.

3. **DiffusionBERT / Masked-Diffuse LM (2023):** Can language-specific priors help?
   - Answer: yes — reuse pretrained denoisers and design token-aware/linguistic corruption.

4. **MDLM (2024):** Does discrete diffusion training need to be mathematically cumbersome in implementation?
   - Answer: often no — masked diffusion can reduce to simple clean-token cross-entropy training.

5. **DiffuGPT / DiffuLLaMA (2024):** Can pretrained autoregressive knowledge be reused?
   - Answer: yes — GPT/LLaMA checkpoints can be converted through continual diffusion training.

6. **LLaDA / GIDD / Duo (2025):** What matters once the basic model works?
   - Answer: scale, confidence-guided iterative decoding, self-correction, corruption family, and curriculum.

7. **Seed / Dream / Soft-Masked DLM (2025):** Can the training trajectory or intermediate predictions be exploited?
   - Answer: yes — switch denoising tasks during training, adapt large AR models, and carry information between denoising steps.

8. **2026 work:** Which recipe scales best?
   - Mask vs Uniform remains open.
   - Pretrained vs from-scratch are both viable.
   - Attention adaptation, curriculum, noise allocation, sampler design, and model scale are now separate research axes.

This progression changes the modern research question. It is no longer simply:

> **Can diffusion generate language?**

It is increasingly:

> **What training path best transforms linguistic representations into a robust iterative denoiser?**

---

# Our Idea — Progressive Corruption-Family Adaptation with Hard Self-Refinement

The implemented study sits at the intersection of the threads above rather than trying to replace the diffusion framework itself. It trains a denoiser on directly corrupted clean sequences; it does not implement explicit forward transition matrices or a probabilistic reverse diffusion sampler.

## Initialization

- Start from a **pretrained GPT-2 autoregressive backbone** (GPT-2 small in the current experiment configurations).
- Add `<|pad|>` and `<|mask|>` tokens, used as padding and masking tokens.
- Enable **bidirectional attention** for denoising.
- Add an explicit **timestep embedding** to condition the model on noise severity.

## Corruption

Three procedural discrete corruption families are used:

1. **Similar** — replace selected token IDs with one of their nearest neighbors by cosine similarity in the model's input embedding table at initialization. This is a static token-level approximation to semantic similarity, not a contextual semantic check.
2. **Mask** — replace selected tokens with `<|mask|>`.
3. **Random** — replace selected tokens with IDs drawn uniformly from the tokenizer vocabulary, including special tokens and, occasionally, the original token ID.

For each training example, a timestep is sampled independently. Its per-token corruption probability grows linearly from:

$$
0.01 \rightarrow 0.95
$$

as the timestep increases.

The same timestep-to-probability schedule applies independently within each corruption family:

$$
q_{\text{Similar}}(x_t\mid x_0,t),\qquad
q_{\text{Mask}}(x_t\mid x_0,t),\qquad
q_{\text{Random}}(x_t\mid x_0,t).
$$

The key curriculum operates on a **different axis**: training progress.

For example:

$$
\text{Similar} \rightarrow \text{Mask} \rightarrow \text{Random}
$$

with a schedule such as:

$$
30\% \rightarrow 30\% \rightarrow 40\%
$$

of the training epochs.

Thus the model does not randomly mix corruption families at every step. The **active family changes at epoch boundaries**, while a newly sampled timestep determines severity inside that family for each example. The current curriculum has three phases; it does not add a fourth phase that progressively increases random replacement over training epochs.

## Training

The four-step experiment configurations use **multi-step direct \(x_0\)-reconstruction with hard self-conditioning**. One-step configurations use the same setup without intermediate feedback.

Given corrupted input \(x_t\):

$$
x_t
\rightarrow \hat{x}^{(1)}
\rightarrow \hat{x}^{(2)}
\rightarrow \cdots
\rightarrow \hat{x}^{(K)}.
$$

In the current multi-step configurations, \(K=4\).

At each rollout step, with feedback after the first \(K-1\) steps:

1. The model predicts token distributions.
2. `argmax` predictions are inserted back into **all originally corrupted positions**.
3. The resulting sequence — including the model's own mistakes — becomes the input to the next rollout step.
4. Every rollout step is supervised directly against the original clean sequence \(x_0\).

The loss at rollout step \(k\) is approximately:

$$
L_k = CE(\hat{x}^{(k)},x_0)
$$

on originally corrupted positions. Rollout steps have different weights:

$$
w_k = \lambda^{k-1},\qquad
L_{\text{train}} = \frac{\sum_{k=1}^{K} w_k L_k}{\sum_{k=1}^{K} w_k}.
$$

The current four-step configurations set \(\lambda=0.5\), giving weights \(1, 0.5, 0.25, 0.125\): earlier rollout steps contribute more to the training loss. Feedback uses `argmax` under `no_grad`, so later losses do not backpropagate through the discrete replacement operation.

This exposes the model to **its own imperfect intermediate reconstructions**. Validation performed inside the training loop uses this same all-positions rollout; the separate final evaluation uses the selective procedure below.

## Denoising / inference

Separate final evaluation uses a **confidence-guided, hard-token iterative refinement** procedure:

1. Start from a corrupted sequence sampled at the maximum timestep with the configured corruption rate.
2. Predict all still-unresolved corrupted positions.
3. Rank predictions by confidence.
4. Commit only the most confident positions.
5. Leave the remaining positions as they were and run the model again; committed positions are not revisited.
6. Repeat while the model timestep decreases from its maximum to zero.

This is close in spirit to confidence-guided masked decoding, but it also operates on visible Similar and Random replacements. Training differs because it feeds back hard predictions at all originally corrupted positions, while final evaluation commits only selected positions.

## Key idea / feature

The method combines **two progressive mechanisms**:

### 1. Progressive corruption-family curriculum

The intended information profile is:

$$
\text{Similar}
\rightarrow
\text{Mask}
\rightarrow
\text{Random}.
$$

Intuitively:

- **Similar** substitutes embedding-neighbor token IDs, which may preserve some lexical information but are not guaranteed to be semantically close in context.
- **Mask** removes token information and forces contextual reconstruction.
- **Random** usually inserts unrelated visible token IDs and asks the model to recover the original; the sampled ID can also equal the original or be a special token.

Whether this ordering is truly easier→harder is an **empirical hypothesis**, not an assumption that should be treated as proven.

### 2. Hard self-refinement during training

One-step denoising training can create a train–inference mismatch:

- during training the model only sees corruption generated by a known corruption function;
- during iterative inference the model sees **its own previous predictions**, including mistakes.

The multi-step rollout addresses part of this mismatch by training on model-generated intermediate states. Its all-positions feedback still differs from the selective commitment used in final evaluation.

Whether this improves iterative stability is an empirical question. The current evaluation records one-step and final denoising accuracy, but not accuracy after every intermediate inference iteration.

## Why this is a logical next research question

The literature already establishes the individual pieces:

- **D3PM:** corruption family matters.
- **DiffusionBERT:** pretrained linguistic knowledge can initialize a diffusion denoiser.
- **MDLM/LLaDA:** direct clean-token prediction and iterative masked decoding can be simple and effective.
- **DiffuGPT/DiffuLLaMA, Dream, PreDiff-LM, UNIFUSION:** pretrained autoregressive backbones can be adapted into diffusion denoisers.
- **GIDD / Duo:** editable random/uniform states and self-correction matter.
- **Seed Diffusion / Duo / Zarya:** curriculum and training trajectory can matter.
- **Soft-Masked DLM:** previous denoising predictions can be useful at subsequent steps.
- **2026 scaling studies:** no single corruption family has eliminated the others; Mask and Uniform remain meaningfully different design choices.

What remains less directly explored in the reviewed literature is the **combination**:

$$
\boxed{
\text{pretrained AR backbone}
+
\text{staged Similar}\rightarrow\text{Mask}\rightarrow\text{Random curriculum}
+
\text{shared timestep-dependent severity within each family}
+
\text{hard multi-step }x_0\text{ self-refinement}
}
$$

The resulting research question is:

> **Can a pretrained autoregressive language model be adapted into a more robust iterative denoiser by progressively changing the nature of corruption during training, rather than training under one fixed corruption family or a static mixture?**

A second linked question is:

> **Does exposing the model during training to its own imperfect intermediate reconstructions improve stability and cross-corruption generalization during iterative denoising?**

These questions should be evaluated against controlled baselines rather than treated as assumed advantages.

## Recommended controlled comparisons

To isolate the effects, compare models with the same backbone, data, optimizer, update budget, seed, and evaluation pipeline. The current configurations cover the three single-family baselines and the staged curriculum at one and four training steps; the one-step configurations omit a fixed seed, whereas the four-step configurations use seed 42. Additional comparisons in the list remain recommendations:

- Similar only
- Mask only
- Random only
- Static mixed corruption (not implemented; the current `mix_30_30_40` configuration is a staged curriculum)
- Similar → Mask → Random curriculum
- Alternative curriculum orders, if compute permits
- 1-step training rollout
- 2-step training rollout (not configured yet)
- 4-step training rollout

Evaluate both:

- **in-family denoising** — train corruption matches test corruption;
- **cross-corruption generalization** — test on a different corruption family;
- **overall robustness** — aggregate performance over all corruption families;
- **iterative stability** — accuracy/quality after denoising iteration 1, 2, 3, ... rather than only at the final step (not currently recorded by final evaluation).

A particularly informative diagnostic is the denoising trajectory:

$$
\text{metric after iteration }1,2,3,\dots,N.
$$

Recording this trajectory would directly test whether repeated self-refinement improves or degrades the sequence; current final-evaluation files contain only one-step and final iterative accuracy.

---

# Short takeaway

The field evolved roughly as follows:

```text
2021  D3PM
      discrete Q_t kernels; corruption family matters
          ↓
2022  Diffusion-LM
      continuous embedding diffusion branch
          ↓
2023  DiffusionBERT / Masked-Diffuse LM
      pretrained denoisers + linguistic/token-aware corruption
          ↓
2024  MDLM
      masked diffusion training becomes simple x0 prediction
          ↓
2024  DiffuGPT / DiffuLLaMA
      pretrained AR → diffusion conversion
          ↓
2025  LLaDA / GIDD / Duo
      scale + confidence decoding + self-correction + curriculum
          ↓
2025  Seed / Dream / Soft-Masked DLM
      staged training + large AR initialization + cross-step information
          ↓
2026  Scaling / Duo II / iLLaDA / UNIFUSION / PreDiff / Zarya
      compare families at scale; optimize adaptation, curricula and samplers
          ↓
OUR QUESTION
      Can pretrained AR → denoiser adaptation be made better by
      a staged Similar → Mask → Random curriculum plus explicit
      training on the model's own intermediate predictions?
```

The proposed work is therefore best positioned not as a new definition of discrete diffusion, but as a study of **training trajectory for adapting pretrained autoregressive representations into robust iterative denoisers**.
