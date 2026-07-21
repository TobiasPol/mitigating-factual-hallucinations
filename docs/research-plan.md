# Mitigating Factual Hallucinations in LLMs
## Introduction

Large language models can generate fluent and seemingly authoritative responses even when the underlying information is incorrect. In factual question answering, this creates two opposing failure modes. A model may provide a confident but fabricated answer, or it may refuse to answer a question for which it actually possesses sufficient knowledge. A reliable system must therefore do more than simply reduce the number of incorrect responses: it must learn to distinguish between questions it can answer correctly and questions for which abstention is the safer behavior.

Recent work indicates that factual correctness, uncertainty, and hallucination risk are partially represented in the internal activations of language models. The baseline paper demonstrates that adding a contrastively derived steering vector to internal FFN or MoE activations can substantially change model behavior at inference time, reducing incorrect responses and increasing admissions of uncertainty without retraining the full model. However, the reported results also reveal important limitations, including increased abstention, higher perplexity, and occasional language drift. 

This study builds on that approach by investigating whether activation steering can be made more adaptive, selective, and behaviorally precise. Instead of applying one fixed direction to every prompt, the proposed experiments examine prompt-conditioned steering, sparse and disentangled interventions, and the interaction between internal steering and system-level instructions. The methods will be evaluated on AA-Omniscience, SimpleQA Verified, and TriviaQA using the single, exactly pinned `mlx-community/Qwen3.6-27B-4bit` representation of `Qwen/Qwen3.6-27B` on Apple silicon.

## Motivation

A reduction in hallucination rate is not sufficient evidence that a model has become more knowledgeable or reliable. A method can appear successful simply because it causes the model to refuse more frequently. In the extreme case, a system that refuses every query would produce no factual hallucinations, but it would also have no practical utility. The central challenge is therefore to reduce incorrect attempted answers while preserving correct answers and limiting unnecessary abstention.

Activation steering offers a promising direction because it modifies model behavior directly within the inference process. Unlike retrieval-based systems, it does not depend on an external knowledge source, and unlike conventional fine-tuning, it can potentially be applied without modifying all model weights. However, static dense steering vectors may combine several different behavioral features, including factuality, uncertainty, refusal style, language, and safety behavior. This entanglement may explain why interventions intended to improve factuality can also produce over-refusal or other undesirable side effects.

The motivation for the proposed study is therefore threefold:

1. **Improve adaptivity.** Different questions may require different steering directions or intervention strengths. A prompt-conditioned controller may intervene only when the model is at meaningful risk of hallucinating.

2. **Separate factuality from refusal.** The experiments must determine whether steering increases the probability of the correct answer or merely increases the probability of saying “I do not know.”

3. **Reduce collateral effects.** Sparse and disentangled steering may isolate factuality-related internal features while preserving utility, language consistency, and safety behavior.

System prompts must also be treated as a serious baseline. Instructions that explicitly encourage abstention may reproduce part of the apparent benefit of activation steering at substantially lower implementation cost. Testing prompts and steering both independently and jointly is therefore necessary to establish whether activation-based interventions provide a genuine advantage beyond ordinary instruction following.

The long-term objective is to develop a risk-aware inference system that answers questions when the model is likely to be correct, applies targeted activation steering when factual knowledge appears recoverable, and abstains only when the remaining risk of an incorrect answer is too high. Completely eliminating hallucinations in all possible situations cannot be established through a finite benchmark. Nevertheless, the proposed study aims to move toward **near-zero observed hallucination at meaningful answer coverage**, while explicitly measuring the trade-off between factual reliability and practical usefulness.

# Research plan: Risk-gated activation steering for near-zero factual hallucination

## 1. Scientific objective

The study should treat hallucination mitigation as a **selective prediction problem**:

> **Maximize the proportion of questions the model answers correctly, subject to a strict upper bound on the probability of an incorrect attempted answer.**

Formally:

[
\max_{\pi} ; \mathrm{Coverage}(\pi)
]

subject to:

[
\mathrm{HallucinationRisk}(\pi) \leq \epsilon
]

and:

[
\Delta \mathrm{Utility} \geq -\delta_u,\quad
\Delta \mathrm{Safety} \geq -\delta_s
]

where the policy (\pi) includes the system prompt, steering method, steering strength, layer selection, and abstention decision.

This framing is preferable to simply maximizing accuracy. A model that refuses every question has no attempted hallucinations but also has no utility. AA-Omniscience partly addresses this through its Omniscience Index, but abstention receives zero rather than a negative score, meaning a refuse-all system can still obtain an index of zero. Coverage, accuracy-given-attempted, and risk therefore need to be reported alongside the benchmark’s official score. ([arXiv][1])

The ultimate target should be stated as:

> **Zero observed incorrect attempted answers on a preregistered test set, at nontrivial answer coverage, with a statistical upper confidence bound on the residual hallucination risk.**

No finite benchmark can establish that a model will never hallucinate on every possible input. It can establish that a system achieved zero observed errors under specified conditions. For example, if the system attempts (n) questions and produces zero incorrect answers, the approximate one-sided 95% upper bound on the true error rate is (3/n). Zero errors across 1,000 attempted answers therefore supports an upper bound of approximately 0.3%, not universal zero risk.

---

# 2. Important setup decisions

## 2.1 Clarify the two meanings of “prompt-conditioned”

Your research questions use two related but distinct concepts:

* **Prompt-conditioned activation steering in RQ1** means that the steering vector, layer, or strength is selected dynamically from the internal representation of the current user query.
* **System-prompt intervention in RQ4** means changing the textual instruction given to the model.

These should remain separate experimental factors. Otherwise, the effects of adaptive internal control and ordinary instruction-following become confounded.

## 2.2 Approved local single-model amendment

The approved local experiment uses `Qwen/Qwen3.6-27B` on an Apple M4 Max with 48 GiB unified memory. Every E0–E10 work directory and ledger belongs to the `qwen36-27b-mlx4-m4max48-v1` study namespace. The exact single-model decision is frozen in `configs/experiments/model-selection-amendment.json`; no legacy-model artifact is a prerequisite.

The sole active model is:

| Role | Exact checkpoint and runtime |
| --- | --- |
| Activation extraction, steering, probes, SAE work, adaptive controllers, and final evaluation | Uniform affine group-64 4-bit MLX artifact `mlx-community/Qwen3.6-27B-4bit` at revision `c000ac2c2057d94be3fa931000c31723aac53282`, representing `Qwen/Qwen3.6-27B`, using official `mlx==0.31.2` and `mlx-lm==0.31.3` |

The pinned runtime artifact has 64 text blocks, hidden size 5,120, 48 linear-attention blocks, 16 full-attention blocks, and a 16,081,490,064-byte snapshot. The manifest pins all 16 files by Git-blob SHA-1 or LFS SHA-256. The uniform 4-bit artifact was selected instead of a layer-dependent mixed-precision conversion to reduce quantization heterogeneity in the activation study. Before E0, a live preflight on the approved M4 Max must verify the exact model class, tokenizer, complete layer-type sequence, deterministic no-thinking chat rendering, zero-vector parity, exact prompt-token scope, and nonzero cached-continuation sensitivity for all three intervention sites on both a linear-attention and a full-attention block. ([Qwen 3.6 model card][2]) ([Qwen 3.6 MLX artifact][13]) ([MLX-LM][14])

The exact model inventory and static runtime requirements are bound by `configs/models/qwen3.6-27b-mlx-4bit.snapshot.json` and `configs/runtimes/qwen3.6-27b-mlx-4bit-policy.json`. The live preflight command freezes the actual machine identity, macOS build, Xcode/Metal toolchain, hook evidence, code hashes, and peak memory in a non-overwritable receipt. The 500-question cohort and contamination-review artifacts are model-independent and remain valid. A new Qwen E0 must complete inside the Qwen namespace before a new E1 is admitted.

The Colab notebook, bundle, and result-import workflow are retired. Cross-model and cross-runtime replication are outside the current single-model study and require a separately frozen amendment.

## 2.3 Scope consequence of using one model

All primary comparisons are within the exact Qwen MLX checkpoint: each intervention is paired against its own M0 baseline on the same questions. This design can test the causal and selective-prediction hypotheses, but it cannot establish cross-model or cross-runtime replication. Conclusions must therefore be limited to this model/runtime combination. Any future second model or deployment representation requires a new amendment, independently learned vectors, and a separate confirmatory run.

---

# 3. Research questions, hypotheses, and falsification criteria

| RQ                                              | Main hypothesis                                                                                                                                       | Evidence supporting the hypothesis                                                                                                              | Evidence against the hypothesis                                                                                                               |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **RQ1: Adaptive versus static steering**        | A query-conditioned vector, layer, or strength will outperform one global centroid vector.                                                            | Better AA Omniscience Index and SimpleQA F1; lower risk at matched coverage; better transfer from TriviaQA to AA/SimpleQA.                      | Gains disappear after matching answer coverage, or adaptive steering overfits TriviaQA and underperforms static steering out of distribution. |
| **RQ2: Knowledge expression versus abstention** | Steering will recover some latent factual knowledge rather than merely increasing “I don’t know.”                                                     | More incorrect-to-correct transitions, higher gold-answer likelihood, improved accuracy-given-attempted under a forced-answer stress condition. | Almost all avoided errors become abstentions; gold-answer likelihood does not increase; improvement disappears when abstention is prohibited. |
| **RQ3: Sparse/disentangled steering**           | Restricting the intervention to factuality-related features will reduce collateral effects.                                                           | Similar or better factuality at matched coverage, with better instruction following, language preservation, safety, and perplexity.             | Sparse methods lose most of the factuality gain, or protected subspaces fail to prevent safety/refusal/language regressions.                  |
| **RQ4: System-prompt interaction**              | System prompts will reproduce part of the effect, but internal steering will provide additional benefit and may interact non-additively with prompts. | Steering improves factuality beyond prompt-only; prompt × steering interaction is measurable; prompt paraphrases do not eliminate the effect.   | A simple calibrated prompt fully reproduces the effect, or steering only works under one exact prompt formulation.                            |

---

# 4. Experimental scope and operational definitions

The main study should be explicitly scoped to:

> **Closed-book, short-form factual question answering without retrieval, external tools, browsing, or supplied evidence.**

This matches AA-Omniscience and SimpleQA Verified, both of which are intended to test parametric factual knowledge without external tools. ([arXiv][1])

In this setting, classify every response into:

* **Correct (C):** Semantically matches an accepted reference answer.
* **Partial (P):** Contains a materially correct but incomplete answer. Preserve this category for AA-Omniscience.
* **Incorrect attempted (I):** Makes a factual commitment that does not match the reference.
* **Abstention (A):** Explicitly declines to provide a factual answer.
* **Unscorable (U):** Malformed, empty, off-topic, or truncated output.

An incorrect attempted answer is the study’s operational definition of a factual hallucination. This does not automatically generalize to long-form summarization hallucinations, fabricated citations, multimodal hallucinations, or reasoning errors.

## 4.1 Central metrics

Let:

[
N=C+P+I+A
]

For benchmarks without partial credit:

[
\mathrm{Accuracy} = \frac{C}{N}
]

[
\mathrm{Coverage} = \mathrm{AttemptRate} = \frac{C+I}{N}
]

[
\mathrm{HallucinationRisk} = \frac{I}{C+I}
]

[
\mathrm{AccuracyGivenAttempted} = \frac{C}{C+I}
]

[
\mathrm{AbstentionRate} = \frac{A}{N}
]

The two most important quantities are **coverage** and **hallucination risk**. Every method should be compared on a risk–coverage curve rather than at only one arbitrarily selected steering coefficient.

## 4.2 Transition metrics

Because every intervention is run on the same questions, construct paired transition matrices from the base response to the intervened response.

Report:

[
\mathrm{KnowledgeRecovery} = \frac{I\rightarrow C}{I_{\mathrm{base}}}
]

[
\mathrm{AbstentionSubstitution} = \frac{I\rightarrow A}{I_{\mathrm{base}}}
]

[
\mathrm{StrictOverRefusal} = \frac{C\rightarrow A}{C_{\mathrm{base}}}
]

[
\mathrm{Regression} = \frac{C\rightarrow I}{C_{\mathrm{base}}}
]

[
\mathrm{CorrectPreservation} = \frac{C\rightarrow C}{C_{\mathrm{base}}}
]

These directly answer RQ2. Calling all abstentions “over-refusal” would be misleading. The strongest operational definition of over-refusal is a question the base model answered correctly but the intervention causes it to refuse.

---

# 5. Benchmark protocol

## 5.1 TriviaQA: development, training, and in-distribution testing

TriviaQA contains approximately 95,000 question-answer pairs and more than 650,000 question-answer-evidence triples. For this study, discard the evidence documents and use a closed-book question-only setting. ([arXiv][6])

TriviaQA should be the primary source for:

* Building static steering vectors.
* Training adaptive risk probes and routers.
* Training sparse autoencoders.
* Hyperparameter selection.
* In-distribution evaluation.
* Hard-negative mining.

Recommended split:

| Split          |     Size | Purpose                                                        |
| -------------- | -------: | -------------------------------------------------------------- |
| `T-steer`      |   30,000 | Centroids, CAA pairs, SAE feature identification               |
| `T-controller` |    5,000 | Risk probe, vector router, confidence calibration              |
| `T-dev`        |    5,000 | Layer, token position, (\alpha), sparsity and prompt selection |
| `T-test`       |    5,000 | Frozen TriviaQA confirmatory test                              |
| Remaining data | Reserved | Hard-negative mining and robustness analysis only              |

The split should be deduplicated by normalized question text, named entity, and accepted answer aliases. Where possible, make `T-test` entity-disjoint from the steering set to reduce memorization of specific entity-relation patterns.

Because TriviaQA predates the models by many years, it has a comparatively high contamination risk. It should therefore be treated mainly as a large development and in-distribution benchmark, not as the strongest evidence of novel factual generalization.

## 5.2 SimpleQA Verified: untouched out-of-distribution evaluation

SimpleQA Verified contains 1,000 curated short-form factual prompts. Its evaluation separates correct, incorrect, and not-attempted responses. Its F1 score combines overall accuracy with accuracy-given-attempted, making it particularly useful for detecting systems that obtain lower hallucination rates merely by refusing more. ([arXiv][7])

Protocol:

* Do not use any SimpleQA Verified item for steering construction, SAE feature selection, layer selection, or prompt optimization.
* Freeze all method hyperparameters before the first full run.
* Use the benchmark’s released grader instructions.
* Report both official metrics and the unified (C/I/A) metrics from this plan.
* Audit all grader disagreements and a stratified human sample.

## 5.3 AA-Omniscience: calibrated knowledge-boundary evaluation

AA-Omniscience consists of 6,000 questions across 42 topics and six domains. It evaluates both factual recall and whether the model recognizes knowledge gaps. ([arXiv][8])

A significant limitation is that the public release contains only 10%, or 600 questions. Unless access to the complete set is obtained, the study should:

* Name the evaluation explicitly as **AA-Omniscience Public-600**.
* Report overall results only.
* Avoid strong domain-level conclusions.
* Treat domain and topic analyses as exploratory because individual cells will be small. ([arXiv][1])

AA’s official answerer prompt already instructs the model to abstain rather than guess. This is directly relevant to RQ4. Run two tracks:

1. **Official track:** Exact official answerer prompt and official scoring.
2. **Controlled prompt track:** The study’s neutral, direct-answer, and calibrated-abstention prompts.

Custom-prompt AA results should not be presented as directly leaderboard-comparable to the official condition. ([arXiv][1])

## 5.4 Contamination control

Before final testing:

1. Normalize punctuation, casing, articles, and whitespace.
2. Perform exact and fuzzy matching between TriviaQA training questions and AA/SimpleQA questions.
3. Use MinHash or character n-gram similarity for near duplicates.
4. Use sentence-embedding similarity to identify paraphrased duplicates.
5. Manually inspect the highest-similarity pairs.
6. Remove overlapping TriviaQA items from all representation-training sets.
7. Publish the overlap report.

The exact MLX conversion does not establish a sufficiently precise training-data cutoff for temporal contamination claims. Contamination status must therefore be documented as unknown rather than assumed absent; exact, fuzzy, semantic, and manual overlap checks remain mandatory.

---

# 6. Standardized inference protocol

The primary evaluation must minimize decoding-related variance.

| Setting          | Primary value                         |
| ---------------- | ------------------------------------- |
| Retrieval/tools  | Disabled                              |
| Temperature      | 0                                     |
| Sampling         | Disabled                              |
| Maximum output   | 32–48 answer tokens                   |
| Conversation     | One system message, one user question |
| Explanation      | Not requested                         |
| Chain-of-thought | Disabled/not exposed                  |
| Stop condition   | EOS or first completed short answer   |
| Batch ordering   | Randomized across conditions          |
| Seeds            | Fixed and logged                      |
| Chat template    | Official model-specific template      |

Thinking is disabled through the checkpoint's official chat-template control in the primary experiment because hidden reasoning length and additional token generation would introduce another intervention factor. Any thinking-enabled robustness run must use a separately frozen 10–20% diagnostic subset and cannot replace the primary text-only evidence.

Each output record should contain:

```text
question_id
benchmark
model_revision
runtime
quantization
system_prompt_id
rendered_prompt_hash
steering_method
layer
token_scope
alpha
sparsity
controller_scores
raw_output
normalized_answer
C/P/I/A/U label
generation_latency
input_tokens
output_tokens
```

---

# 7. Intervention methods and baselines

The uploaded paper should be implemented as the exact primary static baseline. It computes the normalized difference between truthful and hallucinated FFN/MoE activation centroids and adds the resulting dense direction at inference. Its results also motivate measuring abstention, perplexity, and language drift rather than reporting only apparent accuracy gains. 

Existing activation-steering methods provide several relevant baselines. CAA averages activation differences between positive and negative behavioral examples; ITI intervenes in selected attention heads associated with truthfulness; ACT uses multiple truthfulness-related vectors and adaptive intensity; SADI dynamically changes selected internal elements based on input semantics; and TruthX uses an autoencoder to separate semantic content from truthfulness-related representations. ([ACL Anthology][9])

Sparse-autoencoder methods such as SAE-SSV and SPARE provide methodological baselines for sparse feature selection and representation editing. Recent work also uses sparse representations and protected subspaces to separate hallucination mitigation from refusal and safety behavior. ([arXiv][10])

## 7.1 Main method set

### M0 — Unmodified model

No steering. Evaluated under every system-prompt condition.

### M1 — Static final-FFN centroid steering

Exact uploaded-paper baseline:

[
v_l =
\frac{\mu_{C,l}-\mu_{I,l}}
{\left|\mu_{C,l}-\mu_{I,l}\right|_2}
]

[
h'*{l,t}=h*{l,t}+\alpha v_l
]

Implement two versions:

* **M1-R:** Centroids computed from response-token activations, matching the uploaded paper as closely as possible.
* **M1-P:** Centroids computed at the final prompt token, before generation begins.

M1-P is especially important because response-token directions may encode answer style, wording, or refusal markers in addition to factuality.

### M2 — CAA-style paired steering

Construct contrast pairs using:

* Teacher-forced correct answers.
* The model’s original incorrect answers.
* Matched correct and incorrect examples within similar semantic clusters.

Then:

[
v_l^{CAA} =
\frac{1}{N}\sum_i
\left(h_l(x_i^+)-h_l(x_i^-)\right)
]

CAA should be applied at the residual stream and compared with the final-MLP centroid baseline.

### M3 — Prompt-conditioned adaptive steering

The proposed adaptive system should contain three components.

#### A. Hallucination-risk probe

At the final prompt token, extract activations from selected layers and train a calibrated classifier:

[
q(x)=
\left[
P(C\mid x),
P(I\mid x),
P(A\mid x)
\right]
]

Start with logistic regression and a two-layer MLP. Calibrate the probabilities with temperature scaling or isotonic regression on `T-controller`.

#### B. Vector bank

Cluster prompt representations into (K) semantic regions:

[
{v_1,\ldots,v_K}
]

Evaluate:

[
K\in{1,4,8,16}
]

The (K=1) condition reduces to global static steering.

A router computes mixture weights:

[
w(x)=\mathrm{softmax}(W h_{\text{prompt}})
]

and selects:

[
v(x)=\sum_{k=1}^{K}w_k(x)v_k
]

#### C. Dynamic intensity and layer selection

Define:

[
\alpha(x)=
\alpha_{\max}
\cdot
\sigma\left(
\beta\left[P(I\mid x)-\tau\right]
\right)
]

This ensures that questions predicted to be safely answerable receive little or no intervention.

An advanced version can select both the layer and vector:

[
(l^*,v^*,\alpha^*)=\pi(h_{\text{prompt}})
]

The initial confirmatory method should keep layer selection simple and route only between two or three preregistered candidate layers to avoid overfitting.

### M4 — Sparse steering

Implement two levels.

#### M4a: Coordinate-sparse steering

Compute a standardized effect size per activation dimension:

[
d_j =
\frac{\mu_{C,j}-\mu_{I,j}}
{\sqrt{\frac{1}{2}
(\sigma^2_{C,j}+\sigma^2_{I,j})+\epsilon}}
]

Keep only the largest:

[
s\in{1%,5%,10%,25%}
]

of dimensions by absolute (d_j).

Then:

[
h'=h+\alpha(m_s\odot v)
]

This is computationally inexpensive and provides a necessary baseline before attributing improvements to SAE-based interpretability.

#### M4b: Sparse-autoencoder steering

Train an SAE on the selected residual-stream or MLP-output activations:

[
z=\mathrm{Encoder}(h)
]

[
\hat{h}=\mathrm{Decoder}(z)
]

with sparse (z).

Identify latent features associated with:

* Correct answers.
* Incorrect attempted answers.
* Abstention.
* Language switching.
* Safe refusal behavior.

Construct a sparse factuality direction in latent space:

[
v_z =
\mu_{z,C}-\mu_{z,I}
]

and decode the intervention:

[
h' =
h+\alpha D(v_z)
]

where (D) is the SAE decoder.

Use an expansion factor of approximately (8\times) initially, with TopK or (L_1)-regularized sparsity. Compare several sparsity levels and verify SAE reconstruction quality before steering.

### M5 — Disentangled or protected-subspace steering

Learn separate directions or low-dimensional bases for:

* Truthful versus incorrect factual answering.
* Attempt versus abstention.
* Safe refusal versus harmful compliance.
* Requested-language consistency.
* General answer verbosity/style.

Let (U) be an orthonormal basis for protected behavior directions. Remove these components from the factuality vector:

[
v_{\mathrm{protected}}
======================

(I-UU^\top)v_{\mathrm{truth}}
]

Also evaluate a covariance-aware version:

[
v^*=
\arg\max_v
\left[
v^\top d_{\mathrm{truth}}
-\lambda v^\top \Sigma_{\mathrm{protected}}v
\right]
]

This explicitly optimizes factuality gain while penalizing activation changes in directions associated with refusal, safety, language, and utility.

### M6 — Final composite system

After RQ1–RQ4 have been analyzed independently, combine the winning components:

1. Calibrated system prompt.
2. Prompt-end risk probe.
3. Prompt-conditioned vector bank.
4. Sparse or protected steering direction.
5. Minimum necessary (\alpha).
6. Post-first-token risk re-evaluation.
7. Abstention only if residual risk remains above threshold.

M6 is the principal candidate for the near-zero-hallucination deployment system.

## 7.2 Negative and causal controls

Every steering experiment should include:

* Label-shuffled centroid vector.
* Random direction with matched norm.
* Opposite-direction intervention (-v).
* Correct vector applied at an unrelated layer.
* Norm-matched Gaussian perturbation.
* Zero-vector hook to test implementation overhead.
* Vector trained under a different system prompt.

These controls help distinguish a factuality-specific causal effect from generic perturbation, regularization, or refusal induction.

---

# 8. Layer and token-position search

Use the frozen Qwen MLX candidate set. Because its 64 blocks comprise 48 linear-attention and 16 full-attention blocks, the search deliberately pairs architecture types near the midpoint and three-quarter depth while retaining early, late, and final controls.

| Purpose | Zero-indexed block |
| --- | ---: |
| Early linear-attention control | 16 |
| Mid-depth full-attention block | 31 |
| Mid-depth linear-attention block | 32 |
| Three-quarter full-attention block | 47 |
| Three-quarter linear-attention block | 48 |
| Late linear-attention control | 57 |
| Final full-attention control | 63 |

At each layer, test:

* Post-attention residual.
* Post-MLP output.
* Block output/residual stream.

Token scopes:

1. Final prompt token only.
2. First generated token only.
3. First four generated tokens.
4. First eight generated tokens.
5. Every generated token.
6. Exponentially decaying steering:

[
\alpha_t=\alpha_0 e^{-\lambda t}
]

The main layer and token scope must be selected on `T-dev` and then frozen.

Use standardized steering strength:

[
\tilde{\alpha}
==============

\frac{\alpha}{\mathrm{RMS}(h_l)}
]

rather than transferring raw (\alpha) values across layers.

Suggested pilot grid:

[
\tilde{\alpha}
\in
{0,0.1,0.25,0.5,1.0,2.0}
]

Negative values should be retained only as causal controls.

---

# 9. System-prompt experiment

## 9.1 Primary prompt conditions

### P0 — Neutral

> You are a helpful assistant. Answer the user’s factual question.

### P1 — Direct short answer

> Give only the shortest answer that directly answers the question. Do not add an explanation.

### P2 — Calibrated abstention

> Answer only when you are sufficiently confident that the answer is correct. Otherwise respond exactly with “I don’t know.” Do not guess.

### P3 — Forced-answer stress test

> Give your best short answer even when uncertain. Do not abstain.

P3 is not a proposed deployment policy. It is a diagnostic condition for RQ2 and should be used only on benign factual benchmarks, not on safety benchmarks.

## 9.2 Prompt paraphrase robustness

For P0 and P2, create five semantically equivalent paraphrases. Test them on a fixed 200-question subset from each benchmark.

This measures whether an intervention works because of the underlying policy or because it was tuned to one exact string.

## 9.3 Cross-prompt steering transfer

For each steering method, compare:

1. Vector learned under P0 and applied under P0.
2. Vector learned under P0 and applied under P2.
3. Vector learned under P2 and applied under P0.
4. Vector learned under P2 and applied under P2.

This distinguishes:

* A genuinely prompt-invariant truthfulness direction.
* A prompt-specific refusal or style direction.
* Destructive interference between system instructions and activation steering.
* Cases where the prompt and steering redundantly push toward abstention.

---

# 10. Staged experimental program

## E0 — Runtime and checkpoint validation

**Purpose:** Ensure the exact local model/runtime is scientifically usable before representation learning.

Run 500 shared benign factual prompts twice on the sole Qwen 3.6 27B 4-bit MLX checkpoint under deterministic decoding.

Measure:

* Correctness.
* Abstention.
* Average answer length.
* Latency and memory.
* Exact snapshot, tokenizer, and chat-template identity.
* Repeat-run output and token identity.
* Zero-vector parity and nonzero intervention sensitivity at post-attention residual, MLP-output, and block-output sites in uncached and cached paths.

E0 is complete only after the local MLX artifact replays successfully and the model-independent contamination review is complete. Superseded runtime outputs cannot satisfy this gate.

## E1 — Baseline prompt characterization

**Conditions:**

[
1 \text{ model}
\times
3 \text{ benchmarks}
\times
3 \text{ primary prompts}
]

Objectives:

* Establish baseline C/P/I/A distributions.
* Measure prompt-only hallucination reduction.
* Identify over-refusal caused by P2.
* Quantify whether P1 increases attempted hallucinations.
* Build model-specific labels for later representation analysis.

This is the prompt-only baseline required for RQ4.

## E2 — Activation separability and knowledge-boundary probing

Before steering, determine whether internal states predict response outcome.

Train probes to distinguish:

* (C) versus (I).
* Attempt versus abstention.
* (C/I/A) jointly.
* Correct versus incorrect under forced-answer P3.

Metrics:

* AUROC.
* Macro F1.
* Brier score.
* Expected calibration error.
* Out-of-distribution AUROC on AA and SimpleQA.

Gate:

> Continue with prompt-conditioned steering only if prompt-end activations provide materially better prediction than trivial confidence baselines such as output entropy or maximum token probability.

## E3 — Static steering replication

Implement M1-R and M1-P.

Run:

* Layer sweep.
* Site sweep.
* Alpha sweep.
* Token-scope sweep.
* Causal controls.

Primary question:

> Can the uploaded paper’s static centroid method reproduce factuality gains on different architectures and more demanding benchmarks?

A successful replication requires improvement on `T-dev` that is not explained solely by a major decrease in coverage.

## E4 — Existing baseline comparison

Compare:

* Uploaded-paper centroid steering.
* CAA.
* One attention-head intervention such as ITI, where technically feasible.
* One adaptive baseline inspired by ACT or SADI.
* TruthX-style autoencoder editing if implementation resources permit.

Screen these methods on a 2,000-question development subset. Promote only the strongest external baselines to the full final evaluation.

## E5 — Prompt-conditioned adaptive steering

Train the risk probe, vector bank, and dynamic (\alpha) controller on TriviaQA only.

Ablations:

| Component           | Values                                                 |
| ------------------- | ------------------------------------------------------ |
| Number of vectors   | 1, 4, 8, 16                                            |
| Router              | Nearest centroid, linear softmax, two-layer MLP        |
| Alpha               | Fixed, risk-gated, risk-gated with hard threshold      |
| Layer               | Fixed best layer, two-layer router, three-layer router |
| Intervention timing | Prompt end, first token, first four tokens             |
| Controller input    | One layer, concatenated layers, layer differences      |

The main RQ1 comparison is adaptive steering versus M1 static steering at:

* Equal coverage.
* Equal abstention rate.
* Equal average intervention norm.
* Equal latency budget.

## E6 — Knowledge expression versus abstention

Run the strongest static and adaptive methods under P0, P2, and P3.

For every question, compute:

1. Generated outcome transition.
2. Teacher-forced log-likelihood of every accepted gold-answer alias.
3. Teacher-forced log-likelihood of the standard abstention phrase.
4. Rank of the gold answer among plausible alternatives where available.

Define:

[
\Delta LL_{\mathrm{gold}}
=========================

## LL_{\mathrm{steered}}(y^*)

LL_{\mathrm{base}}(y^*)
]

[
\Delta LL_{\mathrm{abstain}}
============================

## LL_{\mathrm{steered}}(\text{“I don’t know”})

LL_{\mathrm{base}}(\text{“I don’t know”})
]

Interpretation:

* Large positive (\Delta LL_{\mathrm{gold}}) plus (I\rightarrow C): evidence of improved knowledge expression.
* Large positive (\Delta LL_{\mathrm{abstain}}) plus (I\rightarrow A): evidence of abstention induction.
* (C\rightarrow A): strict over-refusal.
* Improvement under P3, where abstention is discouraged: stronger evidence that steering changes factual answer selection rather than only response policy.

## E7 — Sparse steering

First run coordinate-sparse M4a. This establishes whether most of the dense direction can be removed without losing benefit.

Then train the SAE and run M4b.

Necessary SAE checks:

* Reconstruction error.
* Fraction of variance explained.
* Average active features per activation.
* Stability of selected features across random seeds.
* Whether selected features causally affect factuality when individually activated or suppressed.
* Whether the same feature controls refusal, language, or safety behavior.

Do not interpret a feature as “truthfulness” solely because it correlates with correct responses. Require intervention evidence.

## E8 — Disentanglement and protected steering

Build protected subspaces from:

* Correct-to-abstain transitions.
* XSTest safe-prompt refusals.
* Harmful-prompt refusal activations.
* Language-switching examples.
* General instruction-following failures.

Compare dense, sparse, and protected directions at matched factuality gain.

RQ3’s main comparison is not “which method obtains the highest raw factuality.” It is:

> At the same hallucination risk or the same coverage, which method causes the least utility, safety, and language degradation?

## E9 — Full system-prompt factorial

Run:

[
\text{Prompt}
\in
{P0,P1,P2}
]

crossed with:

[
\text{Method}
\in
{
M0,
M1,
M2,
M3,
M4,
M5
}
]

for the sole active model and all three factuality benchmarks.

If the TriviaQA final set contains 5,000 questions, the core confirmatory matrix contains:

[
1\times3\times6\times(5000+1000+600)
====================================

118{,}800
]

short generations.

P3 and prompt-paraphrase conditions should run on smaller preregistered diagnostic subsets.

## E10 — Frozen composite evaluation

Construct M6 from the best independently selected components.

Freeze:

* Model revision.
* Prompt.
* Risk threshold.
* Vector bank.
* SAE checkpoint.
* Protected subspace.
* Layer.
* Alpha policy.
* Abstention rule.
* Grader.
* Evaluation scripts.

Then run once on:

* `T-test`.
* SimpleQA Verified.
* AA-Omniscience Public-600 or the complete 6,000 if obtained.
* Utility, language, and safety suites.

No tuning should follow this run. Any later changes constitute a new experiment.

---

# 11. Main confirmatory evaluation matrix

Recommended full matrix:

| Method             | Neutral P0 | Direct P1 | Calibrated P2 |
| ------------------ | ---------: | --------: | ------------: |
| Base M0            |          ✓ |         ✓ |             ✓ |
| Static centroid M1 |          ✓ |         ✓ |             ✓ |
| CAA M2             |          ✓ |         ✓ |             ✓ |
| Adaptive M3        |          ✓ |         ✓ |             ✓ |
| Sparse M4          |          ✓ |         ✓ |             ✓ |
| Disentangled M5    |          ✓ |         ✓ |             ✓ |

Run this for the sole active model.

The composite M6 is evaluated after selection and is not used for component-level hypothesis tests.

For computational efficiency, develop components on the designated TriviaQA partitions, freeze candidate designs before target-benchmark evaluation, stream activation statistics, and keep the model and SAE training stages out of memory at the same time. Cross-model and cross-runtime replication are explicitly deferred.

---

# 12. Evaluation metrics by benchmark

## 12.1 AA-Omniscience

Primary:

* Official Omniscience Index.
* Hallucination risk.
* Coverage.
* Accuracy-given-attempted.
* Correct, partial, incorrect, and not-attempted rates.

Secondary:

* Risk–coverage curve.
* Results by domain only with sufficient sample size.
* Prompt sensitivity.

Use the benchmark’s released scoring implementation as the source of truth, especially for partial-credit handling.

## 12.2 SimpleQA Verified

Primary:

* Official F1.
* Overall accuracy.
* Correct-given-attempted.
* Attempt rate.
* Incorrect-attempted rate.

The benchmark’s F1 is especially valuable because it balances overall correctness against correctness among attempted answers. ([arXiv][7])

Secondary:

* Hedging rate.
* Punting/abstention rate.
* Risk–coverage curve.
* Base-to-intervention transition matrix.

## 12.3 TriviaQA

Primary:

* Exact match.
* Token-level F1.
* Accuracy-given-attempted.
* Hallucination risk.
* Coverage.
* Area under the risk–coverage curve.

Use all accepted aliases. For teacher-forced likelihood, calculate normalized per-token likelihood for each alias and use the best matching accepted alias.

---

# 13. Utility, safety, and language evaluation

The three selected factual benchmarks cannot by themselves answer RQ3. A separate side-effect suite is necessary.

## 13.1 General utility

Use:

* **IFEval** for verifiable instruction-following performance.
* A fixed stratified 1,000-question **MMLU-Pro** sample for general knowledge and reasoning retention.
* A fixed **WikiText-103** subset for next-token negative log-likelihood and perplexity.

IFEval provides verifiable instruction constraints, while MMLU-Pro offers a more challenging multi-domain knowledge and reasoning evaluation than the original MMLU. ([arXiv][11])

Report:

[
\Delta \mathrm{IFEval}
]

[
\Delta \mathrm{MMLUPro}
]

[
\Delta \mathrm{Perplexity}
]

relative to each model’s unsteered baseline.

## 13.2 Over-refusal and safety

Use:

* **XSTest** to detect exaggerated safety behavior and refusal of benign prompts.
* **StrongREJECT** or **HarmBench** to detect loss of refusal behavior on genuinely harmful prompts.

XSTest was designed specifically to expose exaggerated safety and unnecessary refusal, while StrongREJECT and HarmBench provide standardized evaluations of harmful compliance and refusal robustness. ([arXiv][12])

Report:

* Safe-prompt refusal rate.
* Harmful-prompt compliance rate.
* Harmful-prompt refusal rate.
* Safety grader score.
* Base-to-steered safety transitions.

Do not use the forced-answer system prompt P3 on harmful requests.

## 13.3 Language consistency

Because the factual benchmarks are predominantly English, construct a compact auxiliary suite.

Recommended design:

* 100 English questions containing names from non-Latin scripts.
* 100 German questions.
* 100 Spanish questions.
* 100 French questions.
* 100 Japanese questions.

Use human-verified translations of a TriviaQA subset and explicitly request an answer in the prompt language.

Metrics:

* Correct output-language rate.
* Non-target script-token rate.
* Code-switching rate.
* Factual accuracy by language.
* Abstention rate by language.
* Human language-consistency score.
* (C\rightarrow) wrong-language transitions.

A model should not receive full credit merely because its factual content is correct if the intervention causes unintended language switching.

---

# 14. Analysis plan for each research question

## RQ1 — Adaptive versus static steering

### Primary comparisons

[
M3-M1
]

on:

* AA Omniscience Index.
* SimpleQA Verified F1.
* Hallucination risk at 25%, 50%, 75%, and 90% coverage.
* Area under the risk–coverage curve.
* TriviaQA held-out EM/F1.

### Required controls

Compare at:

* Matched coverage.
* Matched abstention rate.
* Matched average intervention norm.
* Matched inference latency where possible.

Otherwise, an adaptive method could appear superior simply because it abstains more often or applies a stronger perturbation.

### Generalization tests

1. Train all vectors/controllers only on TriviaQA.
2. Evaluate unchanged on SimpleQA Verified and AA.
3. Train on neutral P0 and test under P2.
4. Leave one semantic cluster out during training and evaluate on it.
5. Relearn only the calibration threshold versus relearning the entire vector bank.

A method that performs well only after benchmark-specific tuning should not be described as a general factuality controller.

## RQ2 — Knowledge expression versus abstention

Use four forms of evidence.

### A. Outcome transitions

Compare (I\rightarrow C) with (I\rightarrow A).

### B. Gold-answer likelihood

Measure whether the intervention increases the internal probability of accepted answers.

### C. Forced-answer condition

If the intervention still improves correctness when P3 discourages abstention, that supports a factual-answer-selection effect.

### D. Correct-answer preservation

Measure (C\rightarrow C), (C\rightarrow A), and (C\rightarrow I).

A strong result would look like:

* Substantial (I\rightarrow C).
* Some appropriate (I\rightarrow A).
* Low (C\rightarrow A).
* Low (C\rightarrow I).
* Increased gold-answer likelihood.
* Improved accuracy-given-attempted at matched coverage.

A weak result would look like:

* Large drop in attempted rate.
* Most (I) converted to (A), not (C).
* No improvement under P3.
* Higher abstention-phrase likelihood without higher gold-answer likelihood.

## RQ3 — Sparse and disentangled steering

### Primary comparison

Compare M4 and M5 against M1 and M3 at a matched factuality operating point.

For example:

> Among methods producing a 30% relative reduction in hallucination risk, which method preserves the most utility, language consistency, and safety?

### Non-inferiority analysis

Predefine pilot-informed non-inferiority margins, for example approximately 1–2 absolute percentage points for major utility and safety metrics.

Evaluate:

* MMLU-Pro accuracy.
* IFEval pass rate.
* XSTest safe refusal.
* HarmBench/StrongREJECT safety.
* Language preservation.
* Perplexity.
* Latency.

### Interpretability analysis

For selected SAE features:

* Show top-activating examples.
* Suppress the feature and test whether hallucination changes.
* Activate the feature and test whether refusal, safety, or language also changes.
* Test stability across random SAE seeds.
* Test whether features transfer across prompt formulations.

“Disentangled” should be claimed only when factuality can be changed while protected behaviors remain statistically non-inferior.

## RQ4 — System-prompt effects

Use a two-factor model:

[
Y
\sim
\mathrm{Method}
+
\mathrm{Prompt}
+
\mathrm{Method}\times\mathrm{Prompt}
]

The interaction can also be summarized as:

[
\mathrm{Interaction}
====================

\left[
Y(M,P2)-Y(M0,P2)
\right]
-------

\left[
Y(M,P0)-Y(M0,P0)
\right]
]

Interpretation:

* Positive interaction: the calibrated prompt amplifies steering.
* Near-zero interaction: prompt and steering are approximately additive.
* Negative interaction: prompt and steering interfere.
* Strong negative coverage effect: combination produces over-refusal.

Also report:

* Prompt-only gain.
* Steering-only gain.
* Combined gain.
* Prompt-paraphrase variance.
* Cross-prompt vector-transfer performance.
* Official-AA-prompt versus neutral-AA-prompt results.

---

# 15. Grading and human validation

## 15.1 Automated grading

**Pre-E1 grader amendment (2026-07-17).** The originally frozen AA judge,
`google/gemini-2.5-flash-preview-09-2025`, was retired by OpenRouter before any
E1 generation or grading and returned no available endpoint. By explicit operator
instruction, AA grading now uses the stable `google/gemini-2.5-flash` successor
through OpenRouter. The released AA grading prompt, C/P/I/A label mapping,
temperature, reasoning setting, parsing, and failure policy remain unchanged.
This is a grader-checkpoint deviation: reports must identify AA labels as
stable-Gemini-2.5-Flash rubric grades and must not claim exact judge-checkpoint
comparability with the preview-based released implementation. The exact transport,
catalog evidence, superseded identity, and replacement identity are frozen in
`configs/experiments/grader-selection-amendment.json`.

**SimpleQA execution qualification (frozen before E1).** The study preserves the
released SimpleQA Verified rubric, prompt, label semantics, and pinned GPT-4.1
judge, but its fail-closed OpenRouter adapter is deliberately stricter than the
pinned starter notebook: it sends temperature zero, accepts only an atomic
`A`/`B`/`C` label, retries transient failures up to three times, and records a
terminal failure as unscorable `U`. The notebook instead omits temperature,
recovers labels with regular expressions and keywords, invokes once, and defaults
an unparseable result to `C` (not attempted). Reports must therefore describe
these as released-rubric GPT-4.1 grades under the study adapter, not as byte-for-byte
execution of the released notebook implementation.

* TriviaQA: deterministic alias-aware exact match and token F1.
* SimpleQA Verified: released grader rubric.
* AA-Omniscience: released grading and scoring implementation.
* Side benchmarks: official scorers where available.

Freeze:

* Grader model and version.
* Grader prompt.
* Temperature.
* Parsing logic.
* Failure handling.

The SimpleQA Verified paper provides an explicit grading rubric for correct, incorrect, and not-attempted answers, including handling of hedging and punts. ([arXiv][7])

## 15.2 Human audit

Use two annotators who are blind to:

* Model.
* Steering condition.
* System prompt.
* Experimental hypothesis.

Audit:

* At least 200 responses per benchmark per model, stratified across methods and outcomes.
* All automated-grader disagreements.
* All partial AA responses.
* All language-switching detections.
* All suspected safety regressions.
* A random sample of abstentions and incorrect attempts.

Report:

* Cohen’s (\kappa) or Krippendorff’s (\alpha).
* Adjudicated final labels.
* Automated-versus-human confusion matrix.

---

# 16. Statistical analysis

## 16.1 Preregister primary comparisons

Recommended confirmatory contrasts:

1. **RQ1:** M3 versus M1 under P0 and P2 on AA and SimpleQA.
2. **RQ2:** Transition decomposition for M1 and M3 versus M0.
3. **RQ3:** M4 and M5 versus M1 at matched hallucination risk or coverage.
4. **RQ4:** Prompt × method interaction for M0, M1, M3, and M5.

All other layer, vector-count, SAE, and alpha analyses should be declared exploratory unless promoted before the frozen test.

## 16.2 Confidence intervals and paired tests

Use:

* 10,000-question-level paired bootstrap resamples for metric differences.
* McNemar’s test for paired binary correctness.
* Stuart–Maxwell or Bowker tests for paired (C/I/A) distributions.
* Mixed-effects multinomial or logistic regression.
* Question-level random intercepts.
* Fixed effects for model, benchmark, method, prompt, and interactions.
* Holm correction for the preregistered family of primary comparisons.
* Non-inferiority tests for utility and safety.

The question—not the generated token—is the statistical unit.

## 16.3 Risk–coverage analysis

For each method, vary:

* Steering threshold.
* Risk threshold.
* Alpha.
* Abstention threshold.

Plot:

* Hallucination risk versus coverage.
* Accuracy versus coverage.
* Over-refusal versus hallucination reduction.
* Utility degradation versus factuality gain.
* Safety degradation versus factuality gain.

The most defensible “best method” is the one dominating the Pareto frontier, not necessarily the one with the lowest raw hallucination count.

## 16.4 Power analysis

After E1, use the observed paired transition rates to simulate statistical power.

Do not base power only on independent-proportion formulas, because every question is evaluated under all conditions. Paired discordant transitions determine the relevant power.

The 1,000 SimpleQA questions and 600 public AA questions are likely adequate for overall medium-size effects, but AA Public-600 is underpowered for detailed topic-level analysis.

---

# 17. Implementation architecture

## 17.1 Research runtime

Use the pinned official MLX and MLX-LM releases on Apple Metal for model forward passes, activation extraction, steering injection, cached generation, and teacher-forced answer likelihoods. Use NumPy or PyTorch on CPU/MPS for probes and SAE optimization only when their arrays and checkpoints remain numerically compatible with the MLX activation schema. The runtime must expose the three validated intervention sites while loading the exact released 4-bit weights without conversion.

Pin:

* Model revision, complete snapshot inventory, and verified snapshot digest.
* Tokenizer and chat-template digests.
* Official MLX and MLX-LM versions and wheel hashes.
* Python dependency lock and project metadata digests.
* macOS, Xcode, Metal compiler/toolchain, chip, and unified-memory identity.
* Activation site, block, token scope, cache behavior, and random seeds.

## 17.2 Activation storage

Do not store full activations for every token and every layer.

Store:

* Final prompt-token activations.
* First one, four, or eight output-token activations.
* Selected candidate layers only.
* Float16 or bfloat16 shards.
* Outcome label and question ID.

For centroid construction, use online statistics such as Welford accumulation rather than retaining every activation.

For SAE training, create a separately sampled activation corpus and store it in sharded memory-mapped arrays. An 8x expansion at hidden size 5,120 has roughly 419 million encoder/decoder weights before optimizer state; the 27B model must be unloaded before SAE training, memory must be measured in a pilot, and the expansion or batch size must be reduced if the 48 GiB envelope is exceeded. This resource adjustment may not change the frozen activation corpus or causal-validation gates.

## 17.3 Local execution and memory discipline

The active study has one native MLX runtime. Run one model process at a time, write resumable immutable shards, aggregate centroid statistics online, and release model arrays plus the Metal cache before probe or SAE training. Every long computation must record wall time, peak unified memory, package-lock identity, snapshot digest, and resumable-chain head. Evidence from any other runtime cannot substitute for the registered MLX run.

---

# 18. Proposed near-zero-hallucination inference system

The final system should not steer every question equally.

## 18.1 Inference policy

### Step 1: Prompt-level assessment

Extract prompt-end activations and estimate:

[
P(C),P(I),P(A)
]

### Step 2: Classify the query into one of three regimes

#### Known

[
P(I)<\tau_{\mathrm{low}}
]

Answer normally or with minimal steering.

#### Potentially recoverable

[
\tau_{\mathrm{low}}
\leq
P(I)
<
\tau_{\mathrm{high}}
]

Apply prompt-conditioned sparse/protected steering.

#### Likely unknown

[
P(I)\geq\tau_{\mathrm{high}}
]

Abstain or escalate.

### Step 3: Early-generation re-evaluation

After the first answer token or first short token block:

* Recompute risk.
* Detect whether steering is increasing gold-answer likelihood.
* Detect language or refusal drift.
* Stop or abstain if residual risk remains too high.

### Step 4: Output gate

Only release the answer when:

[
\widehat{\mathrm{Risk}}
\leq\epsilon
]

and no safety or language constraint is violated.

### Step 5: Optional real-world escalation

Outside the closed-book benchmark:

* Route abstained questions to retrieval.
* Request external verification.
* Ask a clarification question.
* Return a sourced answer.

Retrieval should remain disabled in the central experiment so activation steering’s contribution is isolated.

---

# 19. Success criteria

Use three levels.

## Level 1 — Meaningful hallucination reduction

* At least 30% relative reduction in incorrect attempted answers.
* No more than five percentage points of coverage loss.
* No statistically meaningful safety degradation.
* Reproduced across the frozen prompt paraphrases and benchmark domains within the sole model.

## Level 2 — High-reliability selective answering

* Hallucination risk below 1% at at least 50% coverage on both AA and SimpleQA.
* Significant improvement over prompt-only.
* Sparse/disentangled method non-inferior on utility and safety.
* Prompt paraphrase robustness.
* Improvement transferred from TriviaQA without target-benchmark tuning.

## Level 3 — Zero observed hallucinations

* Zero incorrect attempted answers on the frozen AA and SimpleQA evaluation.
* Meaningful preregistered coverage, for example at least 40–50%.
* Exact one-sided confidence bound reported.
* No severe over-refusal.
* No significant safety, language, or utility regression.
* Stable results across the preregistered prompt paraphrases and held-out benchmarks.
* A transparent single-model limitation statement; no claim of cross-model or cross-runtime replication.

---

# 20. Timeline

| Phase                                     | Weeks | Main outputs                                    |
| ----------------------------------------- | ----: | ----------------------------------------------- |
| Infrastructure and benchmark reproduction |   1–2 | Model loading, chat templates, scorers, logging |
| Baseline and prompt-only runs             |   3–4 | E0–E1, baseline C/I/A distributions             |
| Activation probing                        |     5 | E2, layer separability and calibrated probes    |
| Static steering replication               |   6–7 | E3, layer/alpha/token-scope selection           |
| External baseline screening               |     8 | E4, CAA/ITI/ACT/SADI/TruthX comparison          |
| Adaptive steering                         |  9–11 | E5, router and dynamic-alpha controller         |
| Knowledge-versus-abstention study         |    12 | E6, transitions and likelihood analysis         |
| Sparse SAE work                           | 13–15 | E7, coordinate and SAE steering                 |
| Disentanglement and protected subspaces   | 16–17 | E8, safety/refusal/language protection          |
| Full prompt factorial                     |    18 | E9                                              |
| Side-effect suites and human audit        | 19–20 | Utility, safety, language results               |
| Frozen composite evaluation               |    21 | E10                                             |
| Local runtime stress and replay audit      | 22–23 | MLX throughput, memory, and artifact replay     |
| Statistical analysis and writing          |    24 | Final tables, confidence intervals, paper       |

The approved local version executes only the Qwen 3.6 27B 4-bit MLX checkpoint on the 48 GiB M4 Max. It must stream or aggregate activation statistics and avoid retaining full-token/full-layer activations. The model is unloaded before memory-heavy probe or SAE work. Results from other models or runtimes are not interchangeable evidence.

---

# 21. Required final figures and tables

The paper should include:

1. **Risk–coverage curves** for the model on every benchmark.
2. **C/I/A transition diagrams** showing knowledge recovery versus abstention substitution.
3. **Gold-answer versus abstention log-likelihood changes.**
4. **Layer × alpha heatmaps.**
5. **Static versus adaptive steering at matched coverage.**
6. **Dense versus sparse versus disentangled Pareto plots.**
7. **Prompt × steering interaction heatmaps.**
8. **Prompt-paraphrase robustness plots.**
9. **Safety and utility non-inferiority plots.**
10. **Language-switching confusion matrices.**
11. **Local MLX runtime identity, intervention-site, latency, and memory results.**
12. **Zero-error confidence-bound table.**

---

# 22. Expected research contribution

The strongest contribution would not be another result stating that activation steering lowers factual error rates. It would be a complete decomposition of what that improvement represents:

1. Whether adaptive steering improves over one global vector.
2. Whether it recovers latent knowledge or primarily teaches the model to abstain.
3. Whether factuality can be separated from refusal, language, safety, and general capability.
4. Whether ordinary system prompts already account for the apparent gains.
5. Whether the resulting intervention can be converted into a practical, risk-gated inference system.

The final system should therefore be presented as:

> **A calibrated, prompt-conditioned, sparse activation-steering policy that answers when estimated factual risk is below a threshold, intervenes only when the answer appears recoverable, and abstains when residual risk remains too high.**

That is a scientifically defensible route toward near-zero observed hallucination while avoiding the trivial and unusable solution of refusing every question.

[1]: https://arxiv.org/html/2511.13029v1 "https://arxiv.org/html/2511.13029v1"
[2]: https://huggingface.co/Qwen/Qwen3.6-27B "https://huggingface.co/Qwen/Qwen3.6-27B"
[3]: https://github.com/ml-explore/mlx "https://github.com/ml-explore/mlx"
[4]: https://github.com/ml-explore/mlx-lm "https://github.com/ml-explore/mlx-lm"
[5]: https://github.com/ml-explore/mlx "https://github.com/ml-explore/mlx"
[6]: https://arxiv.org/abs/1705.03551 "https://arxiv.org/abs/1705.03551"
[7]: https://arxiv.org/html/2509.07968v2 "https://arxiv.org/html/2509.07968v2"
[8]: https://arxiv.org/abs/2511.13029 "https://arxiv.org/abs/2511.13029"
[9]: https://aclanthology.org/2024.acl-long.828/ "https://aclanthology.org/2024.acl-long.828/"
[10]: https://arxiv.org/html/2505.16188v1 "https://arxiv.org/html/2505.16188v1"
[11]: https://arxiv.org/abs/2311.07911 "https://arxiv.org/abs/2311.07911"
[12]: https://arxiv.org/abs/2308.01263 "https://arxiv.org/abs/2308.01263"
[13]: https://huggingface.co/mlx-community/Qwen3.6-27B-4bit "https://huggingface.co/mlx-community/Qwen3.6-27B-4bit"
[14]: https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3 "https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3"
