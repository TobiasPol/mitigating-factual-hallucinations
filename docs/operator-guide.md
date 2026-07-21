# Research operator guide

This guide turns the scientific design in `research-plan.md` into an executable
workflow. The repository supplies the contracts, inference hooks, methods,
evaluation logic, phase gates, immutable ledgers, analysis schema, and reporting
checks. It does **not** ship model weights, benchmark snapshots, grader service
credentials, completed human annotations, or scientific E0–E10 results.

## 1. Scientific boundary

There are two deliberately separate execution paths:

- A scientific path built around `StudyProtocol`, `PhaseRunContract`, and
  `PhaseRunLedger`. Only a complete, verified ledger with the exact required
  inputs, prerequisites, records, and passing gates counts as phase evidence.
- A deterministic synthetic smoke path. It integrates representative production
  components across E0–E10, but every file says `scientific_eligible: false` and
  `runtime: synthetic`. It cannot be opened as a phase ledger, does not write a
  `complete.json`, and cannot reserve or satisfy the one-shot E10 run.

The smoke path is a software check, not a miniature experiment. Its E4 operating
points are fabricated and its E9 matrix is intentionally small.

## 2. Environment and protocol preflight

Development and verification:

```bash
uv sync --extra dev
uv lock --check
uv run pytest
uv run ruff check .
uv run mypy --strict src/mfh
```

Install the research dependencies and native MLX runtime only on the approved
Apple-silicon host:

```bash
uv sync --extra dev --extra research --extra mlx-macos
```

The active study uses only `mlx-community/Qwen3.6-27B-4bit` at revision
`c000ac2c2057d94be3fa931000c31723aac53282`, representing
`Qwen/Qwen3.6-27B`, on an Apple M4 Max with exactly 48 GiB unified memory. The
optional extra installs the lock-pinned official `mlx==0.31.2` and
`mlx-lm==0.31.3` packages. Install Xcode and its matching Metal Toolchain
component first; if either is installed after a failed build, use a clean build
cache.

No command downloads a model or benchmark implicitly. Place every artifact on
disk deliberately, retain its upstream provenance, and verify it before a run.

The analysis protocol is bound to the exact bytes of the research plan. Run this
after any checkout and before constructing scientific artifacts:

```bash
uv run mfh validate-study \
  configs/experiments/phases.yaml \
  configs/analysis/confirmatory.yaml \
  docs/research-plan.md
```

This must report all phases E0–E10 and `valid: true`. Editing the research plan
invalidates the frozen analysis digest until the change is reviewed and the
protocol is intentionally updated.

Method grids and thresholds have no detached YAML override. Their authoritative
values are the bound research plan and study/analysis protocols; each selected
controller, sparse model, protected direction, and M6 policy is then serialized
as a typed, recursively verified frozen artifact. This prevents an apparently
normative configuration file from changing without affecting execution.

Individual model, benchmark, prompt, grader, study, and analysis YAML files can
be checked with:

```bash
uv run mfh validate-config configs/models/qwen3.6-27b-mlx-4bit.yaml
uv run mfh validate-config configs/benchmarks/triviaqa.yaml
uv run mfh validate-config configs/prompts/primary.yaml
uv run mfh validate-config configs/experiments/phases.yaml
uv run mfh validate-config configs/analysis/confirmatory.yaml
```

## 3. Deterministic integration smoke

Run the representative E0–E10 component integration before allocating model
compute:

```bash
uv run mfh synthetic-smoke artifacts/synthetic-smoke-1701 --seed 1701
uv run mfh verify-synthetic-smoke artifacts/synthetic-smoke-1701
```

The second command verifies the exact directory inventory, checks the digest
chain, and re-executes every deterministic phase. Reusing the destination is
rejected; create a new directory for a new smoke run.

Expected layout:

```text
artifacts/synthetic-smoke-1701/
├── manifest.json
└── phases/
    ├── E0.json
    ├── ...
    └── E10.json
```

Do not copy these files into a scientific run or cite their metrics.

## 4. Acquire and freeze external inputs

The active model config, complete snapshot manifest, runtime policy, live receipt, and
amendment jointly pin the repository revision, all file sizes and SHA-256 values,
tokenizer, chat template, quantization, dependency lock,
hardware, and hook preflight. The exact snapshot lives at:

```text
artifacts/models/qwen3.6-27b-mlx-4bit/c000ac2c2057d94be3fa931000c31723aac53282/
```

Acquire that exact revision explicitly; no experiment command downloads it:

```bash
MODEL=artifacts/models/qwen3.6-27b-mlx-4bit/c000ac2c2057d94be3fa931000c31723aac53282
STUDY=artifacts/studies/qwen36-27b-mlx4-m4max48-v1

uv run hf download mlx-community/Qwen3.6-27B-4bit \
  --revision c000ac2c2057d94be3fa931000c31723aac53282 \
  --local-dir "$MODEL"
```

Remove only the Hugging Face download metadata directory `$MODEL/.cache` after
the download. Do not alter any of the 16 declared model files. Then verify the
symlink-free, exact inventory:

```bash
uv run mfh verify-transformers-snapshot \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json
```

Verify the model-independent 500-question cohort before allocating compute:

```bash
uv run mfh verify-runtime-validation \
  artifacts/e0/runtime-validation-500 \
  artifacts/splits/triviaqa-auto-clean/reserved.jsonl \
  --expected-manifest-digest bb89b2da16d899f8a38c0b090f84d1e43ffd1132e0fe0693295230b804f44442 \
  --parent-split-manifest-digest a3b646d7057c3e863c06b7ed0f446a28c63b8fb12e203e9b6b61cb2f2f8027f0 \
  --contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2
```

Before E0, verify the exact snapshot against
`configs/models/qwen3.6-27b-mlx-4bit.snapshot.json`, validate the static
`configs/runtimes/qwen3.6-27b-mlx-4bit-policy.json`, and run the live MLX hook
preflight on the approved machine:

```bash
mkdir -p "$STUDY/frozen"
uv run mfh preflight-mlx-runtime \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  configs/runtimes/qwen3.6-27b-mlx-4bit-policy.json \
  "$STUDY/frozen/mlx-preflight.json" \
  --project-root .
```

The destination is write-once. If preflight fails, diagnose the machine or
runtime and use a new study namespace only after protocol review; never edit a
receipt into a passing state.

All mutable scientific work, output bundles, checkpoints, and E0--E10 ledgers
must be strict descendants of `$STUDY`. The implementation rejects `..`
traversal, a symlinked study root, nested symlink components, copied ledgers
outside the namespace, and a path equal to the namespace root. The external E10
one-shot reservation under `~/.local/state/mfh/one-shot` is the sole documented
exception.

The preflight must pass uncached and cached zero-vector parity, exact prompt-token
scope, and nonzero sensitivity at all three intervention sites in both linear-
attention and full-attention blocks. Preserve the
external resume-chain head for every partial 500-question execution. The model
process must be released before SAE training or other memory-heavy analysis.

The earlier Gemma, Bonsai, Transformers, AWQ, GGUF, and Colab artifacts remain
immutable exploratory or pilot provenance. In particular, the completed Bonsai
E0 and its 17,244/19,800-record partial E1 stay at their original paths. They are
excluded from all active gates and must not be copied into the Qwen namespace.

Before benchmark splitting:

1. Acquire the pinned TriviaQA, SimpleQA Verified, and AA-Omniscience artifacts.
2. Canonicalize them through the benchmark loaders in `mfh.data.benchmarks`.
3. Run contamination checks before creating any split.
4. Freeze the exact source artifacts alongside confirmatory question bundles.

The frozen contamination scan first removes every normalized-question collision,
runs the 0.8 character-5-gram check, encodes all retained TriviaQA and OOD rows
with the pinned CPU/float32 `all-MiniLM-L6-v2` revision, applies the preregistered
0.9 cosine threshold, and publishes the global top-200 manual-review queue:

```bash
uv run mfh contamination-scan \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  artifacts/contamination/triviaqa-ood \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl

uv run mfh verify-contamination-scan \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2
```

The bundle deliberately marks its top-200 semantic queue `pending`. Publish and
independently verify the deterministic blinded worksheet before a human labels
it:

```bash
uv run mfh prepare-contamination-review \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  artifacts/contamination/triviaqa-ood-manual-review \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2

uv run mfh verify-contamination-review-queue \
  artifacts/contamination/triviaqa-ood-manual-review \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2 \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6
```

The frozen `annotation-template.csv` contains only randomized review IDs, the
two question texts, an empty `label`, and empty `notes`; it withholds similarity,
automatic flags, and benchmark IDs. Copy it to an operator-controlled editable
file, label every row manually as exactly `overlap` or `distinct` using
`rubric.md`, and do not edit the row order, IDs, or question text:

Give the reviewer only the copied annotation worksheet and `rubric.md`. Do not
give them `operator-bindings.jsonl`, either manifest, or the automatic scan
outputs; those reveal benchmark IDs, similarities, and automatic flags and
would break the blinded protocol.

```bash
cp artifacts/contamination/triviaqa-ood-manual-review/annotation-template.csv \
  artifacts/operator-inputs/triviaqa-ood-manual-annotations.csv
chmod u+w artifacts/operator-inputs/triviaqa-ood-manual-annotations.csv
```

Create `artifacts/operator-inputs/triviaqa-ood-reviewer-attestation.json` by hand
with the actual reviewer identity and timezone-bearing completion timestamp:

```json
{
  "schema_version": 1,
  "reviewer_id": "REPLACE_WITH_REVIEWER_ID",
  "reviewed_at": "REPLACE_WITH_ISO_8601_TIMESTAMP_AND_TIMEZONE",
  "attestation": "I manually compared every blinded question pair using the frozen overlap rubric without automated label generation."
}
```

Freeze and replay the completed decisions. Replace `REVIEW_RESULT_DIGEST` in the
second command with the digest printed by the first:

```bash
uv run mfh finalize-contamination-review \
  artifacts/contamination/triviaqa-ood-manual-review \
  artifacts/operator-inputs/triviaqa-ood-manual-annotations.csv \
  artifacts/operator-inputs/triviaqa-ood-reviewer-attestation.json \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  artifacts/contamination/triviaqa-ood-manual-review-result \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2 \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6

uv run mfh verify-contamination-review-result \
  artifacts/contamination/triviaqa-ood-manual-review-result \
  artifacts/contamination/triviaqa-ood-manual-review \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2 \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6 \
  --expected-result-manifest-digest REVIEW_RESULT_DIGEST
```

Two review rows currently touch the provisional E0 cohort:
`cr-93c88097a6616d8e44c1` (`triviaqa:wh_2951`) and
`cr-ea2a791868a2df746cc6` (`triviaqa:odql_5742`). If either is labeled
`overlap`, discard and rebuild the E0 cohort and the sole local runtime output. The
`complete-e0` command enforces this intersection check and cannot publish an E1
admission receipt otherwise.

Run the sole Qwen MLX E0 leg locally into a mutable work directory and a
never-before-used final directory. Preserve the resume-chain token printed after
each append outside the work directory. The runner validates the exact snapshot,
runtime receipt, prompt template, 500-question cohort, two-pass deterministic
schedule, token identities, memory evidence, and all three intervention sites.
E0's C/I counts use strict whole-response TriviaQA alias exact match under
P0-neutral and are diagnostic only; explanatory answers can contain the correct
alias while receiving I. Do not report those counts as model accuracy or reuse
them as E1 labels. E1 produces and grades its own frozen prompt-factorial outputs.

```bash
uv run mfh run-e0-mlx \
  artifacts/e0/runtime-validation-500 \
  artifacts/splits/triviaqa-auto-clean/reserved.jsonl \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$STUDY/work/E0-mlx" \
  "$STUDY/outputs/E0-mlx" \
  --expected-cohort-manifest-digest bb89b2da16d899f8a38c0b090f84d1e43ffd1132e0fe0693295230b804f44442 \
  --parent-split-manifest-digest a3b646d7057c3e863c06b7ed0f446a28c63b8fb12e203e9b6b61cb2f2f8027f0 \
  --contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2 \
  --checkpoint-file "$STUDY/operator-inputs/E0-mlx-resume.json"
```

If execution stops, read `resume_checkpoint` from the external checkpoint file
and rerun the same command with `--expected-resume-checkpoint DIGEST`. Never copy
the checkpoint file into either the mutable work directory or final bundle.

Once the native MLX leg and manual review are complete, publish the scientific E0
receipt. Replace every uppercase digest placeholder with the value recorded
outside the corresponding bundle:

```bash
uv run mfh complete-e0 \
  "$STUDY/outputs/E0-completion" \
  "$STUDY/outputs/E0-mlx" \
  artifacts/contamination/triviaqa-ood-manual-review-result \
  artifacts/contamination/triviaqa-ood-manual-review \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  artifacts/e0/runtime-validation-500 \
  artifacts/splits/triviaqa-auto-clean/reserved.jsonl \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-mlx-manifest-digest E0_MLX_MANIFEST_DIGEST \
  --expected-mlx-plan-identity E0_MLX_PLAN_IDENTITY \
  --expected-review-result-manifest-digest REVIEW_RESULT_DIGEST \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6 \
  --expected-cohort-manifest-digest bb89b2da16d899f8a38c0b090f84d1e43ffd1132e0fe0693295230b804f44442 \
  --parent-split-manifest-digest a3b646d7057c3e863c06b7ed0f446a28c63b8fb12e203e9b6b61cb2f2f8027f0 \
  --contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2
```

Replay it with `verify-e0-completion`: prepend the receipt path, add
`--expected-manifest-digest E0_COMPLETION_DIGEST`, and pass the identical live
evidence paths and digest options shown above. This CLI verification is also what
`authorize_e0_completion_receipt(...)` executes before issuing the capability
accepted by E0 ledger finalization.

The verifier always replays the exhaustive lexical discovery step and requires
the manifest digest recorded outside the bundle when it was created; copying a
digest out of the file being verified is not an independent trust anchor.
`--replay-embeddings` additionally performs the slower full model replay.

The completion receipt is not itself the E0 phase ledger. Record its manifest
digest and atomically publish the 500-row terminal ledger before starting E1:

```bash
uv run mfh finalize-e0-phase \
  "$STUDY/runs/E0" \
  "$STUDY/outputs/E0-completion" \
  "$STUDY/outputs/E0-mlx" \
  artifacts/contamination/triviaqa-ood-manual-review-result \
  artifacts/contamination/triviaqa-ood-manual-review \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  artifacts/e0/runtime-validation-500 \
  artifacts/splits/triviaqa-auto-clean/reserved.jsonl \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  --expected-manifest-digest E0_COMPLETION_DIGEST \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-mlx-manifest-digest E0_MLX_MANIFEST_DIGEST \
  --expected-mlx-plan-identity E0_MLX_PLAN_IDENTITY \
  --expected-review-result-manifest-digest REVIEW_RESULT_DIGEST \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6 \
  --expected-cohort-manifest-digest bb89b2da16d899f8a38c0b090f84d1e43ffd1132e0fe0693295230b804f44442 \
  --parent-split-manifest-digest a3b646d7057c3e863c06b7ed0f446a28c63b8fb12e203e9b6b61cb2f2f8027f0 \
  --contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2

uv run mfh verify-phase \
  "$STUDY/runs/E0" configs/experiments/phases.yaml
```

`E0_COMPLETION_DIGEST`, `E0_MLX_MANIFEST_DIGEST`,
`E0_MLX_PLAN_IDENTITY`, and `REVIEW_RESULT_DIGEST` are the externally recorded
values printed by the corresponding write-once commands. `verify-phase` must
succeed before E1 preparation.

The reviewed splitter replays the final review result and creates disjoint
T-steer, T-controller, T-dev, and T-test partitions. Replace
`REVIEW_RESULT_DIGEST` in both commands with the finalization digest and replace
`REVIEWED_SPLIT_DIGEST` in the verifier with the digest printed by the creator:

```bash
uv run mfh prepare-reviewed-splits \
  artifacts/splits/triviaqa-reviewed \
  artifacts/contamination/triviaqa-ood-manual-review-result \
  artifacts/contamination/triviaqa-ood-manual-review \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2 \
  --expected-review-result-manifest-digest REVIEW_RESULT_DIGEST \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6 \
  --steer 30000 --controller 5000 --dev 5000 --test 5000 --seed 17

uv run mfh verify-reviewed-splits \
  artifacts/splits/triviaqa-reviewed \
  artifacts/contamination/triviaqa-ood-manual-review-result \
  artifacts/contamination/triviaqa-ood-manual-review \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2 \
  --expected-review-result-manifest-digest REVIEW_RESULT_DIGEST \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6 \
  --expected-split-manifest-digest REVIEWED_SPLIT_DIGEST \
  --steer 30000 --controller 5000 --dev 5000 --test 5000 --seed 17
```

The reviewed clean source contains both the automatic removals and every
human-confirmed overlap. Do not reuse `artifacts/splits/triviaqa-auto-clean` for
E1 or later phases; it is provisional evidence created before manual review. The
reviewed splitter replays its exact-duplicate exclusion and atomically publishes
`curation-report.json` plus a digest-bound manifest beside five TriviaQA split
files and the canonical SimpleQA/AA evaluation schedules.
The pinned TriviaQA rows currently expose no populated named-entity field; this
workflow proves normalized-question and accepted-alias disjointness, not
NER-derived entity disjointness.

For E1, call `authorize_reviewed_split_bundle(...)` with the same live paths and
external digests immediately before creating the phase ledger. Pass the returned
`VerifiedReviewedSplits` as `verified_reviewed_splits` to
`PhaseRunLedger.create`; the `deduplicated_splits` input must be the exact same
`artifacts/splits/triviaqa-reviewed` directory. Creation rejects a missing
capability, a changed bundle, a provisional split, or a TriviaQA question
schedule that is not exactly the authorized T-controller, T-dev, or T-test
partition selected by the E1 conditions. The same authorization packages the
exact canonical SimpleQA Verified and AA-Omniscience schedules: E1 conditions
must use `simpleqa-eval` and `aa-eval` respectively, with question IDs exactly
matching those two reviewed-bundle artifacts.

Use `mfh overlap SOURCE TARGET` for exact and normalized n-gram contamination
checks. Never use T-test, SimpleQA, or AA results to tune components.

The source snapshot verifier rejects wrong revisions, sizes, hashes, schemas, and
question membership when confirmatory question bundles are built.

## 5. Scientific phase lifecycle

Most phases follow the same immutable lifecycle through the Python API in
`mfh.experiments.runner`. E3 uses the seven-stage exception described below.

1. Load `StudyProtocol` from `configs/experiments/phases.yaml`.
2. Build exact `EvaluationCondition` objects. Use
   `expand_factorial_conditions` for declared factorial phases.
3. Construct a `PhaseRunContract` with exact question IDs, input artifact
   fingerprints, prerequisite completion digests, and required gates.
4. Call `contract.assert_matches_study(study)`. This enforces the phase's model,
   benchmark, partition, prompt, method, factorial, and question-count rules.
5. Create the ledger with `PhaseRunLedger.create(...)`, passing live input paths
   and live prerequisite run directories. Creation independently fingerprints
   inputs and verifies every prerequisite completion.
6. Iterate `ledger.iter_pending()`, generate one canonical `GenerationRecord` per
   pending condition/question pair, and append records with `ledger.checkpoint()`.
7. Produce raw evidence for each declared phase gate and evaluate it with
   `ledger.evaluate_gate(gate_name, evidence_path)`.
8. Finalize only with the exact set of passing `GateResult` objects. For E0,
   first call `authorize_e0_completion_receipt(...)` with all live MLX runtime,
   review, cohort, and external-digest inputs, then pass the resulting
   `VerifiedE0CompletionReceipt` as `verified_e0_completion`; a receipt path or
   caller-declared digest is deliberately insufficient.
9. Reopen and call `verify_complete()` before using the phase downstream.

Operational checks are available without mutating a run:

```bash
uv run mfh phase-progress "$STUDY/runs/E5" configs/experiments/phases.yaml
uv run mfh verify-phase "$STUDY/runs/E5" configs/experiments/phases.yaml
```

Shards are append-only and resumable. A finalized ledger binds the contract,
record set, gate results, and packaged gate evidence. Unexpected files, symlinks,
changed shards, changed inputs, forged prerequisites, and incomplete matrices are
rejected.

### Confirmatory freezes

E9 and E10 use operator commands that derive every packaged question,
component, grader, prerequisite, and provenance field from verified upstream
artifacts. Do not hand-edit a confirmatory runbook and do not call the low-level
bundle serializers directly.

For E9, atomically stage the external grader/reviewed/source inputs, freeze the
evaluator source, and let `freeze-e9-inputs` write and preflight the complete
runbook:

```bash
uv run mfh stage-e9-inputs --help
uv run mfh freeze-execution-snapshot --help
uv run mfh freeze-e9-inputs --help
```

For E10, the high-level operator derives the E0--E9 selection provenance,
prepares and verifies the 10,000-row development-only early-probe capture, fits
the registered eight-candidate probe grid, writes M6 and all eleven freeze
fields, and atomically publishes the E10 runbook:

```bash
uv run mfh prepare-e10-freezes --help
uv run mfh run-e10-early-probe --help
uv run mfh verify-e10-early-probe --help
uv run mfh finalize-e10-freezes --help
```

E10 ledger creation independently re-derives the component provenance, replays
the early-probe selection, and rejects any other controller, SAE, protected
direction, prompt, threshold, probe, or prerequisite chain. The early-probe
workflow is development selection only: it never reads E9 or E10 outcomes when
fitting or ranking candidates.

E10 has an external, write-once reservation registry bound to the study protocol.
Creating the ledger consumes the one-shot reservation. A failed or disappointing
run does not authorize tuning and rerunning; a new scientific protocol is needed.

### Native M4 Max confirmatory execution

E9 and E10 have a complete operator-facing lifecycle. The runbook is a strict,
secret-free JSON file: it contains only the phase, exact config/snapshot paths,
frozen inputs, prerequisite ledgers, output paths, and seed. Unknown fields are
rejected, so an API key or private signing key cannot accidentally become part of
the frozen runbook.

The freeze operators create the runbooks; the operator only preflights and runs
them. The exact source paths and complete cross-device command sequence are in
the repository [README](../README.md#11-freeze-e9-inputs-and-run-robustness-diagnostics).
The common lifecycle is:

```bash
uv run mfh preflight-confirmatory "$RUNBOOK"
uv run mfh prepare-confirmatory "$RUNBOOK"
uv run mfh run-confirmatory "$RUNBOOK" \
  --env-file .env --checkpoint-size 1 --limit 100
uv run mfh verify-confirmatory "$RUNBOOK"
uv run mfh finalize-confirmatory "$RUNBOOK"
uv run mfh verify-confirmatory "$RUNBOOK"
```

For E10 only, add `--authorize-e10-one-shot` to `prepare-confirmatory` after
reviewing and externally recording the preflight contract digest. Preflight does
not consume the reservation. Preparation does.

Put `OPENROUTER_API_KEY` and the Ed25519 `MFH_EXECUTION_PRIVATE_KEY` in the local
`.env`; neither value is printed or serialized. On the M4 Max, keep checkpoint
size at one so every completed row is immediately durable. Use `--limit` to bound
one session and rerun the identical command until no rows remain:

The run command reconstructs the live MLX runtime from the frozen attestation's
seed and research provenance, then requires its identity and signing public key
to match the packaged confirmatory grader bundle before generating a row.
Generation is resumable; finalization is refused until the exact matrix is
complete.

If an E10 gate fails, finalization publishes the immutable falsification receipt;
it does not create authority to alter M6 or reserve another run.

The development E6--E8 computations use the same runtime-owned execution
surface before confirmatory preparation. E6 M0 is generated inside
`execute_and_bind_e6_likelihood`; E6 M3 uses
`execute_e6_adaptive_generation`; E7 coordinate, causal, and interpretability
rows use `execute_coordinate_screen_generation` and `execute_e7_generation`;
E8 fixed and adaptive rows use `execute_e8_generation` and
`execute_e8_adaptive_generation`. Their phase finalizers replay the attested
unified-memory envelope and the exact prompt-feature or teacher-forced auxiliary
peak before admitting an E9 prerequisite.

## 6. E0–E10 execution map

The phase configuration is the executable checklist. The main implementation
surfaces are:

| Phase | Operator action | Main implementation |
| --- | --- | --- |
| E0 | Validate the exact Qwen MLX checkpoint, chat template, deterministic decoding, runtime identity, and hook sites | native MLX runner and live preflight receipt |
| E1 | Run M0 prompt factorial and freeze C/P/I/A/U records | grading, metrics, risk curves |
| E2 | Capture prompt-end features; train and calibrate C/I/A probes; evaluate separability gate | probes and feature schemas |
| E3 | Build residual/post-MLP centroids; sweep layer, alpha, scope; run causal controls | static steering and controls |
| E4 | Screen feasible CAA/ITI/ACT/SADI/TruthX baselines and freeze promotion evidence | empirical operating-point matching |
| E5 | Fit TriviaQA-only vector banks, routers, dynamic alpha, and layer policy | adaptive controller |
| E6 | Run paired transitions and teacher-forced gold/abstention likelihoods | transitions and runtime scoring |
| E7 | Train coordinate and SAE variants on separate activation corpora; require two-seed stability and per-feature causality | sparse and SAE stability modules |
| E8 | Build protected behavior subspaces and covariance-aware directions; compare at matched measured risk/coverage | protected steering and non-inferiority |
| E9 | Run the frozen one-model × three-benchmark × prompt × method matrix | factorial ledger plus confirmatory statistics |
| E10 | Execute frozen M6 once, including early re-evaluation and output gate | composite policy and one-shot ledger |

AA-Omniscience also has a frozen auxiliary baseline track: 600 M0 generations
under the exact released `P-AA-official` answerer prompt, followed by the released
AA scoring rubric. It is intentionally outside E1's 19,800-row controlled prompt
factorial and E9's 118,800-row confirmatory matrix. Its only registered comparison
is paired `P-AA-official` versus controlled `P0-neutral` within M0; only the
official track is leaderboard-comparable.

E3 is deliberately not a single generic ledger. Execute and verify its
`geometry`, `alpha`, `scope`, `controls`, `cross-prompt`, `P3-diagnostic`, and
`final` append-only stage stores, then call `finalize_e3_phase`. The completed E3
phase directory contains a compact `analysis-surface.json` bound to every stage
record-chain head, condition metric, gate, and phase manifest. Later analysis
must receive that completed phase directory as its `E3` input; passing one stage
store or fabricating a generic `PhaseRunLedger` is rejected.

Use the E3 operator state machine; do not call the component APIs by hand. It
reconstructs every stage asset from the frozen construction and predecessor
selection, materializes the label-shuffled, norm-matched random, and Gaussian
controls before they can be evaluated, and performs exactly one durable action
per `advance-e3` invocation.

```bash
E3_RUNBOOK="$STUDY/runbooks/E3.json"
E3_OUTPUT_ROOT="$STUDY/E3-operator"

uv run mfh write-e3-runbook "$E3_RUNBOOK"
```

Edit only the absolute paths in the generated secret-free JSON. In particular,
`source_runtime_plan` must be the completed native-MLX E2 capture plan containing
the exact `runtime_identity`; `input_artifacts` must name `E1_outcome_labels` and
`activation_feature_schemas`; and `prerequisite_runs` must name the completed E1
and E2 phase directories. Keep the frozen 5,120 hidden width, construction
fingerprints, and checkpoint policies unchanged.

Run the read-only preflight before loading Qwen:

```bash
uv run mfh preflight-e3 "$E3_RUNBOOK"
```

Then invoke the resumable advance command repeatedly. The default budget of 4,096
performs at most 4,096 new model requests in the active construction,
shuffled-control, or
evaluation stage; non-compute invocations atomically prepare/finalize one
artifact or write one deterministic selection.

```bash
uv run mfh advance-e3 "$E3_RUNBOOK" --request-budget 4096
```

The reported actions progress through construction, static-vector publication,
`geometry` selection, `alpha` selection, `scope` selection, control
materialization, all four remaining stage stores, and the terminal E3 phase. The
fixed output layout under `output_root` is:

```text
construction/
vectors/
selections/{geometry,alpha,scope}.json
controls/{shuffle-work,shuffled-vectors,fixed}/
stages/{geometry,alpha,scope,controls,cross-prompt,P3-diagnostic,final}/
phase/
```

Once `advance-e3` reports `finalized-phase`, replay the entire lifecycle:

```bash
uv run mfh verify-e3 "$E3_RUNBOOK"
```

Only `output_root/phase` is the E3 prerequisite supplied to later phases. The
verifier is read-only and refuses to create a missing terminal phase.

The runtime captures only selected layers and token positions. Centroid directions
use online accumulation; SAE training uses separate train/validation feature
schemas. Adaptive M3/M6 records must include signed decision receipts proving that
the frozen probe, router, vector bank, alpha, layer, and token scope were actually
used rather than merely named in metadata.

### 6.1 Build the native-MLX M2 CAA artifact

Run this after the E3 construction work and its final E3 vector bundle both
verify. M2 uses only the original incorrect P0 generations from the frozen E3
construction. Each negative answer is paired with the first gold alias for the
identical question and semantic group; both responses are teacher-forced through
Qwen and their residual block-output means are differenced at the seven E2
layers. The pair schedule, activation digests, checkpoint chain, runtime sessions,
peak unified memory, and final normalized vectors are all replayable.

The commands below are intentionally resumable for the 48 GiB M4 Max. `T-steer`
is the reviewed question file used by the E3 construction, and `E3-CONSTRUCTION`
is that construction work directory (not merely the final vector bundle).

```bash
E3_CONSTRUCTION="$E3_OUTPUT_ROOT/construction"
T_STEER="artifacts/splits/triviaqa-reviewed/T-steer.jsonl"

uv run mfh prepare-m2-caa \
  "$E3_CONSTRUCTION" "$T_STEER" "$STUDY/work/E4-M2-CAA"

uv run mfh run-m2-caa \
  "$E3_CONSTRUCTION" "$T_STEER" "$STUDY/work/E4-M2-CAA" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  --request-budget 64

uv run mfh verify-m2-caa-work \
  "$E3_CONSTRUCTION" "$T_STEER" "$STUDY/work/E4-M2-CAA" \
  --require-complete

uv run mfh finalize-m2-caa \
  "$E3_CONSTRUCTION" "$T_STEER" "$STUDY/work/E4-M2-CAA" \
  "$STUDY/frozen/E4-M2-CAA"
```

Repeat `run-m2-caa` with a bounded request budget until verification reports all
pairs complete. Record the returned digest as `M2_CAA_MANIFEST_DIGEST`, then do a
portable replay without loading Qwen:

```bash
uv run mfh verify-m2-caa-artifact "$STUDY/frozen/E4-M2-CAA" \
  --expected-manifest-digest "$M2_CAA_MANIFEST_DIGEST"
```

E4 method-policy creation reads the direction hash, unit norm, and reference RMS
from this artifact. Callers cannot substitute those values. M2 interventions use
standardized alpha (`raw alpha = standardized alpha × reference RMS`) at
`block_output`; M1 is separately restricted to E3 `post_mlp` centroids.

### 6.2 Run the native-MLX E4 baseline screen

E4 runs only the methods that passed the frozen runtime capability screen. The
required native comparison is M1, M2, and an ACT/SADI-style adaptive baseline
that uses the selected calibrated E2 C/I/A probe to gate the M2 direction and
intensity. ITI and TruthX remain in the capability report with structured
infeasibility receipts when compatible per-head or autoencoder hooks are absent.
Those receipts record every candidate artifact path, its digest when present,
the attempted MLX hook capability, and machine-readable failure codes. If local
implementations or a compatible TruthX autoencoder exist, pass them to
`prepare-e4-mlx-screen` with `--iti-implementation`,
`--truthx-implementation`, and `--truthx-autoencoder`; they cannot be skipped by
an unexplained boolean assertion.

Use the one Ed25519 execution key created in README section 3. The key signs
every fixed and adaptive runtime receipt from E4 through E10, is never copied
into an artifact, and must not be printed in a terminal transcript. Do not
generate a phase-specific replacement key:

```bash
mkdir -p "$STUDY/secrets"
umask 077
test -s "$STUDY/secrets/execution-private-key.hex"
```

Set the frozen upstream paths. `E2_PROBES` is the terminal selected E2 probe
bundle, `E3_VECTORS` is the exact `E3_static_vectors` output named by the
completed E3 phase, and the layer values are the independently frozen M1 and M2
operating layers. They are not selected using E4 outcomes.

```bash
E2_PROBES="$STUDY/frozen/E2-probes"
E2_WORKSPACE="$STUDY/work/E2-workspace"
E2_PHASE="$STUDY/runs/E2"
E3_VECTORS="$E3_OUTPUT_ROOT/vectors"
E3_PHASE="$E3_OUTPUT_ROOT/phase"
M2_CAA="$STUDY/frozen/E4-M2-CAA"
E4_ACT="$STUDY/frozen/E4-ACT-SADI"
E4_SETUP="$STUDY/frozen/E4-mlx-setup"
E4_LEDGER="$STUDY/runs/E4"
E4_KEY="$STUDY/secrets/execution-private-key.hex"
T_DEV="artifacts/splits/triviaqa-reviewed/T-dev.jsonl"
E3_SCOPE_SELECTION="$E3_OUTPUT_ROOT/selections/scope.json"
M1_LAYER="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["M1-P"]["layer"])' "$E3_SCOPE_SELECTION")"
M2_LAYER="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["M1-R"]["layer"])' "$E3_SCOPE_SELECTION")"
```

`M1_LAYER` and `M2_LAYER` above are read from the terminal E3 development-only
scope selection; `31` is not a protocol constant. E6--E8 use the same frozen
`M1-P` layer through each runbook writer's required `--m1-layer` argument.
Never replace these values using an E4--E10 outcome.

The ACT builder first replays the full E2 activation workspace, complete probe
screen/final grids, selected gate, and completed E2 phase. It then cross-binds
that completion to the E2 prerequisite and probe input recorded by E3. First
package the calibrated adaptive comparison and freeze all six conditions
(two prompts by three feasible methods):

```bash
uv run mfh build-e4-act-baseline \
  "$E2_PROBES" "$E2_WORKSPACE" "$E2_PHASE" "$M2_CAA" "$E4_ACT" \
  --intervention-layer "$M2_LAYER"

uv run mfh prepare-e4-mlx-screen \
  "$T_DEV" configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E2_PROBES" "$E3_VECTORS" "$M2_CAA" "$E4_ACT" "$E3_PHASE" \
  "$E4_SETUP" "$E4_LEDGER" \
  --execution-key-file "$E4_KEY" \
  --m1-layer "$M1_LAYER" --m2-layer "$M2_LAYER"
```

Run in bounded sessions on the M4 Max. Each checkpoint is append-only, so repeat
the run command until verification reports 12,000 of 12,000 rows (2,000
questions × 6 conditions):

```bash
uv run mfh run-e4-mlx-screen \
  "$E4_SETUP" "$E4_LEDGER" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  --execution-key-file "$E4_KEY" \
  --request-budget 64 --checkpoint-rows 8

uv run mfh verify-e4-mlx-screen \
  "$E4_SETUP" "$E4_LEDGER" --require-complete
```

The runner rejects a live runtime that differs from the exact Qwen/MLX identity
used for M2, a receipt signed by another key, a missing activation edit, or a
peak-memory observation above 48 GiB. After complete portable verification,
derive the coverage-constrained promotion and terminally freeze E4:

```bash
uv run mfh finalize-e4-mlx-screen \
  "$E4_SETUP" "$E4_LEDGER" \
  "$STUDY/frozen/E4-promotion.json" \
  "$STUDY/evidence/E4-promotion-decision.json"
```

The registered gate requires M2 to remain in the eligible comparison and only
promotes methods that beat M1 without losing more than five percentage points of
coverage in either prompt stratum.

### 6.3 Capture the native E5 controller-fitting tensors

E5 reuses the exact calibrated E2 C/I/A probes for controller risk, but learns
its routed vector banks only from disjoint `T-steer` rows. The capture replays
the complete E3 construction and keeps only P0 rows with native C or I outcomes
and a non-empty response. For each eligible question it captures the final
prompt token at the registered controller-input layers and the mean
teacher-forced response activation at all three candidate intervention layers.
Prompt and response arrays enter one atomic float16 shard, so an interruption
cannot leave them positionally misaligned.

Use the M1 layer/site chosen by the frozen E3 development stages. The two- and
three-layer candidates are likewise development-frozen inputs; do not choose
them from E5 ablation results. The E2 controller-input layers are derived from
the selected E2 C/I/A probe and its two nearest registered neighbors.

```bash
E5_CAPTURE="$STUDY/work/E5-fit-capture"
E5_KEY="$STUDY/secrets/execution-private-key.hex"
E5_SPLITS="$STUDY/frozen/E5-controller-splits"
E5_SPLIT_MANIFEST_DIGEST="4265de0ec0d4991882bb91bc8cd813c0992709b3705b7823ffe977cba44da437"
T_STEER="artifacts/splits/triviaqa-reviewed/T-steer.jsonl"
T_CONTROLLER_SOURCE="artifacts/splits/triviaqa-reviewed/T-controller.jsonl"

if [ ! -e "$E5_SPLITS" ]; then
  uv run mfh materialize-e5-controller-splits \
    "$T_CONTROLLER_SOURCE" "$E5_SPLITS"
fi

uv run mfh verify-e5-controller-splits \
  "$T_CONTROLLER_SOURCE" "$E5_SPLITS" \
  --expected-manifest-digest "$E5_SPLIT_MANIFEST_DIGEST"

uv run mfh prepare-e5-fit-capture \
  "$E5_CAPTURE" "$E3_CONSTRUCTION" "$T_STEER" \
  "$E2_PROBES" "$E2_WORKSPACE" "$E3_VECTORS" \
  "$STUDY/frozen/mlx-preflight.json" \
  --execution-key-file "$E5_KEY" \
  --fixed-best-layer "$M1_LAYER" \
  --two-layer-candidates 31 32 \
  --three-layer-candidates 16 31 32 \
  --intervention-site post_mlp \
  --shard-rows 64

uv run mfh run-e5-fit-capture \
  "$E5_CAPTURE" "$E3_CONSTRUCTION" "$T_STEER" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  --execution-key-file "$E5_KEY" \
  --request-budget 2048

uv run mfh verify-e5-fit-capture \
  "$E5_CAPTURE" "$E3_CONSTRUCTION" "$T_STEER" \
  --execution-key-file "$E5_KEY" \
  --require-complete
```

Repeat the run command until complete. A run holds the capture's process lock,
removes only abandoned atomic shard stages, verifies the existing chain once,
and then advances the in-memory verified head after each append. This keeps a
multi-shard session linear in artifact size. Every shard records signed session,
wall-time, lock-identity, process, and peak-memory evidence and is Ed25519-signed by the key
whose public half was frozen in the plan, hash-chained to its predecessor, and
bound to the exact E3 generation record, rendered prompt, model/runtime
identity, E2 probe bundle, E3 vector bundle, split manifest, layer geometry, and
48 GiB peak-memory ceiling. A newly generated key cannot self-authorize altered
fit inputs. After complete verification, close the MLX runtime and unload Qwen
before materializing the float32 controller datasets or fitting routers.

The controller subdivision command replays the exact E2 seed-17 semantic-group
assignment and atomically freezes 4,000 training rows plus 1,000 calibration
rows under the active Qwen study namespace. Its verifier requires a complete,
disjoint union of the original ordered 5,000-row `T-controller` source. The E5
recipe accepts only registered Qwen layers `(16, 31, 32, 47, 48, 57, 63)` and
freezes the fixed layer plus its one/two nearest registered neighbours. For the
illustrated frozen M1 layer 31, the only valid pairs are `31 32` and
`16 31 32`; if E3 selects another M1 layer, derive the corresponding nearest
sets before preparing E5 rather than copying these example values.

The fitter additionally requires exact ordered IDs, semantic groups, and
outcomes across all three controller-input compositions; it rejects any
question or semantic-group overlap between `T-controller-train` and `T-steer`.
The exact serialized E2 risk-probe artifact and its in-memory tensors are both
signed into the capture attestation. Save controllers only through
`save_e5_fitted_controller`; the resulting `e5-fit-provenance.json` is mandatory
for every E5 binding and preserves the capture attestation, fit recipe, runtime,
E2/E3 inputs, layer-label receipt, and risk-probe identities.

After the T-steer capture is complete, generate the supervised labels for the
two- and three-layer routers. Each T-controller-train question is replayed once
at each of the three development-frozen candidate layers with the exact E3
M1-R/P0 direction, the recipe's RMS-standardized maximum alpha, and final-prompt
timing. Labels rank counterfactual outcomes as correct, then abstention, then
incorrect; ties prefer the frozen fixed-best layer and then recipe order.

```bash
E5_LABELS="$STUDY/work/E5-layer-labels"
E5_CONTROLLERS="$STUDY/artifacts/E5-controller-grid"
T_CONTROLLER="$E5_SPLITS/T-controller-train.jsonl"

uv run mfh prepare-e5-layer-labels \
  "$E5_LABELS" "$E5_CAPTURE" "$E3_CONSTRUCTION" \
  "$T_STEER" "$T_CONTROLLER" "$E2_PROBES" "$E2_WORKSPACE" "$E3_VECTORS" \
  --execution-key-file "$E5_KEY" \
  --shard-rows 64

uv run mfh run-e5-layer-labels \
  "$E5_LABELS" "$E5_CAPTURE" "$E3_CONSTRUCTION" \
  "$T_STEER" "$T_CONTROLLER" "$E2_PROBES" "$E2_WORKSPACE" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  --execution-key-file "$E5_KEY" \
  --request-budget 2048

uv run mfh verify-e5-layer-labels \
  "$E5_LABELS" "$E5_CAPTURE" "$E3_CONSTRUCTION" \
  "$T_STEER" "$T_CONTROLLER" "$E2_PROBES" "$E2_WORKSPACE" \
  --execution-key-file "$E5_KEY" \
  --require-complete

uv run mfh fit-e5-controller-grid \
  "$E5_CONTROLLERS" "$E5_CAPTURE" "$E5_LABELS" "$E3_CONSTRUCTION" \
  "$T_STEER" "$T_CONTROLLER" "$E2_PROBES" "$E2_WORKSPACE" \
  --execution-key-file "$E5_KEY"

uv run mfh verify-e5-controller-grid "$E5_CONTROLLERS"
```

Repeat the native layer-label run until all 15,000 default candidate executions
are complete. Then close and unload MLX before fitting. The fitter accepts only
the verified label object, rehashes its artifact, and cross-checks its plan and
chain head against the exact fit capture. It fits 324 timing-independent
controllers for the default 972-arm grid, maps the three timing variants to the
same fitted controller, and atomically packages every arm with mandatory signed
fit lineage.

Package those 972 arm-to-controller bindings and freeze the native developmental
ablation before loading MLX. The default implicit schedule contains 9,730,000
generations: M1 plus 972 adaptive arms, crossed with P0/P2 and all 5,000 ordered
T-dev questions. It is computed by sequence number and is never serialized or
held in memory as a giant schedule.

```bash
E5_BINDINGS="$STUDY/artifacts/E5-controller-bindings"
E5_ABLATION="$STUDY/work/E5-native-ablation"

uv run mfh package-e5-controller-bindings \
  "$E5_BINDINGS" "$E5_CONTROLLERS" \
  --execution-key-file "$E5_KEY"

uv run mfh verify-e5-controller-bindings "$E5_BINDINGS"

# Replace 1.0 with the measured end-to-end generations/second from an M4 Max
# pilot. At 1.0 generation/second the exact grid is about 112.6 continuous days.
uv run mfh estimate-e5-native-ablation \
  --generations-per-second 1.0 \
  --checkpoint-opens-per-second 2.0 \
  --verification-rows-per-second 10000 \
  --request-budget 8192

uv run mfh prepare-e5-native-ablation \
  "$E5_ABLATION" "$E4_SETUP/screen-receipt.json" "$E5_BINDINGS" \
  "$E4_SETUP/policies/m1.json" "$E3_VECTORS" "$E5_CAPTURE" \
  "$STUDY/frozen/mlx-preflight.json" \
  --execution-key-file "$E5_KEY" \
  --acknowledge-exact-grid-records 9730000 \
  --shard-rows 1024

uv run mfh run-e5-native-ablation \
  "$E5_ABLATION" configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  --execution-key-file "$E5_KEY" \
  --request-budget 8192

uv run mfh verify-e5-native-ablation \
  "$E5_ABLATION" --execution-key-file "$E5_KEY" --require-complete

uv run mfh finalize-e5-native-ablation \
  "$E5_ABLATION" --execution-key-file "$E5_KEY"
```

Repeat the run command on the 48 GiB M4 Max until complete. Each session takes
the process lock, removes only abandoned atomic stages, authenticates the latest
signed append head, opens bounded binding metadata, validates only the controller
arm reached in that session, and then advances the head incrementally. It never
rescans historical row payloads or reloads all 972 controllers during resume.
Use `verify-e5-native-ablation --structural-only` for a quick operator check;
the required pre-finalization verification remains a signed semantic-transcript
replay. Adaptive rows bind the exact prompt-feature schema and feature
commitment, retain the calibrated C/I/A scores plus vector- and layer-routing
weights, and recompute the selected direction, layer/site, alpha, timing, and
action from the frozen controller. Hook evidence retains pre/post/delta
commitments plus per-application norms, direction products, and residuals, so
the verifier checks the material edit without storing full 5,120-dimensional
activation deltas for 9.73 million rows. M1 rows replay the exact promoted E4
policy and E3 M1-R/P0 geometry. Every M1 and M3 record, shard, and final receipt
is Ed25519-signed by the external execution trust root. Finalization atomically
creates `$E5_ABLATION/final/records.jsonl` and its signed receipt; the selection
reader verifies the signature on every row, including the M1 matched-budget
reference.

The estimate and the exact-record acknowledgement are mandatory. Measure the
generation rate with the final snapshot, prompt length, 48-token limit, and
intervention hooks; also measure checkpoint opens/second and the slowest rows/second
across semantic replay, final materialization, and selection parsing. The estimate
includes every session checkpoint, three mandatory full-row passes, and one full
signed-manifest-entry pass. If the returned duration is
not operationally feasible, stop and record a protocol amendment; do not reduce
the 972-arm grid or T-dev cohort after seeing any ablation result.

After native finalization, unload Qwen. Derive the matched selection and promote
the already executed M1 plus selected-M3 rows into the ordinary E5 phase ledger.
Promotion performs zero model generations: it semantically replays only the
selected signed native rows, converts their exact outputs, C/I/A scores,
controller actions, routing weights, alpha, hook evidence, token counts, and
end-to-end latency, and signs the standard adaptive ledger receipts with the
same external key. The resulting schedule is exactly 20,000 records: two
methods by two prompts by 5,000 T-dev questions.

```bash
E4_PROMOTION="$STUDY/frozen/E4-promotion.json"
E5_SELECTION="$STUDY/frozen/E5-selection"
E5_LEDGER="$STUDY/runs/E5"
E5_PHASE="$STUDY/frozen/E5-phase"

uv run mfh derive-e5-selection \
  "$E5_SELECTION" "$E5_ABLATION" "$E5_BINDINGS" \
  "$E2_PROBES" "$E3_VECTORS" "$E4_PROMOTION" \
  --execution-key-file "$E5_KEY"

uv run mfh verify-e5-selection \
  "$E5_SELECTION" "$E5_ABLATION" \
  --execution-key-file "$E5_KEY"

uv run mfh prepare-e5-phase-ledger \
  "$E5_LEDGER" "$E5_SELECTION" "$E5_ABLATION" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$E2_PHASE" "$E3_PHASE" "$E4_LEDGER" \
  --execution-key-file "$E5_KEY"

# Repeat until completed_records equals 20000. This command never loads MLX.
uv run mfh promote-e5-phase-records \
  "$E5_LEDGER" "$E5_SELECTION" "$E5_ABLATION" \
  --execution-key-file "$E5_KEY" \
  --request-budget 2000 --checkpoint-rows 250

uv run mfh verify-e5-phase-ledger \
  "$E5_LEDGER" "$E5_SELECTION" "$E5_ABLATION" \
  --execution-key-file "$E5_KEY" --require-complete

uv run mfh finalize-e5-phase \
  "$E5_PHASE" "$E5_LEDGER" "$E5_SELECTION" "$E5_ABLATION" \
  --execution-key-file "$E5_KEY"

uv run mfh verify-e5-phase \
  "$E5_PHASE" "$E5_LEDGER" "$E5_SELECTION" "$E5_ABLATION" \
  --execution-key-file "$E5_KEY"
```

The selection directory contains canonical `selection.json` plus an Ed25519
receipt bound to the native plan, final chain head, 9.73-million-record digest,
all 972 controller bindings, and E2/E3/E4 upstream fingerprints. Native
finalization has already performed a complete semantic scan. Later resumable
promotion sessions verify the signed plan, source artifacts, every shard
manifest in the chain, and the signed selection/finalization receipts; they hash
and semantically replay every shard that contributes an M1 or selected-M3 row.
Terminal E5 verification additionally replays the full selection and all four
registered matched-coverage, matched-abstention, matched-norm, and
matched-latency gates. A failed gate freezes E5 as falsified and cannot satisfy
downstream prerequisites.

### 6.4 Run the auxiliary AA official track

Run this only after E1 is complete. The track binds the exact E1 completion,
reviewed Public-600 questions (including domain/topic metadata), official prompt,
Qwen snapshot, runtime receipt, grader route, and fixed-seed randomized schedule.
Generation and OpenRouter evidence are append-only and resume only from the
external checkpoint.

```bash
uv run mfh prepare-aa-official \
  artifacts/splits/triviaqa-reviewed artifacts/graders/e1-frozen-v2 \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$STUDY/work/aa-official" "$STUDY/runs/E1" "$STUDY/runs/E0" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST"

uv run mfh run-aa-official \
  artifacts/splits/triviaqa-reviewed artifacts/graders/e1-frozen-v2 \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$STUDY/work/aa-official" "$STUDY/runs/E1" "$STUDY/runs/E0" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST" \
  --checkpoint-file "$STUDY/checkpoints/aa-official.json"
```

For later invocations add `--resume`. After all 600 rows are present, freeze the
portable terminal artifact:

```bash
uv run mfh finalize-aa-official \
  artifacts/splits/triviaqa-reviewed artifacts/graders/e1-frozen-v2 \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$STUDY/work/aa-official" "$STUDY/runs/E1" "$STUDY/runs/E0" \
  "$STUDY/auxiliary/aa-official" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST"
```

Record the returned identity as `AA_OFFICIAL_MANIFEST_DIGEST`. Final analysis
requires it and replays every generation, grader receipt, metric, and paired
neutral transition. Exhausted provider attempts remain in a separate chained
failure journal and never enter the 600 scorable rows.

Before analysis, replay the terminal artifact against the live frozen inputs:

```bash
uv run mfh verify-aa-official \
  artifacts/splits/triviaqa-reviewed artifacts/graders/e1-frozen-v2 \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$STUDY/work/aa-official" "$STUDY/runs/E1" "$STUDY/runs/E0" \
  "$STUDY/auxiliary/aa-official" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST" \
  --expected-manifest-digest "$AA_OFFICIAL_MANIFEST_DIGEST"
```

### 6.5 Freeze and run the preregistered robustness diagnostics

The E9 freeze operator creates the signed diagnostic plan. The high-level
drivers then own question lookup, live MLX reconstruction, OpenRouter grading,
fit-data selection, source/adapted component packaging, threshold calibration,
signed evidence, atomic append, resume, and verification. Operators must not
write integration Python.

The exact cross-device commands and source paths are in the
[README](../README.md#11-freeze-e9-inputs-and-run-robustness-diagnostics). The
driver sequence is:

```bash
uv run mfh verify-robustness-plan "$ROBUSTNESS_PLAN"
uv run mfh create-robustness-results \
  "$ROBUSTNESS_RESULTS" "$ROBUSTNESS_PLAN"
uv run mfh prepare-robustness-execution \
  "$ROBUSTNESS_EXECUTION" "$ROBUSTNESS_RESULTS" \
  "$E9_RUNBOOK" "$E3_CONSTRUCTION" --shard-rows 16

# Repeat until the verifier reports complete.
uv run mfh run-robustness-rq1-capture \
  "$ROBUSTNESS_EXECUTION" "$ROBUSTNESS_RESULTS" \
  "$E9_RUNBOOK" "$E3_CONSTRUCTION" \
  --env-file .env --limit 256
uv run mfh verify-robustness-rq1-capture \
  "$ROBUSTNESS_EXECUTION" "$ROBUSTNESS_RESULTS" \
  "$E9_RUNBOOK" "$E3_CONSTRUCTION" \
  --env-file .env --require-complete

# Repeat both bounded commands until progress is 36,000/36,000 and 60/60.
uv run mfh run-robustness-prompts \
  "$ROBUSTNESS_RESULTS" "$E9_RUNBOOK" \
  --env-file .env --limit 100
uv run mfh run-robustness-rq1 \
  "$ROBUSTNESS_EXECUTION" "$ROBUSTNESS_RESULTS" \
  "$E9_RUNBOOK" "$E3_CONSTRUCTION" \
  "$E2_WORKSPACE" "$E2_PROBES" \
  "$E5_CAPTURE" "$E5_LABELS" "$T_CONTROLLER" \
  --env-file .env --limit 1

uv run mfh finalize-robustness-results "$ROBUSTNESS_RESULTS"
uv run mfh verify-robustness-results \
  "$ROBUSTNESS_RESULTS" --require-complete
```

The 30,000-row P0 T-steer capture is signed, append-only, and shared across all
folds. All C/I/A source identities remain in the capture; only the registered
C/I projection used for vector fitting excludes abstentions. The sixty RQ1
tasks enforce their exact semantic-fold inventories and registered adaptation
scope. The 36,000 paraphrase rows use the exact E9 schedule, component bundle,
grader bundle, evaluator snapshot, and execution key. Neither diagnostic may
influence E9 or E10 component selection.

## 7. Grading, safety, and reviewed language evidence

Official grader configurations bind the upstream source artifact, exact prompt,
model revision, label mapping, and failure behavior. Verify a downloaded source
artifact before grading:

```bash
uv run mfh verify-grader \
  configs/graders/aa-omniscience-public.yaml \
  /absolute/path/AA-Omniscience-Public/README.md
```

The safety scorer is deterministic and signed. Freeze its implementation identity
and execution public key before the phases that depend on it:

```bash
uv run mfh freeze-safety-scorer \
  "$STUDY/frozen/side-effect-scorer.json" \
  LOWERCASE_HEX_ED25519_PUBLIC_KEY
```

Language-consistency evaluation accepts only a packaged, human-reviewed
translation suite with reviewer signatures:

```bash
uv run mfh build-language-suite \
  "$STUDY/frozen/language-suite" \
  artifacts/source/triviaqa/d2ff7f468d3642dbd33123596331950db8a63d0e/rc.nocontext/train-00000-of-00001-e93ee6c1ba181971.parquet \
  "$STUDY/operator-inputs/language-translations.jsonl" \
  "$STUDY/operator-inputs/language-reviewer-public-keys.json"
uv run mfh verify-language-suite "$STUDY/frozen/language-suite"
```

Automated side-effect labels are rebound to the exact response text. A copied
safe label, a false signed score, or a detached response is rejected.

Every SimpleQA E9/E10 row also carries a deterministic, response-bound hedging
receipt. The official released grader remains authoritative for C/I/A; final
analysis separately reports hedging rate, punting rate, attempt rate, incorrect
attempted rate, the preregistered risk–coverage curves, and complete M0-to-method
transition matrices.

For each reviewed language-suite response, the native scorer records the
alias-aware detected language, correct-output-language decision, non-target
script-token rate, code-switching decision, factual outcome, and abstention
decision. The frozen M6-wide default `I don't know.` is recognized as an
abstention in every requested-language stratum (while still counting as a
wrong-language output outside English). Exact accepted aliases are masked only while deciding the surrounding
output language, so a non-Latin proper name in a German answer does not itself
make the answer German-language-inconsistent; that name still contributes to the
separately reported non-target script-token rate. Final derivation reports all
automated rates by requested language, the adjudicated human consistency score,
and paired correct-M0 to wrong-language-M6 transitions. Terminal E8/E10 replay
also recomputes each language grade against the exact aliases in the frozen
source `Question`, not aliases self-declared by the response evidence.

## 8. Human audit

The scientific audit is built only from complete, live, verified E9 and E10
ledgers. It samples the frozen protocol queues and strata, hides model/method/
prompt/hypothesis identities, and uses keyed HMAC identifiers.

Create and protect a 32-byte high-entropy secret. It is required for every later
scientific verification and is never stored in the bundle:

```bash
umask 077
mkdir -p "$STUDY/private"
openssl rand -out "$STUDY/private/human-audit.key" 32
```

Prepare the blinded queue:

```bash
uv run mfh prepare-human-audit \
  "$STUDY/audit/queue" \
  configs/analysis/confirmatory.yaml \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  --blinding-key-file "$STUDY/private/human-audit.key"
```

Give `blind-items.jsonl` and separate copies of `annotation-template.csv` to two
independent annotators. Do not give them `operator-bindings.jsonl`. Each submitted
CSV must contain its distinct embedded annotator ID. Supply all disagreements to
an adjudicator, then finalize:

```bash
uv run mfh finalize-human-audit \
  "$STUDY/audit/queue" "$STUDY/audit/results" \
  configs/analysis/confirmatory.yaml \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  --blinding-key-file "$STUDY/private/human-audit.key" \
  --annotation "annotator-a=$STUDY/audit/annotator-a.csv" \
  --annotation "annotator-b=$STUDY/audit/annotator-b.csv" \
  --adjudications "$STUDY/audit/adjudications.csv"
```

The finalizer enforces two distinct regular files, exact audit IDs and order,
complete annotations, required adjudications, agreement statistics, and confusion
matrices. Copies, hard links, symlinks, reordered queues, wrong keys, and extra
filesystem entries are rejected.

## 9. Confirmatory analysis and reports

Do not author `FinalAnalysisResults` JSON by hand. The final derivation command
replays E1 paired prompt outcomes for power, the completed E3 layer/alpha surface,
E6 likelihood/rank/forced-answer records, the promoted E7 SAE interpretability
artifact, E8 matched side-effect pairs, E9 factorial records and preregistered
analysis, E10 composite records, the completed prompt-robustness and semantic-fold
store, and finalized human annotations. It calculates every reported value from
those raw rows and the verified compact E3 surface, including the complete
per-comparison bootstrap/Holm family and all non-inferiority comparisons.

Before rendering any figure, derive and freeze the result payload against the
exact live record rows, shard identities, completion digests, contracts,
robustness tasks, audit manifest, and pre-run E9/E10 evaluator snapshots:

Here, `$E3_PHASE` is the `E3_OUTPUT_ROOT/phase` output of `finalize_e3_phase`, not an individual
E3 stage work directory.

```bash
uv run mfh freeze-analysis-evidence \
  "$STUDY/analysis/record-bound-evidence" \
  configs/analysis/confirmatory.yaml \
  docs/research-plan.md \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E1" "$E3_PHASE" "$STUDY/runs/E6" \
  "$STUDY/runs/E7" "$STUDY/runs/E8" \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  "$STUDY/robustness/results" \
  "$STUDY/audit/queue" "$STUDY/audit/results" \
  "$STUDY/auxiliary/aa-official" \
  --expected-aa-official-manifest-digest "$AA_OFFICIAL_MANIFEST_DIGEST" \
  --blinding-key-file "$STUDY/private/human-audit.key"
uv run mfh verify-analysis-evidence \
  "$STUDY/analysis/record-bound-evidence" \
  configs/analysis/confirmatory.yaml \
  docs/research-plan.md \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E1" "$E3_PHASE" "$STUDY/runs/E6" \
  "$STUDY/runs/E7" "$STUDY/runs/E8" \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  "$STUDY/robustness/results" \
  "$STUDY/audit/queue" "$STUDY/audit/results" \
  "$STUDY/auxiliary/aa-official" \
  --expected-aa-official-manifest-digest "$AA_OFFICIAL_MANIFEST_DIGEST" \
  --blinding-key-file "$STUDY/private/human-audit.key"
```

Treat this directory as immutable. Its schema-v2 derivation receipt records the
E1/E3/E6/E7/E8/E9/E10 source and completion digests, replayed E9 analysis,
the official-AA auxiliary record identity and prompt comparison, robustness
records, finalized audit manifest, and derived-result digest. Verification
requires all of those sources again; supplying only E9/E10 is rejected.

Pass the returned `DerivedFinalAnalysis` as `derived_analysis` when calling
`write_final_analysis_bundle`. The writer rejects reports or results detached from
that replayed payload and packages the evidence for later source replay.

Generate each required section-21 report together with its typed JSON source data
using `mfh.analysis.reporting`, then call `write_final_analysis_bundle`. Each SVG
uses its registered scientific chart kind and includes a visible, exact source-value
ledger. The writer verifies chart semantics, every value binding, source-data
digests, generator revision, live E9/E10 records, and the full audit evidence before
copying anything.

Verify the published bundle against live evidence:

```bash
uv run mfh verify-analysis \
  "$STUDY/analysis/final" \
  configs/analysis/confirmatory.yaml \
  docs/research-plan.md \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E1" "$E3_PHASE" "$STUDY/runs/E6" \
  "$STUDY/runs/E7" "$STUDY/runs/E8" \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  "$STUDY/robustness/results" \
  "$STUDY/audit/queue" "$STUDY/audit/results" \
  "$STUDY/auxiliary/aa-official" \
  --expected-aa-official-manifest-digest "$AA_OFFICIAL_MANIFEST_DIGEST" \
  --blinding-key-file "$STUDY/private/human-audit.key"
```

The twelve planned scientific figures/tables plus adjudicated labels are required;
placeholder files and result-detached plots are rejected.

## 10. Suggested artifact layout

Keep mutable acquisition, private material, frozen inputs, active runs, and
published results separate:

```text
artifacts/
├── source/                 # manually acquired upstream files
├── questions/              # model-independent canonical questions
├── splits/                 # model-independent development split bundles
├── models/                 # exact local model snapshots
└── studies/
    └── qwen36-27b-mlx4-m4max48-v1/
        ├── private/        # blinding/signing keys; never publish
        ├── frozen/         # receipts, snapshots, components, question bundles
        ├── work/           # resumable, mutable computation
        ├── runs/
        │   ├── E0/
        │   ├── ...
        │   └── E10/
        ├── audit/
        │   ├── queue/
        │   └── results/
        └── analysis/
            └── final/
```

Never edit a frozen bundle in place. Failed verification means reacquire or
rebuild the artifact into a new directory, diagnose the cause, and preserve the
old evidence for auditability. Never treat a passing software test or synthetic
smoke as evidence that a scientific hypothesis succeeded.
