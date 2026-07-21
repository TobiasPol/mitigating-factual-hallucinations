# Mitigating factual hallucinations with activation steering

This repository implements the complete preregistered E0–E10 study in
[`docs/research-plan.md`](docs/research-plan.md). The active experiment uses one
exact model representation:

- upstream model: `Qwen/Qwen3.6-27B`;
- MLX artifact: `mlx-community/Qwen3.6-27B-4bit`;
- revision: `c000ac2c2057d94be3fa931000c31723aac53282`;
- runtime: official `mlx==0.31.2` and `mlx-lm==0.31.3`;
- target host: Apple M4 Max with 48 GiB unified memory.

The software workflow is implemented. The active Qwen scientific experiment has
not been executed yet. Model-generated phase results, the reviewed language
suite, the post-E9/E10 human audit, and final scientific reports therefore do not
exist until the corresponding operator and human steps below are completed.
Earlier Bonsai, Gemma, GGUF, AWQ, Transformers, CUDA, and Colab artifacts are
superseded provenance and cannot satisfy an active Qwen gate.

This README is the cross-device execution checklist. The much more detailed
scientific and artifact-level explanation is in
[`docs/operator-guide.md`](docs/operator-guide.md); use it whenever a phase asks
for an input artifact whose construction is summarized here.

## What is implemented

The repository provides:

- immutable source snapshots, contamination review, and reviewed TriviaQA
  splits;
- native MLX runtime validation and intervention hooks at all registered sites;
- resumable E0–E10 generation, activation, likelihood, fitting, and grading
  workflows;
- dense, CAA, adaptive, coordinate-sparse, SAE, and protected-subspace methods;
- official TriviaQA, SimpleQA Verified, AA-Omniscience, IFEval, StrongREJECT,
  utility, safety, and language evaluation;
- exact phase ledgers, signed runtime receipts, frozen gates, and portable
  terminal artifacts;
- prompt-paraphrase and semantic-fold generalization diagnostics;
- blinded two-annotator human audit with adjudication;
- preregistered statistical derivation and source-bound report validation.

No performance audit is part of this handoff. Bounded `--limit`,
`--request-budget`, and checkpoint options exist to make long jobs resumable;
they do not change the registered scientific schedule.

## 1. Move the project to the M4 Max

The `artifacts/` directory is intentionally ignored by Git. A normal clone is
not enough: transfer the existing model-independent source, split, contamination,
and grader artifacts separately, preserving file bytes and permissions. Do not
copy old Bonsai run directories into the new Qwen study namespace.

A practical transfer is:

```bash
# Run from the parent directory on the current machine.
rsync -a --protect-args \
  mitigating-factual-hallucinations/ \
  USER@M4_HOST:/ABSOLUTE/PATH/mitigating-factual-hallucinations/
```

This also transfers ignored pilot models and can be large. To transfer only the
active inputs, copy the Git working tree plus these paths:

```text
artifacts/contamination/triviaqa-ood/
artifacts/contamination/triviaqa-ood-manual-review/
artifacts/contamination/triviaqa-ood-manual-review-result/
artifacts/e0/runtime-validation-500/
artifacts/graders/e1-frozen-v2/
artifacts/graders/openrouter/
artifacts/models/semantic/
artifacts/questions/
artifacts/source/
artifacts/splits/triviaqa-auto-clean/
artifacts/splits/triviaqa-reviewed/
artifacts/studies/qwen36-27b-mlx4-m4max48-v1/frozen/E5-controller-splits/
```

Transfer `.env` and private keys over a protected channel, or recreate them on
the target. Never commit them. After transfer:

```bash
cd /ABSOLUTE/PATH/mitigating-factual-hallucinations
chmod 600 .env 2>/dev/null || true
```

## 2. Install the exact environment

Install Xcode and the matching Metal Toolchain first. Then install current
`pyenv` and `uv`, and run:

```bash
# CPython 3.11.14 is no longer in uv's downloadable Python catalog. Install the
# exact registered runtime with pyenv; .python-version makes uv select it.
pyenv install --skip-existing 3.11.14
pyenv local 3.11.14
python --version  # must print Python 3.11.14

uv sync --extra dev --extra research --extra mlx-macos
uv lock --check
uv run hf version
```

Do not replace 3.11.14 with the newest 3.11 patch: the runtime policy and frozen
artifacts require this exact interpreter. If `uv` reports `No download found for
request: cpython-3.11.14`, it is being asked to install a version no longer in its
catalog; use the `pyenv` commands above. If `uv` still resolves to an older
`~/.local/bin/uv`, remove that stale installation or put the current `uv` earlier
on `PATH`.

`hf` is the current Hugging Face Hub CLI (`huggingface-cli` is deprecated). The
locked research extra supplies it, and it reads `HF_TOKEN` from the environment.

The lock file and runtime policy pin the scientific dependency versions. Do not
upgrade MLX, MLX-LM, Transformers, Torch, NumPy, the tokenizer, or the model
revision within this study namespace.

Run the repository checks before downloading or loading Qwen:

```bash
uv run ruff check .
uv run mypy src/mfh
uv run pytest

uv run mfh validate-study \
  configs/experiments/phases.yaml \
  configs/analysis/confirmatory.yaml \
  docs/research-plan.md
```

The first three commands must pass. `validate-study` must report E0 through E10
and `valid: true`.

## 3. Configure paths and secrets

From the repository root:

```bash
export REPO="$PWD"
export STUDY="$REPO/artifacts/studies/qwen36-27b-mlx4-m4max48-v1"
export MODEL="$REPO/artifacts/models/qwen3.6-27b-mlx-4bit/c000ac2c2057d94be3fa931000c31723aac53282"

mkdir -p \
  "$STUDY"/{operator-inputs,secrets,private,frozen,work,outputs,runs,evidence,checkpoints,analysis,audit,final,runtime,auxiliary}
umask 077
```

The local `.env` must contain:

```dotenv
HF_TOKEN=REPLACE_WITH_YOUR_HUGGING_FACE_TOKEN
OPENROUTER_API_KEY=REPLACE_WITH_YOUR_OPENROUTER_KEY
MFH_EXECUTION_PRIVATE_KEY=REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS
```

Load these values into each new shell before using `hf` or Python APIs that read
the process environment:

```bash
set -a
source .env
set +a
```

Generate the Ed25519 seed if it does not already exist:

```bash
openssl rand -hex 32 > "$STUDY/secrets/execution-private-key.hex"
chmod 600 "$STUDY/secrets/execution-private-key.hex"
```

Copy that file's single 64-character value into
`MFH_EXECUTION_PRIVATE_KEY`. E4–E10 must use this same key. The private key is
never packaged into a run artifact; its public half is frozen into signed
receipts and scorer bundles.

Create the later human-audit key separately:

```bash
openssl rand -out "$STUDY/private/human-audit.key" 32
chmod 600 "$STUDY/private/human-audit.key"
```

## 4. Acquire and verify Qwen

No experiment command downloads model weights implicitly.

```bash
uv run hf download mlx-community/Qwen3.6-27B-4bit \
  --revision c000ac2c2057d94be3fa931000c31723aac53282 \
  --local-dir "$MODEL"

# Preserve hf's resumable local-dir metadata outside the immutable snapshot.
mv "$MODEL/.cache" "$STUDY/private/qwen-hf-download-cache"

uv run mfh verify-transformers-snapshot \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json
```

The verified directory must contain exactly the 16 files declared by the
snapshot manifest, without symlinks.

Run the write-once live MLX preflight:

```bash
uv run mfh preflight-mlx-runtime \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  configs/runtimes/qwen3.6-27b-mlx-4bit-policy.json \
  "$STUDY/frozen/mlx-preflight.json" \
  --project-root .
```

This verifies the exact host, package and source hashes, model class, layer-type
sequence, deterministic no-thinking prompt rendering, cached and uncached
zero-vector parity, prompt-token scope, and intervention sensitivity. Do not edit
a failed receipt. Diagnose the host and create a reviewed new namespace if the
registered environment cannot pass.

## 5. Confirm the transferred data inputs

The existing contamination review and reviewed TriviaQA splits are
model-independent and may be reused byte-for-byte. Replay them before E0. The
full commands and frozen digests are in
[`docs/operator-guide.md`](docs/operator-guide.md#4-acquire-and-freeze-external-inputs).

At minimum, verify:

```bash
uv run mfh verify-runtime-validation \
  artifacts/e0/runtime-validation-500 \
  artifacts/splits/triviaqa-auto-clean/reserved.jsonl \
  --expected-manifest-digest bb89b2da16d899f8a38c0b090f84d1e43ffd1132e0fe0693295230b804f44442 \
  --parent-split-manifest-digest a3b646d7057c3e863c06b7ed0f446a28c63b8fb12e203e9b6b61cb2f2f8027f0 \
  --contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2

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
  --expected-review-result-manifest-digest 6e03e98d9b09ee83fcfbbe5d1268ef42d2991467db043ecc345de84f64607f59 \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6 \
  --expected-split-manifest-digest 05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148 \
  --steer 30000 --controller 5000 --dev 5000 --test 5000 --seed 17

uv run mfh verify-e1-graders artifacts/graders/e1-frozen-v2 \
  --expected-manifest-digest b3af3c847c3488d6228a47c205186caca06bca8de1cd00dd81f0b83ac73e1159
```

Before E7/E8, also produce two required side-effect inputs:

1. Materialize the pinned IFEval evaluator:

   ```bash
   uv run mfh materialize-ifeval-evaluator "$STUDY/frozen/ifeval-evaluator"
   ```

2. Build the 500-row, five-language suite from human-reviewed translations and
   two independent reviewer signatures:

   ```bash
   uv run mfh build-language-suite \
     "$STUDY/frozen/language-suite" \
     artifacts/source/triviaqa/d2ff7f468d3642dbd33123596331950db8a63d0e/rc.nocontext/train-00000-of-00001-e93ee6c1ba181971.parquet \
     "$STUDY/operator-inputs/language-translations.jsonl" \
     "$STUDY/operator-inputs/language-reviewer-public-keys.json"

   uv run mfh verify-language-suite "$STUDY/frozen/language-suite"
   ```

The language suite is a real human-input prerequisite. The repository contains
its schema and verifier, not fabricated translations or signatures.

After both artifacts exist, atomically stage every external E7/E8 input under
the active study namespace. This verifies the reviewed-split manifest, both
signed language reviews, the frozen IFEval evaluator, and the registered bytes
of all six raw sources before and after copying:

```bash
export SPLIT_MANIFEST_DIGEST="05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148"
export TRIVIAQA_SOURCE="$REPO/artifacts/source/triviaqa/d2ff7f468d3642dbd33123596331950db8a63d0e/rc.nocontext/train-00000-of-00001-e93ee6c1ba181971.parquet"
export SIMPLEQA_SOURCE="$REPO/artifacts/source/simpleqa-verified/0dc97e0d28d8233463e005cdc4475cc2a13ba2dc/simpleqa_verified.csv"
export AA_SOURCE="$REPO/artifacts/source/aa-omniscience/4a8ffc87c4650054825fb767fe0da4a4fc97ff32/AA-Omniscience_dataset_public.csv"
export IFEVAL_SOURCE="$REPO/artifacts/source/ifeval/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl"
export MMLU_PRO_SOURCE="$REPO/artifacts/source/mmlu-pro/b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet"
export WIKITEXT_SOURCE="$REPO/artifacts/source/wikitext103/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/test-00000-of-00001.parquet"
export XSTEST_SOURCE="$REPO/artifacts/source/xstest/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv"
export STRONGREJECT_SOURCE="$REPO/artifacts/source/strongreject/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset.csv"
export E7_E8_INPUTS="$STUDY/frozen/E7-E8-external-inputs"

uv run mfh stage-e7-e8-inputs \
  "$E7_E8_INPUTS" \
  "$REPO/artifacts/splits/triviaqa-reviewed" \
  "$STUDY/frozen/language-suite" \
  "$STUDY/frozen/ifeval-evaluator" \
  --triviaqa-source "$TRIVIAQA_SOURCE" \
  --ifeval-source "$IFEVAL_SOURCE" \
  --mmlu-pro-source "$MMLU_PRO_SOURCE" \
  --wikitext103-source "$WIKITEXT_SOURCE" \
  --xstest-source "$XSTEST_SOURCE" \
  --strongreject-source "$STRONGREJECT_SOURCE" \
  --expected-reviewed-split-manifest-digest "$SPLIT_MANIFEST_DIGEST"

uv run mfh verify-e7-e8-inputs \
  "$E7_E8_INPUTS" \
  --expected-reviewed-split-manifest-digest "$SPLIT_MANIFEST_DIGEST"
```

The staging directory is write-once. Do not replace it with symlinks or point
E7/E8 directly at the repository copies; their packaged scientific inputs are
deliberately confined to `$STUDY`.

## 6. Execution rules that apply to every phase

- Run phases in order. A later phase opens and verifies every prerequisite.
- Keep all new Qwen work, ledgers, and outputs under `$STUDY`.
- Treat `frozen/`, completed ledgers, terminal artifacts, snapshots, and receipts
  as write-once.
- Never change a runbook after a stage has created work from it.
- Use the same execution key throughout E4–E10.
- A bounded run is resumed by rerunning the identical command. Do not delete
  shards or edit checkpoint JSON.
- After a model-heavy command, let the process exit before SAE fitting or another
  model load so MLX releases unified memory.
- `verify-*` and `preflight-*` commands are read-only unless their help explicitly
  says they materialize an artifact.
- A falsified scientific gate is a valid terminal result. Do not tune past it in
  the same study.
- E10 is one-shot. Its explicit authorization consumes an external reservation.

Useful read-only checks:

```bash
uv run mfh phase-progress "$STUDY/runs/E5" configs/experiments/phases.yaml
uv run mfh verify-phase "$STUDY/runs/E5" configs/experiments/phases.yaml
uv run mfh --help
```

## 7. Run E0–E5

E0–E5 have phase-specific data, capture, fitting, and promotion commands. Follow
the exact command blocks in these operator-guide sections in order:

1. [E0 runtime validation and Qwen admission](docs/operator-guide.md#4-acquire-and-freeze-external-inputs)
2. [Scientific lifecycle and confirmatory freezes](docs/operator-guide.md#5-scientific-phase-lifecycle)
3. [E0–E10 execution map and E3 operator](docs/operator-guide.md#6-e0e10-execution-map)
4. [Native M2 CAA](docs/operator-guide.md#61-build-the-native-mlx-m2-caa-artifact)
5. [Native E4 screen](docs/operator-guide.md#62-run-the-native-mlx-e4-baseline-screen)
6. [E5 controller capture, fitting, ablation, selection, and phase promotion](docs/operator-guide.md#63-capture-the-native-e5-controller-fitting-tensors)
7. [AA official auxiliary track](docs/operator-guide.md#64-run-the-auxiliary-aa-official-track)

The high-level terminal sequence is:

```text
E0: preflight runtime -> run native validation -> complete E0 -> finalize E0 ledger
E1: prepare -> generate -> OpenRouter grade -> finalize -> verify
E2: prepare capture -> capture -> verify -> fit probes -> finalize
E3: write runbook -> preflight -> repeat advance -> verify terminal phase
E4: build M2/ACT -> prepare screen -> run -> verify -> finalize
E5: materialize controller split -> capture -> layer labels -> fit controller grid
    -> native ablation -> derive selection -> promote records -> finalize phase
```

Important expected schedules include E1's 19,800 rows, E2's 21,600 capture
rows, E4's 12,000 screen rows, 15,000 E5 layer-label executions, and the frozen
9,730,000-row E5 native ablation. These are protocol identities, not runtime
estimates. The E5 preparation command deliberately requires
`--acknowledge-exact-grid-records 9730000` after the separate schedule check.

Do not continue after a phase until its terminal verifier passes or publishes a
valid falsification outcome.

### E0 exact ledger finalization

The operator-guide E0 section runs the native 500-row validation and creates
`$STUDY/outputs/E0-completion`. Record the four identities printed by the MLX,
manual-review, and completion commands, then publish the actual E0 phase ledger:

```bash
export E0_MLX_MANIFEST_DIGEST="REPLACE_WITH_E0_MLX_MANIFEST_DIGEST"
export E0_MLX_PLAN_IDENTITY="REPLACE_WITH_E0_MLX_PLAN_IDENTITY"
export REVIEW_RESULT_DIGEST="REPLACE_WITH_REVIEW_RESULT_DIGEST"
export E0_COMPLETION_DIGEST="REPLACE_WITH_E0_COMPLETION_MANIFEST_DIGEST"

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
  --expected-manifest-digest "$E0_COMPLETION_DIGEST" \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-mlx-manifest-digest "$E0_MLX_MANIFEST_DIGEST" \
  --expected-mlx-plan-identity "$E0_MLX_PLAN_IDENTITY" \
  --expected-review-result-manifest-digest "$REVIEW_RESULT_DIGEST" \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6 \
  --expected-cohort-manifest-digest bb89b2da16d899f8a38c0b090f84d1e43ffd1132e0fe0693295230b804f44442 \
  --parent-split-manifest-digest a3b646d7057c3e863c06b7ed0f446a28c63b8fb12e203e9b6b61cb2f2f8027f0 \
  --contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2

uv run mfh phase-progress \
  "$STUDY/runs/E0" configs/experiments/phases.yaml
uv run mfh verify-phase \
  "$STUDY/runs/E0" configs/experiments/phases.yaml
```

E1 is blocked until the final `verify-phase` succeeds. The finalizer replays the
completion receipt, live model and preflight, contamination review, cohort,
snapshot, and all 500 native records before atomically publishing the ledger.

### E1 exact baseline commands

The following paths are reused by every E1 command. The split and grader
digests below are the externally recorded identities of the transferred,
already-reviewed inputs.

```bash
export SPLITS="artifacts/splits/triviaqa-reviewed"
export E1_GRADERS="artifacts/graders/e1-frozen-v2"
export SPLIT_MANIFEST_DIGEST="05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148"
export GRADER_MANIFEST_DIGEST="b3af3c847c3488d6228a47c205186caca06bca8de1cd00dd81f0b83ac73e1159"
export E1_WORK="$STUDY/work/E1"
export E1_LEDGER="$STUDY/runs/E1"
export E1_OUTPUT="$STUDY/frozen/E1-outputs"
```

Create the immutable 19,800-row schedule only after `$STUDY/runs/E0` verifies
as complete. Preparation re-authorizes the human-reviewed split against all
live contamination evidence:

```bash
uv run mfh prepare-e1-mlx \
  "$SPLITS" "$E1_GRADERS" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml \
  "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E1_WORK" "$E1_LEDGER" "$STUDY/runs/E0" \
  artifacts/contamination/triviaqa-ood-manual-review-result \
  artifacts/contamination/triviaqa-ood-manual-review \
  artifacts/contamination/triviaqa-ood \
  configs/contamination/triviaqa-ood.yaml \
  artifacts/models/semantic/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  artifacts/questions/triviaqa-canonical.jsonl \
  --target artifacts/questions/simpleqa_verified-canonical.jsonl \
  --target artifacts/questions/aa_omniscience_public_600-canonical.jsonl \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST" \
  --expected-contamination-manifest-digest ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2 \
  --expected-review-result-manifest-digest 6e03e98d9b09ee83fcfbbe5d1268ef42d2991467db043ecc345de84f64607f59 \
  --expected-review-queue-manifest-digest 02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6 \
  --steer 30000 --controller 5000 --dev 5000 --test 5000 --seed 17
```

Generate in bounded sessions. Use the first command once; after any partial
session, use the second command verbatim until generation progress is complete.
The external checkpoint is deliberately outside the work and ledger bundles.

```bash
uv run mfh run-e1-mlx \
  "$SPLITS" "$E1_GRADERS" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E1_WORK" "$E1_LEDGER" "$STUDY/runs/E0" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST" \
  --checkpoint-file "$STUDY/checkpoints/E1-generation.json" \
  --request-budget 64

uv run mfh run-e1-mlx \
  "$SPLITS" "$E1_GRADERS" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E1_WORK" "$E1_LEDGER" "$STUDY/runs/E0" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST" \
  --checkpoint-file "$STUDY/checkpoints/E1-generation.json" \
  --request-budget 64 --resume
```

Once all local generations exist, grade the frozen external-rubric subset. As
above, run without `--resume` once, then repeat the second command after partial
or interrupted provider sessions:

```bash
uv run mfh grade-e1-openrouter \
  "$SPLITS" "$E1_GRADERS" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E1_WORK" "$E1_LEDGER" "$STUDY/runs/E0" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST" \
  --checkpoint-file "$STUDY/checkpoints/E1-grading.json" \
  --request-budget 100 --env-file .env

uv run mfh grade-e1-openrouter \
  "$SPLITS" "$E1_GRADERS" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E1_WORK" "$E1_LEDGER" "$STUDY/runs/E0" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST" \
  --checkpoint-file "$STUDY/checkpoints/E1-grading.json" \
  --request-budget 100 --env-file .env --resume
```

Freeze the labels, reporting gates, and prompt metrics. Record the printed
`manifest_digest` as `E1_MANIFEST_DIGEST`, then replay it independently:

```bash
uv run mfh finalize-e1 \
  "$SPLITS" "$E1_GRADERS" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E1_WORK" "$E1_LEDGER" "$STUDY/runs/E0" "$E1_OUTPUT" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-grader-manifest-digest "$GRADER_MANIFEST_DIGEST"

export E1_MANIFEST_DIGEST="REPLACE_WITH_FINALIZE_E1_MANIFEST_DIGEST"
uv run mfh verify-e1-outputs \
  "$E1_OUTPUT" "$E1_WORK" "$E1_LEDGER" \
  --expected-manifest-digest "$E1_MANIFEST_DIGEST"
```

### E2 exact capture and probe commands

E2 is fully local. It reuses E1's frozen records and captures 21,600 registered
prompt-end feature rows. Keep the model loaded only for `run-e2-mlx`; the probe
fit runs after that command exits and releases MLX memory.

```bash
export E2_WORKSPACE="$STUDY/work/E2-workspace"
export E2_CAPTURE="$STUDY/work/E2-capture"
export E2_PROBES="$STUDY/frozen/E2-probes"
export E2_PROBE_WORK="$STUDY/work/E2-probe-fit"
export E2_PHASE="$STUDY/runs/E2"

uv run mfh prepare-e2-mlx \
  "$SPLITS" "$E1_OUTPUT" "$E1_WORK" "$E1_LEDGER" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E2_WORKSPACE" "$E2_CAPTURE" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-e1-manifest-digest "$E1_MANIFEST_DIGEST" \
  --shard-rows 64
```

Record the printed `workspace_plan_identity`. Repeat the run command until the
verifier reports `rows_completed: 21600` and `complete: true`:

```bash
uv run mfh run-e2-mlx \
  "$SPLITS" "$E1_OUTPUT" "$E1_WORK" "$E1_LEDGER" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E2_WORKSPACE" "$E2_CAPTURE" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-e1-manifest-digest "$E1_MANIFEST_DIGEST" \
  --request-budget 64

uv run mfh verify-e2-capture \
  "$SPLITS" "$E1_OUTPUT" "$E1_WORK" "$E1_LEDGER" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E2_WORKSPACE" "$E2_CAPTURE" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-e1-manifest-digest "$E1_MANIFEST_DIGEST" \
  --require-complete
```

Record the verifier's `capture_plan_identity`. Fit the preregistered probe grid,
record its `manifest_digest`, and verify the full bundle:

```bash
uv run mfh fit-e2-probes \
  "$SPLITS" "$E1_OUTPUT" "$E1_WORK" "$E1_LEDGER" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E2_WORKSPACE" "$E2_CAPTURE" "$E2_PROBES" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-e1-manifest-digest "$E1_MANIFEST_DIGEST" \
  --probe-work-directory "$E2_PROBE_WORK"

uv run mfh verify-e2-probes "$E2_PROBES" "$E2_WORKSPACE"
```

Export the three identities printed above and finalize E2. A failed separability
gate is a valid immutable falsification and must not be tuned around.

```bash
export E2_WORKSPACE_PLAN_ID="REPLACE_WITH_PREPARE_E2_WORKSPACE_PLAN_IDENTITY"
export E2_CAPTURE_PLAN_ID="REPLACE_WITH_VERIFY_E2_CAPTURE_PLAN_IDENTITY"
export E2_PROBE_MANIFEST_DIGEST="REPLACE_WITH_FIT_E2_PROBE_MANIFEST_DIGEST"

uv run mfh finalize-e2 \
  "$SPLITS" "$E1_OUTPUT" "$E1_WORK" "$E1_LEDGER" \
  configs/models/qwen3.6-27b-mlx-4bit.yaml "$MODEL" \
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json \
  "$STUDY/frozen/mlx-preflight.json" \
  "$E2_WORKSPACE" "$E2_CAPTURE" "$E2_PROBES" "$E2_PHASE" \
  --expected-split-manifest-digest "$SPLIT_MANIFEST_DIGEST" \
  --expected-e1-manifest-digest "$E1_MANIFEST_DIGEST" \
  --expected-workspace-plan-identity "$E2_WORKSPACE_PLAN_ID" \
  --expected-capture-plan-identity "$E2_CAPTURE_PLAN_ID" \
  --expected-probe-manifest-digest "$E2_PROBE_MANIFEST_DIGEST"

uv run mfh phase-progress "$E2_PHASE" configs/experiments/phases.yaml
uv run mfh verify-phase "$E2_PHASE" configs/experiments/phases.yaml
```

## 8. Run E6

Create the exact source-backed E6 question bundle after the reviewed split
verifier passes. This writes the three registered JSONL schedules, their pinned
raw source bytes, and a replayable manifest; it does not load Qwen:

```bash
export E6_QUESTIONS="$STUDY/frozen/E6-questions"

uv run mfh freeze-e6-questions \
  "$E6_QUESTIONS" "$REPO/artifacts/splits/triviaqa-reviewed" \
  --triviaqa-source "$TRIVIAQA_SOURCE" \
  --simpleqa-source "$SIMPLEQA_SOURCE" \
  --aa-source "$AA_SOURCE" \
  --expected-reviewed-split-manifest-digest "$SPLIT_MANIFEST_DIGEST"
```

E3 selected the M1-P layer using development evidence. Read that exact value;
do not assume layer 31 and do not select a replacement from E4–E10 outcomes:

```bash
export E3_SCOPE_SELECTION="$STUDY/E3-operator/selections/scope.json"
export M1_LAYER="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["M1-P"]["layer"])' "$E3_SCOPE_SELECTION")"
case "$M1_LAYER" in ''|*[!0-9]*) echo "invalid E3 M1-P layer" >&2; exit 1;; esac
```

Create the runbook only after E0–E5 are terminal. The required layer argument
prevents a stale illustrative layer from entering E6:

```bash
export E6_RUNBOOK="$STUDY/operator-inputs/E6-runbook.json"
uv run mfh write-e6-runbook "$E6_RUNBOOK" --m1-layer "$M1_LAYER"
```

At that exact runbook location, the generated relative paths resolve to:

| E6 runbook field | Exact resolved artifact |
| --- | --- |
| `snapshot_directory` | `$MODEL` (immutable repository model; manifest-verified) |
| `frozen_question_bundle` | `$STUDY/frozen/E6-questions` |
| `official_grader_bundle` | `$REPO/artifacts/graders/e1-frozen-v2` |
| `e3_static_vectors` | `$STUDY/E3-operator/vectors` |
| `e5_adaptive_controllers` | `$STUDY/frozen/E5-phase/selected-controller` |
| `prerequisite_runs.E3` | `$STUDY/E3-operator/phase` (custom seven-stage terminal) |
| `prerequisite_runs.E5` | `$STUDY/runs/E5` |
| `execution_key_file` | `$STUDY/secrets/execution-private-key.hex` |
| `runtime_artifact` | `$STUDY/runtime/E6-attestation.json` |
| run/work/likelihood/final outputs | `$STUDY/{runs/E6,work/E6,frozen/E6-likelihoods,final/E6}` |

Do not move the JSON: its paths are relative to `$STUDY/operator-inputs`. You
may inspect it with `uv run python -m json.tool "$E6_RUNBOOK"`, but do not edit
scientific constants.

Then execute:

```bash
uv run mfh preflight-e6 "$E6_RUNBOOK"
uv run mfh prepare-e6 "$E6_RUNBOOK"
uv run mfh attest-e6-runtime "$E6_RUNBOOK"

# Repeat until remaining_records is zero.
uv run mfh run-e6 "$E6_RUNBOOK" --limit 128
uv run mfh verify-e6 "$E6_RUNBOOK"

uv run mfh finalize-e6 "$E6_RUNBOOK"
uv run mfh verify-e6 "$E6_RUNBOOK"
```

E6 freezes 59,400 paired generation/likelihood rows and tests whether factual
improvements reflect knowledge recovery instead of abstention substitution.

## 9. Run E7

```bash
export E7_RUNBOOK="$STUDY/operator-inputs/E7-runbook.json"
uv run mfh write-e7-runbook "$E7_RUNBOOK" --m1-layer "$M1_LAYER"
```

The generated E7 runbook is already bound to the standard exact paths:

| E7 runbook field | Exact resolved artifact |
| --- | --- |
| `snapshot_directory` | `$MODEL` |
| `reviewed_splits` | `$E7_E8_INPUTS/reviewed-splits` |
| `source_artifacts.*` | the named file under `$E7_E8_INPUTS/sources/` |
| `reviewed_language_suite` | `$E7_E8_INPUTS/language-suite` |
| `ifeval_evaluator` | `$E7_E8_INPUTS/ifeval-evaluator` |
| `e3_static_vectors` | `$STUDY/E3-operator/vectors` |
| `e5_adaptive_controllers` | `$STUDY/frozen/E5-phase/selected-controller` |
| `runtime_artifact` | `$STUDY/final/E6/gate-artifacts/knowledge_recovery_separated_from_abstention_substitution/likelihood-bundle/runtime-artifact` |
| prerequisites E3/E5/E6 | `$STUDY/E3-operator/phase`, `$STUDY/runs/E5`, `$STUDY/runs/E6` |
| outputs | the declared `$STUDY/work/E7`, `$STUDY/frozen/E7-*`, `$STUDY/runs/E7`, and `$STUDY/final/E7` paths |

The model stays in `$MODEL`; only the external data/evaluator snapshot is copied
under `$STUDY`. Do not create `$STUDY/runs/E3`: the custom E3 terminal is
`$STUDY/E3-operator/phase`.

Run each stage in this order:

```bash
uv run mfh preflight-e7 "$E7_RUNBOOK"
uv run mfh prepare-e7 "$E7_RUNBOOK"

# Repeat each partition until complete.
uv run mfh capture-e7 "$E7_RUNBOOK" T-steer --limit 256
uv run mfh capture-e7 "$E7_RUNBOOK" sae-train --limit 256
uv run mfh capture-e7 "$E7_RUNBOOK" sae-validation --limit 256

uv run mfh screen-e7-coordinate "$E7_RUNBOOK" --limit 64
uv run mfh fit-e7-sae "$E7_RUNBOOK"

# Repeat both audits until their reported rows are complete.
uv run mfh audit-e7-causal "$E7_RUNBOOK" --limit 64
uv run mfh audit-e7-interpretability "$E7_RUNBOOK" --limit 64

uv run mfh promote-e7-sae "$E7_RUNBOOK"
uv run mfh prepare-e7-ledger "$E7_RUNBOOK"

# Repeat until remaining_records is zero.
uv run mfh run-e7 "$E7_RUNBOOK" --limit 128
uv run mfh verify-e7-runbook "$E7_RUNBOOK"

uv run mfh finalize-e7-runbook "$E7_RUNBOOK"
uv run mfh verify-e7-runbook "$E7_RUNBOOK"
```

Promotion is refused unless reconstruction, two-seed feature stability,
individual activation/suppression causality, prompt transfer, and protected
behavior audits pass. The final E7 matrix contains 39,624 rows.

## 10. Run E8

```bash
export E8_RUNBOOK="$STUDY/operator-inputs/E8-runbook.json"
uv run mfh write-e8-runbook "$E8_RUNBOOK" --m1-layer "$M1_LAYER"
```

The E8 template reuses the same `$MODEL`, execution key, E6 runtime attestation,
staged external inputs, and frozen M1 layer. Its remaining lineage is exact:

| E8 runbook field | Exact resolved artifact |
| --- | --- |
| `e6_transition_evidence` | `$STUDY/runs/E6/gate-artifacts/knowledge_recovery_separated_from_abstention_substitution/likelihood-bundle` |
| `e7_finalization` | `$STUDY/final/E7` |
| `prerequisite_runs.E6` | `$STUDY/runs/E6` |
| `prerequisite_runs.E7` | `$STUDY/runs/E7` |
| external question/evaluator fields | the corresponding child of `$E7_E8_INPUTS` |
| outputs | the declared `$STUDY/work/E8`, `$STUDY/frozen/E8-*`, `$STUDY/runs/E8`, and `$STUDY/final/E8` paths |

Run:

```bash
uv run mfh preflight-e8 "$E8_RUNBOOK"
uv run mfh prepare-e8 "$E8_RUNBOOK"

uv run mfh capture-e8-activations "$E8_RUNBOOK" --limit 256
uv run mfh screen-e8-variants "$E8_RUNBOOK" --limit 64
uv run mfh promote-e8-protected "$E8_RUNBOOK"

uv run mfh screen-e8-candidates "$E8_RUNBOOK" --limit 64
uv run mfh prepare-e8-ledger "$E8_RUNBOOK"

# Repeat until remaining_records is zero.
uv run mfh run-e8 "$E8_RUNBOOK" --limit 128
uv run mfh verify-e8-runbook "$E8_RUNBOOK"

uv run mfh finalize-e8-runbook "$E8_RUNBOOK"
uv run mfh verify-e8-runbook "$E8_RUNBOOK"
```

E8 captures six protected behaviors, compares orthogonal and covariance-aware
directions on paired rows, freezes a 40-condition strength screen, selects every
method at a common empirical risk/coverage target, and executes the final
86,040-row side-effect matrix.

## 11. Freeze E9 inputs and run robustness diagnostics

The E9 freeze operator derives the promoted M1–M5 components, confirmatory
graders, exact question bundle, robustness schedule, and a ready E9 runbook from
terminal E0–E8 evidence. No Python integration code or runbook editing is
required.

First copy the verified external inputs into the active Qwen namespace. This is
an atomic, validated byte-for-byte staging step:

```bash
export E9_STAGED="$STUDY/frozen/E9-external-inputs"
export TRIVIAQA_SOURCE="artifacts/source/triviaqa/d2ff7f468d3642dbd33123596331950db8a63d0e/rc.nocontext/train-00000-of-00001-e93ee6c1ba181971.parquet"
export SIMPLEQA_SOURCE="artifacts/source/simpleqa-verified/0dc97e0d28d8233463e005cdc4475cc2a13ba2dc/simpleqa_verified.csv"
export AA_SOURCE="artifacts/source/aa-omniscience/4a8ffc87c4650054825fb767fe0da4a4fc97ff32/AA-Omniscience_dataset_public.csv"

uv run mfh stage-e9-inputs \
  "$E9_STAGED" "$E1_GRADERS" "$SPLITS" \
  --triviaqa-source "$TRIVIAQA_SOURCE" \
  --simpleqa-source "$SIMPLEQA_SOURCE" \
  --aa-source "$AA_SOURCE" \
  --expected-official-grader-manifest-digest "$GRADER_MANIFEST_DIGEST"
```

Freeze the exact E9 evaluator/code snapshot, then derive the complete suite.
`M2_CAA`, `E3_CONSTRUCTION`, and `E3_PHASE` are the same verified artifacts
created in the E4 and E3 operator-guide steps. `E3_PHASE` is the custom
seven-stage terminal artifact; do not copy or symlink it to `$STUDY/runs/E3`.

```bash
export E9_SNAPSHOT="$STUDY/frozen/e9-evaluation-scripts"
export E9_FREEZE="$STUDY/frozen/E9-inputs"
export E9_RUNBOOK="$STUDY/operator-inputs/E9-runbook.json"
export M2_CAA="$STUDY/frozen/E4-M2-CAA"
export E3_CONSTRUCTION="$STUDY/E3-operator/construction"
export E3_PHASE="$STUDY/E3-operator/phase"

uv run mfh freeze-execution-snapshot \
  "$E9_SNAPSHOT" configs/experiments/phases.yaml E9 \
  --repository-root .

uv run mfh freeze-e9-inputs \
  "$E9_FREEZE" "$E8_RUNBOOK" "$E9_RUNBOOK" \
  "$E9_SNAPSHOT" \
  "$E9_STAGED/official-graders" \
  "$E9_STAGED/reviewed-splits" \
  "$M2_CAA" \
  "$E3_PHASE" \
  --triviaqa-source "$E9_STAGED/sources/triviaqa.parquet" \
  --simpleqa-source "$E9_STAGED/sources/simpleqa_verified.csv" \
  --aa-source "$E9_STAGED/sources/aa_omniscience_public_600.csv" \
  --expected-official-grader-manifest-digest "$GRADER_MANIFEST_DIGEST" \
  --env-file .env

uv run mfh preflight-confirmatory "$E9_RUNBOOK"
```

Now initialize and execute the two diagnostics. The RQ1 capture is a single
signed, resumable 30,000-row P0 T-steer replay shared by all folds. It retains
C/I/A source identities and applies the registered C/I projection only inside
vector fitting.

```bash
export ROBUSTNESS_PLAN="$E9_FREEZE/robustness-plan"
export ROBUSTNESS_RESULTS="$STUDY/work/robustness-results"
export ROBUSTNESS_EXECUTION="$STUDY/work/robustness-execution"
export E5_CAPTURE="$STUDY/work/E5-fit-capture"
export E5_LABELS="$STUDY/work/E5-layer-labels"
export T_CONTROLLER="$STUDY/frozen/E5-controller-splits/T-controller-train.jsonl"

uv run mfh verify-robustness-plan "$ROBUSTNESS_PLAN"
uv run mfh create-robustness-results \
  "$ROBUSTNESS_RESULTS" "$ROBUSTNESS_PLAN"

uv run mfh prepare-robustness-execution \
  "$ROBUSTNESS_EXECUTION" "$ROBUSTNESS_RESULTS" \
  "$E9_RUNBOOK" "$E3_CONSTRUCTION" --shard-rows 16

# Repeat until complete: true.
uv run mfh run-robustness-rq1-capture \
  "$ROBUSTNESS_EXECUTION" "$ROBUSTNESS_RESULTS" \
  "$E9_RUNBOOK" "$E3_CONSTRUCTION" \
  --env-file .env --limit 256

uv run mfh verify-robustness-rq1-capture \
  "$ROBUSTNESS_EXECUTION" "$ROBUSTNESS_RESULTS" \
  "$E9_RUNBOOK" "$E3_CONSTRUCTION" \
  --env-file .env --require-complete
```

The prompt command advances individual generations. The RQ1 command advances
whole semantic-fold tasks; `--limit 1` can therefore include one CPU refit plus
all held-fold generations. Repeat both commands until progress is 36,000/36,000
and 60/60:

```bash
uv run mfh run-robustness-prompts \
  "$ROBUSTNESS_RESULTS" "$E9_RUNBOOK" \
  --env-file .env --limit 100

uv run mfh run-robustness-rq1 \
  "$ROBUSTNESS_EXECUTION" "$ROBUSTNESS_RESULTS" \
  "$E9_RUNBOOK" "$E3_CONSTRUCTION" \
  "$E2_WORKSPACE" "$E2_PROBES" \
  "$E5_CAPTURE" "$E5_LABELS" "$T_CONTROLLER" \
  --env-file .env --limit 1

uv run mfh verify-robustness-results "$ROBUSTNESS_RESULTS"
```

The frozen inventory is 36,000 prompt-paraphrase generations and 60 RQ1 tasks.
Finalize only when both counts are complete:

```bash
uv run mfh finalize-robustness-results "$ROBUSTNESS_RESULTS"
uv run mfh verify-robustness-results \
  "$ROBUSTNESS_RESULTS" --require-complete
```

## 12. Run E9

The previous section already created and preflighted the secret-free E9
runbook. Create the ledger, then resume bounded native MLX/OpenRouter sessions:

```bash
uv run mfh prepare-confirmatory "$E9_RUNBOOK"

# Repeat until remaining_records is zero.
uv run mfh run-confirmatory "$E9_RUNBOOK" \
  --env-file .env --checkpoint-size 1 --limit 100
uv run mfh verify-confirmatory "$E9_RUNBOOK"

uv run mfh finalize-confirmatory "$E9_RUNBOOK"
uv run mfh verify-confirmatory "$E9_RUNBOOK"
```

The E9 contract is exactly 118,800 rows: one model × three prompts × six methods
× 6,600 factual questions. No component is selected from E9 outcomes.

## 13. Freeze M6 and run one-shot E10

After E9 is terminal, prepare the exact 10,000-row development-only early-token
capture. Repeat the run command until the read-only verifier reports complete:

```bash
export E10_FREEZE="$STUDY/frozen/E10-inputs"
export E10_SNAPSHOT="$STUDY/frozen/e10-evaluation-scripts"
export E10_RUNBOOK="$STUDY/operator-inputs/E10-runbook.json"

uv run mfh prepare-e10-freezes \
  "$E10_FREEZE" "$E8_RUNBOOK" "$E9_RUNBOOK"

uv run mfh run-e10-early-probe \
  "$E10_FREEZE" "$E8_RUNBOOK" "$E9_RUNBOOK" \
  --env-file .env --limit 128 --shard-rows 32

uv run mfh verify-e10-early-probe \
  "$E10_FREEZE" --require-complete
```

Freeze the E10 evaluator/code snapshot. The finalizer then fits the registered
probe grid, builds M6, packages the question/component bundles and all eleven
freeze fields, atomically publishes them, writes the E10 runbook, and preflights
the complete 10,204-row contract:

```bash
uv run mfh freeze-execution-snapshot \
  "$E10_SNAPSHOT" configs/experiments/phases.yaml E10 \
  --repository-root .

uv run mfh finalize-e10-freezes \
  "$E10_FREEZE" "$E8_RUNBOOK" "$E9_RUNBOOK" \
  "$E10_SNAPSHOT" "$E10_RUNBOOK"

uv run mfh preflight-confirmatory "$E10_RUNBOOK"
```

Record the reported contract digest externally. The preflight does not consume
the one-shot reservation. If and only if every path and digest is correct:

```bash
uv run mfh prepare-confirmatory \
  "$E10_RUNBOOK" --authorize-e10-one-shot

# Repeat until remaining_records is zero.
uv run mfh run-confirmatory "$E10_RUNBOOK" \
  --env-file .env --checkpoint-size 1 --limit 100
uv run mfh verify-confirmatory "$E10_RUNBOOK"

uv run mfh finalize-confirmatory "$E10_RUNBOOK"
uv run mfh verify-confirmatory "$E10_RUNBOOK"
```

E10 contains 10,204 rows across the three factual benchmarks and all registered
utility, safety, likelihood, and language suites. Its explicit authorization is
one-shot. If a gate fails, preserve the immutable falsification result; do not
alter M6 and rerun.

## 14. Run the blinded human audit

Only after complete E9 and E10 ledgers:

```bash
uv run mfh prepare-human-audit \
  "$STUDY/audit/queue" \
  configs/analysis/confirmatory.yaml \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  --blinding-key-file "$STUDY/private/human-audit.key"

uv run mfh verify-human-audit-queue \
  "$STUDY/audit/queue" \
  configs/analysis/confirmatory.yaml \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  --blinding-key-file "$STUDY/private/human-audit.key"
```

Give `blind-items.jsonl` and separate annotation templates to two independent
annotators. Do not give them `operator-bindings.jsonl`. After both annotate and
an adjudicator resolves every disagreement:

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

uv run mfh verify-human-audit-results \
  "$STUDY/audit/results" "$STUDY/audit/queue" \
  configs/analysis/confirmatory.yaml \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  --blinding-key-file "$STUDY/private/human-audit.key"
```

## 15. Derive and verify the final analysis

Complete the 600-row AA official auxiliary track as described in the operator
guide and record its manifest digest as `AA_OFFICIAL_MANIFEST_DIGEST`.

Freeze the record-bound analysis evidence:

```bash
uv run mfh freeze-analysis-evidence \
  "$STUDY/analysis/record-bound-evidence" \
  configs/analysis/confirmatory.yaml \
  docs/research-plan.md \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E1" "$STUDY/E3-operator/phase" "$STUDY/runs/E6" \
  "$STUDY/runs/E7" "$STUDY/runs/E8" \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  "$STUDY/work/robustness-results" \
  "$STUDY/audit/queue" "$STUDY/audit/results" \
  "$STUDY/auxiliary/aa-official" \
  --expected-aa-official-manifest-digest "$AA_OFFICIAL_MANIFEST_DIGEST" \
  --blinding-key-file "$STUDY/private/human-audit.key"
```

Replay the evidence against every live source:

```bash
uv run mfh verify-analysis-evidence \
  "$STUDY/analysis/record-bound-evidence" \
  configs/analysis/confirmatory.yaml \
  docs/research-plan.md \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E1" "$STUDY/E3-operator/phase" "$STUDY/runs/E6" \
  "$STUDY/runs/E7" "$STUDY/runs/E8" \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  "$STUDY/work/robustness-results" \
  "$STUDY/audit/queue" "$STUDY/audit/results" \
  "$STUDY/auxiliary/aa-official" \
  --expected-aa-official-manifest-digest "$AA_OFFICIAL_MANIFEST_DIGEST" \
  --blinding-key-file "$STUDY/private/human-audit.key"
```

Derive the results again, render all fourteen registered SVG/CSV reports, bind
each report to its typed source-data JSON, and atomically publish the bundle:

```bash
uv run mfh write-analysis \
  "$STUDY/analysis/final" \
  "$STUDY/analysis/record-bound-evidence" \
  configs/analysis/confirmatory.yaml \
  docs/research-plan.md \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E1" "$STUDY/E3-operator/phase" "$STUDY/runs/E6" \
  "$STUDY/runs/E7" "$STUDY/runs/E8" \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  "$STUDY/work/robustness-results" \
  "$STUDY/audit/queue" "$STUDY/audit/results" \
  "$STUDY/auxiliary/aa-official" \
  --expected-aa-official-manifest-digest "$AA_OFFICIAL_MANIFEST_DIGEST" \
  --blinding-key-file "$STUDY/private/human-audit.key"
```

Finally run:

```bash
uv run mfh verify-analysis \
  "$STUDY/analysis/final" \
  configs/analysis/confirmatory.yaml \
  docs/research-plan.md \
  configs/experiments/phases.yaml \
  "$STUDY/runs/E1" "$STUDY/E3-operator/phase" "$STUDY/runs/E6" \
  "$STUDY/runs/E7" "$STUDY/runs/E8" \
  "$STUDY/runs/E9" "$STUDY/runs/E10" \
  "$STUDY/work/robustness-results" \
  "$STUDY/audit/queue" "$STUDY/audit/results" \
  "$STUDY/auxiliary/aa-official" \
  --expected-aa-official-manifest-digest "$AA_OFFICIAL_MANIFEST_DIGEST" \
  --blinding-key-file "$STUDY/private/human-audit.key"
```

The final verifier re-derives every reported value from live phase rows, gate
artifacts, robustness tasks, official-AA records, and human annotations. A chart
or table detached from its typed source data is rejected.

## 16. Restarting and diagnosing a long run

For a normal interruption:

1. let the process exit cleanly when possible;
2. run the corresponding `verify-*` command;
3. rerun the identical `run-*` command with a bounded limit;
4. finalize only when the verifier reports the exact expected count.

Do not delete a partial shard, rewrite a manifest, change the runbook, or move a
live ledger. The framework checks resume-chain heads, signatures, source hashes,
condition IDs, row uniqueness, and packaged prerequisites.

Common failures:

- **Snapshot mismatch:** the model contains `.cache`, symlinks, a different
  revision, or changed files. Reacquire the exact revision.
- **Runtime policy mismatch:** code or locked dependencies changed after the
  preflight was approved. Use the reviewed repository state; do not edit the
  receipt.
- **Namespace rejection:** a mutable/frozen Qwen path is outside `$STUDY` or
  traverses a symlink.
- **Public-key mismatch:** a later phase used a different execution private key.
- **Incomplete prerequisite:** verify and terminally finalize the earlier phase.
- **OpenRouter failure:** retain the failure journal, correct credentials or
  provider availability, and resume; do not fabricate a grade.
- **Scientific gate failure:** preserve the falsification artifact and stop the
  dependent hypothesis path.

## 17. Completion checklist

The full experiment is complete only when all of the following hold:

- repository lint, strict typing, and tests pass;
- the exact Qwen snapshot and M4 Max live preflight verify;
- E0–E9 each have a replayable completion or registered falsification artifact;
- E10 has consumed one reservation and has a terminal completion/falsification;
- all robustness tasks are complete and the result store verifies;
- the AA official auxiliary track verifies;
- the two-annotator human audit and adjudication verify;
- record-bound analysis evidence replays;
- the final report bundle passes `verify-analysis`.

Passing the software test suite alone does not mean the scientific experiment
has been run. Conversely, a preregistered gate failure is scientifically valid
when its immutable evidence verifies.
