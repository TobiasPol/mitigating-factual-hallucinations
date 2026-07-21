"""Frozen adapters for released SimpleQA Verified and AA grading rubrics.

The framework does not substitute alias matching for model-based official
graders.  Provider invocation is injected, while the benchmark source,
autorater model, prompt, parsing, label map, and failure behavior are all part
of an immutable specification.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.config import load_yaml
from mfh.contracts import Outcome
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.metrics import MetricBundle, metric_bundle
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_PROMPT_PLACEHOLDER = re.compile(r"\{(question|target|predicted_answer)\}")
_RELEASED_GRADERS: Mapping[str, Mapping[str, Any]] = {
    "simpleqa_verified": {
        "source_repository": (
            "https://www.kaggle.com/code/nanliao7/"
            "simpleqa-verified-benchmark-starter-code?scriptVersionId=290594993"
        ),
        "source_revision": ("14d0c0513efefdfe7936e05c6fc09b4b4a191cc31273ca8bfbcdeaea0c6fdb1b"),
        "source_artifact": "simpleqa-verified-benchmark-starter-code-v9.ipynb",
        "source_artifact_sha256": (
            "14d0c0513efefdfe7936e05c6fc09b4b4a191cc31273ca8bfbcdeaea0c6fdb1b"
        ),
        "grader_model": "openai/gpt-4.1",
        "grader_model_revision": "gpt-4.1-2025-04-14",
        "reasoning_enabled": False,
        "prompt_sha256": "84c004ec4fcf8f0703bb0d734544036a72e847bfa7116429e5aee5e53ccc8cf3",
        "label_mapping": {"A": Outcome.CORRECT, "B": Outcome.INCORRECT, "C": Outcome.ABSTENTION},
        "maximum_attempts": 3,
    },
    "aa_omniscience_public_600": {
        "source_repository": "ArtificialAnalysis/AA-Omniscience-Public",
        "source_revision": "4a8ffc87c4650054825fb767fe0da4a4fc97ff32",
        "source_artifact": "README.md",
        "source_artifact_sha256": (
            "f3fd2fc7e507898fb7c718c9685c180fe4aa0afcc1d14230f95828baf5f999d4"
        ),
        "grader_model": "google/gemini-2.5-flash",
        "grader_model_revision": "gemini-2.5-flash",
        "reasoning_enabled": True,
        "prompt_sha256": "7d2f6d60367d8f3ca7c92b115c65fa536e321e27ce93e0779d77d5c3700901c9",
        "label_mapping": {
            "A": Outcome.CORRECT,
            "B": Outcome.INCORRECT,
            "C": Outcome.PARTIAL,
            "D": Outcome.ABSTENTION,
        },
        "maximum_attempts": 3,
    },
}


@dataclass(frozen=True, slots=True)
class OfficialGraderSpec:
    benchmark: str
    source_repository: str
    source_revision: str
    source_artifact: str
    source_artifact_sha256: str
    grader_model: str
    grader_model_revision: str
    temperature: float
    reasoning_enabled: bool
    prompt_template: str
    prompt_sha256: str
    label_mapping: Mapping[str, Outcome]
    maximum_attempts: int = 1
    failure_outcome: Outcome = Outcome.UNSCORABLE
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError("unsupported official-grader schema version")
        for name in (
            "benchmark",
            "source_repository",
            "source_artifact",
            "grader_model",
            "grader_model_revision",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise DataValidationError(f"official grader {name} must be non-empty")
            object.__setattr__(self, name, value)
        if not _REVISION.fullmatch(self.source_revision):
            raise DataValidationError("official grader source must use an immutable revision")
        if not _SHA256.fullmatch(self.source_artifact_sha256):
            raise DataValidationError("official grader source artifact must have a SHA-256")
        prompt = self.prompt_template.strip()
        placeholders = _PROMPT_PLACEHOLDER.findall(prompt)
        if not prompt or Counter(placeholders) != Counter(
            {"question": 1, "target": 1, "predicted_answer": 1}
        ):
            raise DataValidationError(
                "official grader prompt must contain each required placeholder exactly once"
            )
        computed_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if self.prompt_sha256 != computed_prompt_hash:
            raise DataValidationError("official grader prompt hash differs from its text")
        if not math.isfinite(self.temperature) or self.temperature != 0:
            raise DataValidationError("frozen official graders must use temperature zero")
        if not isinstance(self.reasoning_enabled, bool):
            raise DataValidationError("official grader reasoning flag must be boolean")
        if self.maximum_attempts <= 0:
            raise DataValidationError("official grader maximum_attempts must be positive")
        if self.failure_outcome is not Outcome.UNSCORABLE:
            raise DataValidationError("official grader failures must remain unscorable")
        labels = {str(key).strip(): Outcome(value) for key, value in self.label_mapping.items()}
        if any(not key or any(character.isspace() for character in key) for key in labels):
            raise DataValidationError("official grader labels must be atomic non-empty strings")
        observed = set(labels.values())
        expected = {Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION}
        if self.benchmark == "aa_omniscience_public_600":
            expected.add(Outcome.PARTIAL)
        if observed != expected:
            raise DataValidationError(
                f"official grader label map must cover {sorted(item.value for item in expected)}"
            )
        released = _RELEASED_GRADERS.get(self.benchmark)
        if released is None:
            raise DataValidationError(f"unsupported released grader benchmark {self.benchmark!r}")
        frozen_identity = {
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_artifact": self.source_artifact,
            "source_artifact_sha256": self.source_artifact_sha256,
            "grader_model": self.grader_model,
            "grader_model_revision": self.grader_model_revision,
            "reasoning_enabled": self.reasoning_enabled,
            "prompt_sha256": self.prompt_sha256,
            "label_mapping": labels,
            "maximum_attempts": self.maximum_attempts,
        }
        differing = [
            key
            for key, expected_value in released.items()
            if frozen_identity[key] != expected_value
        ]
        if differing:
            raise DataValidationError(
                f"released grader identity differs in fields: {sorted(differing)}"
            )
        object.__setattr__(self, "prompt_template", prompt)
        object.__setattr__(self, "label_mapping", MappingProxyType(labels))

    @property
    def digest(self) -> str:
        return stable_hash(
            {
                "schema_version": self.schema_version,
                "benchmark": self.benchmark,
                "source_repository": self.source_repository,
                "source_revision": self.source_revision,
                "source_artifact": self.source_artifact,
                "source_artifact_sha256": self.source_artifact_sha256,
                "grader_model": self.grader_model,
                "grader_model_revision": self.grader_model_revision,
                "temperature": self.temperature,
                "reasoning_enabled": self.reasoning_enabled,
                "prompt_sha256": self.prompt_sha256,
                "label_mapping": {key: value.value for key, value in self.label_mapping.items()},
                "maximum_attempts": self.maximum_attempts,
                "failure_outcome": self.failure_outcome.value,
            }
        )

    def verify_source_artifact(self, path: str | Path) -> None:
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise FrozenArtifactError(f"cannot read grader source artifact {path}: {exc}") from exc
        if actual != self.source_artifact_sha256:
            raise FrozenArtifactError(
                f"grader source artifact changed: expected {self.source_artifact_sha256}, "
                f"found {actual}"
            )


@dataclass(frozen=True, slots=True)
class GradingRequest:
    question_id: str
    question: str
    target: str
    predicted_answer: str

    def __post_init__(self) -> None:
        for name in ("question_id", "question", "target"):
            original = getattr(self, name)
            if not isinstance(original, str):
                raise DataValidationError(f"grading request {name} must be text")
            value = original.strip()
            if not value:
                raise DataValidationError(f"grading request {name} must be non-empty")
            object.__setattr__(self, name, value)
        if not isinstance(self.predicted_answer, str):
            raise DataValidationError("grading request predicted_answer must be text")

    @property
    def digest(self) -> str:
        return stable_hash(
            {
                "question_id": self.question_id,
                "question": self.question,
                "target": self.target,
                "predicted_answer": self.predicted_answer,
            }
        )


@dataclass(frozen=True, slots=True)
class OfficialGradeRecord:
    request_fingerprint: str
    grader_fingerprint: str
    outcome: Outcome
    raw_response: str
    attempts: int
    error: str | None

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.request_fingerprint) or not _SHA256.fullmatch(
            self.grader_fingerprint
        ):
            raise DataValidationError("official grade requires request and grader fingerprints")
        if self.attempts <= 0:
            raise DataValidationError("official grade attempts must be positive")
        if self.error is None and self.outcome is Outcome.UNSCORABLE:
            raise DataValidationError("successful official grades cannot be unscorable")
        if self.error is not None and self.outcome is not Outcome.UNSCORABLE:
            raise DataValidationError("failed official grades must remain unscorable")


GraderInvoker = Callable[[str, OfficialGraderSpec], str]


class PermanentGraderError(RuntimeError):
    """A provider/configuration failure that must not consume blind retries."""


def render_grader_prompt(spec: OfficialGraderSpec, request: GradingRequest) -> str:
    """Render once so placeholder-looking answer text is never re-interpreted."""

    replacements = {
        "question": request.question,
        "target": request.target,
        "predicted_answer": request.predicted_answer,
    }
    return _PROMPT_PLACEHOLDER.sub(lambda match: replacements[match.group(1)], spec.prompt_template)


def run_official_grader(
    spec: OfficialGraderSpec,
    request: GradingRequest,
    invoke: GraderInvoker,
) -> OfficialGradeRecord:
    prompt = render_grader_prompt(spec, request)
    last_response = ""
    last_error = "official grader did not run"
    for attempt in range(1, spec.maximum_attempts + 1):
        try:
            response = invoke(prompt, spec)
            if not isinstance(response, str):
                raise DataValidationError("official grader returned a non-text response")
            last_response = response
            label = response.strip()
            outcome = spec.label_mapping.get(label)
            if outcome is None:
                raise DataValidationError(f"official grader returned unknown label {label!r}")
            return OfficialGradeRecord(
                request_fingerprint=request.digest,
                grader_fingerprint=spec.digest,
                outcome=outcome,
                raw_response=response,
                attempts=attempt,
                error=None,
            )
        except Exception as exc:  # provider/parsing failures share the frozen U policy
            last_error = f"{type(exc).__name__}: {exc}"
    return OfficialGradeRecord(
        request_fingerprint=request.digest,
        grader_fingerprint=spec.digest,
        outcome=spec.failure_outcome,
        raw_response=last_response,
        attempts=spec.maximum_attempts,
        error=last_error,
    )


@dataclass(frozen=True, slots=True)
class AAOfficialMetrics:
    counts: Mapping[str, int]
    total: int
    omniscience_index: float
    accuracy: float
    hallucination_rate: float | None


def aa_official_metrics(outcomes: tuple[Outcome, ...]) -> AAOfficialMetrics:
    """Released AA equations, kept separate from the study's unified risk metric.

    Automated labels use the pre-E1 grader amendment in
    ``configs/experiments/grader-selection-amendment.json``.
    """

    if not outcomes:
        raise DataValidationError("AA official metrics require outcomes")
    counter = Counter(Outcome(value) for value in outcomes)
    if counter[Outcome.UNSCORABLE]:
        raise DataValidationError("AA official metrics cannot include unscorable grader failures")
    total = len(outcomes)
    correct = counter[Outcome.CORRECT]
    partial = counter[Outcome.PARTIAL]
    incorrect = counter[Outcome.INCORRECT]
    abstained = counter[Outcome.ABSTENTION]
    denominator = partial + incorrect + abstained
    return AAOfficialMetrics(
        counts={outcome.value: counter[outcome] for outcome in Outcome},
        total=total,
        omniscience_index=100 * (correct - incorrect) / total,
        accuracy=correct / total,
        hallucination_rate=incorrect / denominator if denominator else None,
    )


def simpleqa_official_metrics(outcomes: tuple[Outcome, ...]) -> MetricBundle:
    if not outcomes or any(
        Outcome(value) not in {Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION}
        for value in outcomes
    ):
        raise DataValidationError("SimpleQA official metrics require only C/I/A outcomes")
    return metric_bundle(outcomes, partial_credit=0.0)


_GRADER_KEYS = {
    "benchmark",
    "source_repository",
    "source_revision",
    "source_artifact",
    "source_artifact_sha256",
    "grader_model",
    "grader_model_revision",
    "temperature",
    "reasoning_enabled",
    "prompt_path",
    "prompt_sha256",
    "label_mapping",
    "maximum_attempts",
    "failure_outcome",
}


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be non-empty text")
    return value.strip()


def load_official_grader_spec(path: str | Path) -> OfficialGraderSpec:
    """Load a frozen official-grader config and its sibling prompt asset."""

    source = Path(path)
    raw = load_yaml(source)
    if set(raw) != {"schema_version", "grader"} or raw.get("schema_version") != 1:
        raise ConfigurationError(
            f"{source}: official grader must contain schema_version 1 and grader"
        )
    section = raw.get("grader")
    if not isinstance(section, Mapping) or set(section) != _GRADER_KEYS:
        observed = set(section) if isinstance(section, Mapping) else set()
        raise ConfigurationError(
            f"{source}: grader keys differ; "
            f"missing={sorted(_GRADER_KEYS - observed)}, "
            f"unknown={sorted(observed - _GRADER_KEYS)}"
        )
    prompt_name = _require_text(section["prompt_path"], f"{source}:grader.prompt_path")
    prompt_relative = Path(prompt_name)
    if prompt_relative.is_absolute() or len(prompt_relative.parts) != 1:
        raise ConfigurationError(f"{source}: grader prompt must be a sibling filename")
    try:
        prompt = (source.parent / prompt_relative).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"{source}: cannot read grader prompt: {exc}") from exc
    labels = section["label_mapping"]
    if not isinstance(labels, Mapping):
        raise ConfigurationError(f"{source}: grader.label_mapping must be a mapping")
    reasoning = section["reasoning_enabled"]
    attempts = section["maximum_attempts"]
    temperature = section["temperature"]
    if not isinstance(reasoning, bool):
        raise ConfigurationError(f"{source}: grader.reasoning_enabled must be a boolean")
    if not isinstance(attempts, int) or isinstance(attempts, bool):
        raise ConfigurationError(f"{source}: grader.maximum_attempts must be an integer")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ConfigurationError(f"{source}: grader.temperature must be numeric")
    text_fields = {
        key: _require_text(section[key], f"{source}:grader.{key}")
        for key in (
            "benchmark",
            "source_repository",
            "source_revision",
            "source_artifact",
            "source_artifact_sha256",
            "grader_model",
            "grader_model_revision",
            "prompt_sha256",
            "failure_outcome",
        )
    }
    try:
        return OfficialGraderSpec(
            benchmark=text_fields["benchmark"],
            source_repository=text_fields["source_repository"],
            source_revision=text_fields["source_revision"],
            source_artifact=text_fields["source_artifact"],
            source_artifact_sha256=text_fields["source_artifact_sha256"],
            grader_model=text_fields["grader_model"],
            grader_model_revision=text_fields["grader_model_revision"],
            temperature=float(temperature),
            reasoning_enabled=reasoning,
            prompt_template=prompt,
            prompt_sha256=text_fields["prompt_sha256"],
            label_mapping={str(key): Outcome(value) for key, value in labels.items()},
            maximum_attempts=attempts,
            failure_outcome=Outcome(text_fields["failure_outcome"]),
        )
    except (TypeError, ValueError, DataValidationError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"invalid official-grader configuration {source}: {exc}") from exc
