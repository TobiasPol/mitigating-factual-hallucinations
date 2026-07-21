"""Human-reviewed, signed translations of a fixed TriviaQA subset."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import Question
from mfh.data.source_snapshots import (
    SOURCE_SNAPSHOTS,
    source_question_index,
    verify_source_artifact,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE = re.compile(r"^[0-9a-f]{128}$")
_LANGUAGES = ("de", "en", "es", "fr", "ja")
_INSTRUCTIONS = {
    "de": "Antworte auf Deutsch.",
    "en": "Answer in English.",
    "es": "Responde en español.",
    "fr": "Répondez en français.",
    "ja": "日本語で答えてください。",
}
_SUITE_ID = "mfh-human-reviewed-triviaqa-languages-v1"


def _source_identity(question: Question) -> str:
    return stable_hash(
        {
            "question_id": question.question_id,
            "benchmark": question.benchmark,
            "text": question.text,
            "aliases": list(question.aliases),
            "split": question.split,
            "entities": list(question.entities),
            "metadata": dict(question.metadata),
        }
    )


def translation_review_body(
    *,
    source_question: Question,
    language: str,
    translated_prompt: str,
    reviewer_id: str,
) -> Mapping[str, Any]:
    """Return the exact human-review claim signed for one translated prompt."""

    language = language.strip()
    translated_prompt = translated_prompt.strip()
    reviewer_id = reviewer_id.strip()
    if (
        language not in _LANGUAGES
        or not translated_prompt
        or not reviewer_id
        or not translated_prompt.endswith(_INSTRUCTIONS[language])
    ):
        raise DataValidationError("translated-language review fields are invalid")
    return MappingProxyType(
        {
            "schema_version": 1,
            "suite_id": _SUITE_ID,
            "source_question_id": source_question.question_id,
            "source_question_sha256": _source_identity(source_question),
            "language": language,
            "translated_prompt": translated_prompt,
            "reviewer_id": reviewer_id,
            "review_claims": {
                "semantic_equivalence": True,
                "answer_aliases_preserved": True,
                "requested_language_explicit": True,
            },
        }
    )


def sign_translation_review(
    *,
    source_question: Question,
    language: str,
    translated_prompt: str,
    reviewer_id: str,
    private_key: Ed25519PrivateKey,
) -> str:
    """Sign one review after the human has checked all declared claims."""

    body = translation_review_body(
        source_question=source_question,
        language=language,
        translated_prompt=translated_prompt,
        reviewer_id=reviewer_id,
    )
    return private_key.sign(canonical_json(body).encode("utf-8")).hex()


def _contains_non_latin_script(text: str) -> bool:
    return any(
        character.isalpha() and "LATIN" not in unicodedata.name(character, "") for character in text
    )


def _validate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_questions: Mapping[str, Question],
    reviewer_public_keys: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    reviewers = {str(key).strip(): str(value) for key, value in reviewer_public_keys.items()}
    if len(reviewers) < 2 or any(
        not reviewer or not _PUBLIC_KEY.fullmatch(key) for reviewer, key in reviewers.items()
    ):
        raise DataValidationError("language suite requires at least two valid reviewer keys")
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema_version",
            "source_question_id",
            "language",
            "translated_prompt",
            "reviews",
        }:
            raise DataValidationError("translated-language row has an invalid schema")
        source_id = raw["source_question_id"]
        language = raw["language"]
        prompt = raw["translated_prompt"]
        reviews = raw["reviews"]
        if (
            raw["schema_version"] != 1
            or not isinstance(source_id, str)
            or source_id not in source_questions
            or not isinstance(language, str)
            or language not in _LANGUAGES
            or not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(reviews, list)
            or len(reviews) != 2
        ):
            raise DataValidationError("translated-language row fields are invalid")
        key = (source_id, language)
        if key in seen:
            raise DataValidationError("translated-language suite repeats a source/language row")
        seen.add(key)
        reviewer_ids: set[str] = set()
        normalized_reviews: list[dict[str, str]] = []
        for review in reviews:
            if not isinstance(review, Mapping) or set(review) != {"reviewer_id", "signature"}:
                raise DataValidationError("translation review receipt has an invalid schema")
            reviewer = review["reviewer_id"]
            signature = review["signature"]
            if (
                not isinstance(reviewer, str)
                or reviewer not in reviewers
                or reviewer in reviewer_ids
                or not isinstance(signature, str)
                or not _SIGNATURE.fullmatch(signature)
            ):
                raise DataValidationError("translation review receipt identity is invalid")
            body = translation_review_body(
                source_question=source_questions[source_id],
                language=language,
                translated_prompt=prompt,
                reviewer_id=reviewer,
            )
            try:
                public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(reviewers[reviewer]))
                public_key.verify(bytes.fromhex(signature), canonical_json(body).encode("utf-8"))
            except (InvalidSignature, ValueError) as exc:
                raise DataValidationError("translation review signature is invalid") from exc
            reviewer_ids.add(reviewer)
            normalized_reviews.append({"reviewer_id": reviewer, "signature": signature})
        if language == "en" and not _contains_non_latin_script(prompt):
            raise DataValidationError(
                "English language-suite prompts must contain a non-Latin-script name"
            )
        parsed.append(
            {
                "schema_version": 1,
                "source_question_id": source_id,
                "language": language,
                "translated_prompt": prompt.strip(),
                "reviews": sorted(normalized_reviews, key=lambda value: value["reviewer_id"]),
            }
        )
    source_ids = {value["source_question_id"] for value in parsed}
    expected_pairs = {(source_id, language) for source_id in source_ids for language in _LANGUAGES}
    if len(source_ids) != 100 or seen != expected_pairs or len(parsed) != 500:
        raise DataValidationError(
            "language suite must contain the same 100 TriviaQA rows in all five languages"
        )
    return tuple(sorted(parsed, key=lambda value: (value["language"], value["source_question_id"])))


def _load_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise DataValidationError(
                        f"translated-language row {line_number} must be a mapping"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read translated-language rows: {exc}") from exc
    return tuple(rows)


def _manifest(source: Path) -> dict[str, Any]:
    try:
        payload = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read language-suite manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError("language-suite manifest must be a mapping")
    digest = payload.pop("manifest_digest", None)
    if digest != stable_hash(payload):
        raise DataValidationError("language-suite manifest digest mismatch")
    payload["manifest_digest"] = digest
    return payload


def load_reviewed_language_suite(path: str | Path) -> tuple[Question, ...]:
    """Verify source bytes and both human signatures before reconstructing questions."""

    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()}
        != {
            "manifest.json",
            "translations.jsonl",
            "triviaqa-source.parquet",
        }
    ):
        raise DataValidationError("reviewed language suite has invalid top-level files")
    manifest = _manifest(source)
    if (
        set(manifest)
        != {
            "schema_version",
            "suite_id",
            "source",
            "reviewer_public_keys",
            "translations_sha256",
            "language_counts",
            "source_question_count",
            "manifest_digest",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("suite_id") != _SUITE_ID
    ):
        raise DataValidationError("reviewed language-suite manifest has an invalid schema")
    snapshot = SOURCE_SNAPSHOTS["triviaqa"]
    source_descriptor = manifest["source"]
    if not isinstance(source_descriptor, Mapping) or dict(source_descriptor) != {
        "repository": snapshot.repository,
        "revision": snapshot.revision,
        "split": snapshot.split,
        "artifact_sha256": snapshot.artifact_sha256,
        "artifact_size_bytes": snapshot.artifact_size_bytes,
    }:
        raise DataValidationError("reviewed language suite uses the wrong TriviaQA source")
    trivia_source = verify_source_artifact(snapshot, source / "triviaqa-source.parquet")
    translations_path = source / "translations.jsonl"
    if sha256_file(translations_path) != manifest["translations_sha256"]:
        raise DataValidationError("reviewed language-suite translations changed")
    source_questions = source_question_index(snapshot, trivia_source)
    reviewers = manifest["reviewer_public_keys"]
    if not isinstance(reviewers, Mapping):
        raise DataValidationError("language-suite reviewer registry is invalid")
    rows = _validate_rows(
        _load_rows(translations_path),
        source_questions=source_questions,
        reviewer_public_keys={str(key): str(value) for key, value in reviewers.items()},
    )
    counts = {
        language: sum(value["language"] == language for value in rows) for language in _LANGUAGES
    }
    if manifest["language_counts"] != counts or manifest["source_question_count"] != 100:
        raise DataValidationError("reviewed language-suite counts differ from its rows")
    revision = str(manifest["manifest_digest"])
    result: list[Question] = []
    for row in rows:
        original = source_questions[row["source_question_id"]]
        language = row["language"]
        source_row_id = str(original.metadata.get("source_row_id", original.question_id))
        result.append(
            Question(
                question_id=f"language_consistency:{language}:{source_row_id}",
                benchmark="language_consistency",
                text=row["translated_prompt"],
                aliases=original.aliases,
                split="evaluation",
                entities=original.entities,
                metadata={
                    "requested_language": language,
                    "suite_id": _SUITE_ID,
                    "translation_suite_revision": revision,
                    "source_repository": snapshot.repository,
                    "source_revision": snapshot.revision,
                    "source_split": snapshot.split,
                    "source_row_id": source_row_id,
                    "source_question_id": original.question_id,
                    "source_question_sha256": _source_identity(original),
                    "human_reviewers": [value["reviewer_id"] for value in row["reviews"]],
                },
            )
        )
    return tuple(result)


def write_reviewed_language_suite(
    destination: str | Path,
    *,
    triviaqa_source: str | Path,
    rows: Iterable[Mapping[str, Any]],
    reviewer_public_keys: Mapping[str, str],
) -> str:
    """Package the fixed 100-question, five-language, two-reviewer suite atomically."""

    target = validate_active_study_artifact_paths(
        {"reviewed-language-suite": destination}
    )["reviewed-language-suite"]
    if target.exists():
        raise FrozenArtifactError(f"refusing to overwrite language suite: {target}")
    snapshot = SOURCE_SNAPSHOTS["triviaqa"]
    trivia_source = verify_source_artifact(snapshot, triviaqa_source)
    source_questions = source_question_index(snapshot, trivia_source)
    parsed = _validate_rows(
        rows,
        source_questions=source_questions,
        reviewer_public_keys=reviewer_public_keys,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        shutil.copyfile(trivia_source, stage / "triviaqa-source.parquet")
        translations = stage / "translations.jsonl"
        translations.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in parsed),
            encoding="utf-8",
        )
        body = {
            "schema_version": 1,
            "suite_id": _SUITE_ID,
            "source": {
                "repository": snapshot.repository,
                "revision": snapshot.revision,
                "split": snapshot.split,
                "artifact_sha256": snapshot.artifact_sha256,
                "artifact_size_bytes": snapshot.artifact_size_bytes,
            },
            "reviewer_public_keys": dict(sorted(reviewer_public_keys.items())),
            "translations_sha256": sha256_file(translations),
            "language_counts": {
                language: sum(value["language"] == language for value in parsed)
                for language in _LANGUAGES
            },
            "source_question_count": 100,
        }
        (stage / "manifest.json").write_text(
            json.dumps({**body, "manifest_digest": stable_hash(body)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        load_reviewed_language_suite(stage)
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sha256_path(target)
