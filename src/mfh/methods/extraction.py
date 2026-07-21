"""Data-collection workflows for M1-R, M1-P, and paired CAA vectors."""

from __future__ import annotations

from enum import StrEnum

from mfh.contracts import Outcome
from mfh.errors import DataValidationError
from mfh.inference.architecture import HookPoint
from mfh.inference.hooks import ActivationSession, CapturePolicy
from mfh.inference.runtime import RenderedPrompt, TransformersRuntime
from mfh.methods.static import CentroidVectorBuilder, PairedDifferenceBuilder, VectorBank


class CentroidExtractionMode(StrEnum):
    RESPONSE_TOKENS = "M1-R"
    PROMPT_FINAL = "M1-P"


class CentroidExtractor:
    def __init__(
        self,
        points: tuple[HookPoint, ...],
        *,
        mode: CentroidExtractionMode,
    ) -> None:
        self.points = points
        self.mode = mode
        self.builder = CentroidVectorBuilder()

    def observe(
        self,
        runtime: TransformersRuntime,
        rendered_prompt: RenderedPrompt,
        *,
        outcome: Outcome,
        response: str | None = None,
    ) -> None:
        if outcome not in {Outcome.CORRECT, Outcome.INCORRECT}:
            return
        if self.mode is CentroidExtractionMode.PROMPT_FINAL:
            session = ActivationSession(self.points, capture_policy=CapturePolicy.PROMPT_FINAL)
            activations = runtime.prompt_activations(rendered_prompt, session)
        else:
            if response is None or not response.strip():
                raise DataValidationError("M1-R extraction requires a non-empty model response")
            session = ActivationSession(self.points, capture_policy=CapturePolicy.RESPONSE_TOKENS)
            activations = runtime.teacher_forced_activations(rendered_prompt, response, session)
        self.builder.update(outcome, activations)

    def build(self, *, data_fingerprint: str) -> VectorBank:
        return self.builder.build(
            source_method=self.mode.value,
            data_fingerprint=data_fingerprint,
        )


class CAAExtractor:
    """Mean-pool variable-length response activations before paired subtraction."""

    def __init__(self, points: tuple[HookPoint, ...]) -> None:
        self.points = points
        self.builder = PairedDifferenceBuilder()

    def observe_pair(
        self,
        runtime: TransformersRuntime,
        rendered_prompt: RenderedPrompt,
        *,
        positive_response: str,
        negative_response: str,
    ) -> None:
        if not positive_response.strip() or not negative_response.strip():
            raise DataValidationError("CAA responses must be non-empty")
        positive_session = ActivationSession(
            self.points, capture_policy=CapturePolicy.RESPONSE_TOKENS
        )
        negative_session = ActivationSession(
            self.points, capture_policy=CapturePolicy.RESPONSE_TOKENS
        )
        positive = runtime.teacher_forced_activations(
            rendered_prompt, positive_response, positive_session
        )
        negative = runtime.teacher_forced_activations(
            rendered_prompt, negative_response, negative_session
        )
        positive_pooled = {
            key: values.mean(dim=0, keepdim=True) for key, values in positive.items()
        }
        negative_pooled = {
            key: values.mean(dim=0, keepdim=True) for key, values in negative.items()
        }
        self.builder.update(positive_pooled, negative_pooled)

    def build(self, *, data_fingerprint: str) -> VectorBank:
        return self.builder.build(data_fingerprint=data_fingerprint)
