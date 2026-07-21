"""Deterministic text-only Transformers runtime with explicit cache steps."""

from __future__ import annotations

import hashlib
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from mfh.contracts import ModelSpec, PromptSpec, Runtime, TransformersModelClass
from mfh.errors import ConfigurationError, DataValidationError, OptionalDependencyError
from mfh.inference.architecture import HookKey
from mfh.inference.hooks import ActivationSession
from mfh.inference.transformers_snapshot import reject_symlink_path_components


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    sha256: str
    messages: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    rendered_prompt: RenderedPrompt
    token_ids: tuple[int, ...]
    raw_text: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    stop_type: str
    stopping_token_id: int | None


@dataclass(frozen=True, slots=True)
class TeacherForcedScore:
    response: str
    token_ids: tuple[int, ...]
    total_log_likelihood: float
    mean_log_likelihood: float


def set_deterministic_seed(seed: int) -> None:
    if seed < 0:
        raise ConfigurationError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class TransformersRuntime:
    def __init__(
        self,
        *,
        model: nn.Module,
        processor: Any,
        model_spec: ModelSpec,
        seed: int = 17,
    ) -> None:
        if model_spec.runtime is not Runtime.TRANSFORMERS:
            raise ConfigurationError("TransformersRuntime requires a transformers ModelSpec")
        self.model = model
        self.processor = processor
        self.model_spec = model_spec
        self.seed = seed
        self.model.eval()
        set_deterministic_seed(seed)

    @classmethod
    def from_spec(
        cls,
        model_spec: ModelSpec,
        *,
        device_map: str | dict[str, Any] = "auto",
        allow_download: bool = False,
        snapshot_path: str | Path | None = None,
        seed: int = 17,
    ) -> TransformersRuntime:
        """Load an exact revision or a previously verified local snapshot."""

        if model_spec.runtime is not Runtime.TRANSFORMERS:
            raise ConfigurationError("only Transformers checkpoints can be loaded here")
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise OptionalDependencyError(
                "model loading requires the research extra: uv sync --extra research"
            ) from exc
        if snapshot_path is None:
            source = model_spec.repository
            common = {
                "revision": model_spec.revision,
                "trust_remote_code": model_spec.trust_remote_code,
                "local_files_only": not allow_download,
            }
        else:
            snapshot = reject_symlink_path_components(snapshot_path, "local Transformer snapshot")
            if snapshot.is_symlink() or not snapshot.is_dir():
                raise ConfigurationError("local Transformer snapshot must be a regular directory")
            source = str(snapshot)
            common = {
                "trust_remote_code": model_spec.trust_remote_code,
                "local_files_only": True,
            }
        processor = AutoTokenizer.from_pretrained(source, **common)
        model_loader = (
            AutoModelForImageTextToText
            if model_spec.transformers_model_class is TransformersModelClass.IMAGE_TEXT_TO_TEXT
            else AutoModelForCausalLM
        )
        dtype: str | torch.dtype
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(model_spec.dtype, model_spec.dtype)
        model = model_loader.from_pretrained(
            source,
            dtype=dtype,
            device_map=device_map,
            **common,
        )
        loaded_revision = getattr(model.config, "_commit_hash", None)
        if loaded_revision is not None and loaded_revision != model_spec.revision:
            raise ConfigurationError(
                f"loaded model revision {loaded_revision!r} differs from {model_spec.revision!r}"
            )
        return cls(model=model, processor=processor, model_spec=model_spec, seed=seed)

    @property
    def input_device(self) -> torch.device:
        getter = getattr(self.model, "get_input_embeddings", None)
        if not callable(getter):
            raise ConfigurationError("model exposes no input embedding accessor")
        embeddings = getter()
        if not isinstance(embeddings, nn.Module) or not isinstance(
            getattr(embeddings, "weight", None), Tensor
        ):
            raise ConfigurationError("model input embeddings expose no tensor weight")
        return cast(Tensor, embeddings.weight).device

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RenderedPrompt:
        try:
            system_text = prompt.text.format_map(metadata or {})
        except KeyError as exc:
            raise DataValidationError(
                f"prompt {prompt.prompt_id!r} requires missing metadata field {exc.args[0]!r}"
            ) from exc
        messages = (
            {"role": "system", "content": system_text},
            {"role": "user", "content": question},
        )
        rendered = self.processor.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str) or not rendered:
            raise DataValidationError("chat template returned an empty or non-text prompt")
        return RenderedPrompt(
            text=rendered,
            sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            messages=messages,
        )

    def _encode(self, text: str) -> dict[str, Tensor]:
        encoded = self.processor(text=text, return_tensors="pt")
        if not isinstance(encoded, dict) and not hasattr(encoded, "items"):
            raise DataValidationError("processor output must be mapping-like")
        tensors = {
            key: value.to(self.input_device)
            for key, value in encoded.items()
            if isinstance(value, Tensor)
        }
        if "input_ids" not in tensors:
            raise DataValidationError("processor output contains no input_ids")
        if tensors["input_ids"].ndim != 2 or tensors["input_ids"].shape[0] != 1:
            raise DataValidationError("runtime currently requires one text prompt per call")
        tensors.setdefault("attention_mask", torch.ones_like(tensors["input_ids"]))
        return tensors

    def _eos_ids(self) -> set[int]:
        candidates = [
            getattr(self.processor, "eos_token_id", None),
            getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None),
            getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
        ]
        result: set[int] = set()
        for candidate in candidates:
            if isinstance(candidate, int):
                result.add(candidate)
            elif isinstance(candidate, (list, tuple)):
                result.update(int(value) for value in candidate)
        return result

    @staticmethod
    def _logits(output: Any) -> Tensor:
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise DataValidationError("model output contains no [batch, sequence, vocab] logits")
        return logits

    def generate(
        self,
        rendered_prompt: RenderedPrompt,
        *,
        max_new_tokens: int = 48,
        session: ActivationSession | None = None,
    ) -> GenerationOutput:
        if not 1 <= max_new_tokens <= 48:
            raise ConfigurationError("primary generation length must be in [1, 48]")
        encoded = self._encode(rendered_prompt.text)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        prompt_length = int(input_ids.shape[1])
        generated: list[int] = []
        context = session if session is not None else nullcontext()
        start = time.perf_counter()
        if session is not None:
            session.clear()
        with torch.inference_mode(), context:
            if session is not None:
                session.set_prompt(prompt_length)
            output = self.model(**encoded, use_cache=True)
            past = getattr(output, "past_key_values", None)
            if past is None:
                raise DataValidationError(
                    "model did not return a KV cache for deterministic decoding"
                )
            next_token = torch.argmax(self._logits(output)[:, -1, :], dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            eos_ids = self._eos_ids()

            while len(generated) < max_new_tokens and token_id not in eos_ids:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (1, 1), device=attention_mask.device, dtype=attention_mask.dtype
                        ),
                    ],
                    dim=1,
                )
                if session is not None:
                    session.set_generated(
                        prompt_length=prompt_length,
                        generation_step=len(generated) - 1,
                    )
                output = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = getattr(output, "past_key_values", None)
                if past is None:
                    raise DataValidationError("model stopped returning its KV cache")
                next_token = torch.argmax(self._logits(output)[:, -1, :], dim=-1, keepdim=True)
                token_id = int(next_token.item())
                generated.append(token_id)
        latency = time.perf_counter() - start
        raw_text = self.processor.decode(generated, skip_special_tokens=False)
        text = self.processor.decode(generated, skip_special_tokens=True)
        stop_type = "eos" if generated and generated[-1] in eos_ids else "limit"
        return GenerationOutput(
            rendered_prompt=rendered_prompt,
            token_ids=tuple(generated),
            raw_text=str(raw_text),
            text=str(text).strip(),
            input_tokens=prompt_length,
            output_tokens=len(generated),
            latency_seconds=latency,
            stop_type=stop_type,
            stopping_token_id=(generated[-1] if stop_type == "eos" else None),
        )

    def prompt_activations(
        self, rendered_prompt: RenderedPrompt, session: ActivationSession
    ) -> dict[HookKey, Tensor]:
        encoded = self._encode(rendered_prompt.text)
        prompt_length = int(encoded["input_ids"].shape[1])
        session.clear()
        with torch.inference_mode(), session:
            session.set_prompt(prompt_length)
            self.model(**encoded, use_cache=False)
        return session.activations()

    def _teacher_forced_inputs(
        self, rendered_prompt: RenderedPrompt, response: str
    ) -> tuple[dict[str, Tensor], int, Tensor]:
        prompt = self._encode(rendered_prompt.text)
        prompt_ids = prompt["input_ids"]
        full_ids = self._encode(rendered_prompt.text + response)["input_ids"]
        prompt_length = int(prompt_ids.shape[1])
        if full_ids.shape[1] <= prompt_length or not torch.equal(
            full_ids[:, :prompt_length], prompt_ids
        ):
            raise DataValidationError(
                "prompt tokenization is not a stable prefix of prompt-plus-response; "
                "teacher-forced likelihood would be misaligned"
            )
        response_ids = full_ids[:, prompt_length:]
        return (
            {
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids),
            },
            prompt_length,
            response_ids,
        )

    def teacher_forced_activations(
        self,
        rendered_prompt: RenderedPrompt,
        response: str,
        session: ActivationSession,
    ) -> dict[HookKey, Tensor]:
        inputs, prompt_length, _ = self._teacher_forced_inputs(rendered_prompt, response)
        session.clear()
        with torch.inference_mode(), session:
            session.set_teacher_forced(prompt_length=prompt_length, response_start=prompt_length)
            self.model(**inputs, use_cache=False)
        return session.activations()

    def score_response(
        self,
        rendered_prompt: RenderedPrompt,
        response: str,
        *,
        session: ActivationSession | None = None,
    ) -> TeacherForcedScore:
        inputs, prompt_length, response_ids = self._teacher_forced_inputs(rendered_prompt, response)
        if session is not None:
            session.clear()
        context = session if session is not None else nullcontext()
        with torch.inference_mode(), context:
            if session is not None:
                session.set_teacher_forced(
                    prompt_length=prompt_length, response_start=prompt_length
                )
            output = self.model(**inputs, use_cache=False)
            logits = self._logits(output)
            answer_logits = logits[:, prompt_length - 1 : -1, :].float()
            log_probabilities = torch.log_softmax(answer_logits, dim=-1)
            token_log_likelihoods = log_probabilities.gather(
                dim=-1, index=response_ids.unsqueeze(-1)
            ).squeeze(-1)
        total = float(token_log_likelihoods.sum().cpu())
        return TeacherForcedScore(
            response=response,
            token_ids=tuple(int(value) for value in response_ids.squeeze(0).tolist()),
            total_log_likelihood=total,
            mean_log_likelihood=total / response_ids.shape[1],
        )

    def score_aliases(
        self,
        rendered_prompt: RenderedPrompt,
        aliases: tuple[str, ...],
        *,
        session_factory: Any | None = None,
    ) -> tuple[TeacherForcedScore, ...]:
        if not aliases:
            raise DataValidationError("at least one answer alias is required")
        results = []
        for alias in aliases:
            session = session_factory() if session_factory is not None else None
            results.append(self.score_response(rendered_prompt, alias, session=session))
        return tuple(results)
