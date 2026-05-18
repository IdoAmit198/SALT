from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .chat_templates import render_hf_prompts


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling settings passed to vLLM SamplingParams."""

    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 42
    logprobs: int | None = 5


@dataclass(frozen=True)
class GenerationResult:
    """One generated completion and the prompt/input that produced it."""

    input: str | None
    prompt: str
    text: str
    logprobs: Any = None


class VLLMGenerator:
    """Minimal vLLM generation wrapper for regular SALT prompts."""

    def __init__(
        self,
        model_name: str,
        *,
        llm=None,
        tokenizer=None,
        llm_factory: Callable[..., Any] | None = None,
        tokenizer_factory: Callable[[str], Any] | None = None,
        sampling_params_factory: Callable[..., Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self._llm = llm
        self._tokenizer = tokenizer
        self._llm_factory = llm_factory
        self._tokenizer_factory = tokenizer_factory
        self._sampling_params_factory = sampling_params_factory
        self._llm_kwargs = llm_kwargs or {}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self._tokenizer_factory is None:
                from transformers import AutoTokenizer

                self._tokenizer_factory = AutoTokenizer.from_pretrained
            self._tokenizer = self._tokenizer_factory(self.model_name)
        return self._tokenizer

    @property
    def llm(self):
        if self._llm is None:
            if self._llm_factory is None:
                from vllm import LLM

                self._llm_factory = LLM
            self._llm = self._llm_factory(model=self.model_name, **self._llm_kwargs)
        return self._llm

    def render_prompts(self, instructions: list[str]) -> list[str]:
        """Render regular SALT instructions through the model tokenizer chat template."""
        return render_hf_prompts(self.tokenizer, instructions)

    def generate(
        self,
        instructions: list[str],
        *,
        inputs: list[str] | None = None,
        config: GenerationConfig | None = None,
        prompts_are_rendered: bool = False,
    ) -> list[GenerationResult]:
        """Generate completions for regular or already-rendered prompts.

        Args:
            instructions: Raw SALT instructions or rendered prompts.
            inputs: Optional original task inputs to attach to returned results.
            config: Sampling settings. Defaults to ``GenerationConfig()``.
            prompts_are_rendered: Set True when ``instructions`` are already model-ready prompts.

        Returns:
            A list of GenerationResult objects aligned to vLLM outputs.
        """
        config = config or GenerationConfig()
        prompts = instructions if prompts_are_rendered else self.render_prompts(instructions)
        sampling_params = self._make_sampling_params(config)
        outputs = self.llm.generate(prompts, sampling_params)
        normalized_inputs = inputs or [None] * len(prompts)
        return [
            GenerationResult(
                input=normalized_inputs[index] if index < len(normalized_inputs) else None,
                prompt=prompt,
                text=_extract_text(output),
                logprobs=_extract_logprobs(output),
            )
            for index, (prompt, output) in enumerate(zip(prompts, outputs))
        ]

    def _make_sampling_params(self, config: GenerationConfig):
        kwargs = asdict(config)
        if self._sampling_params_factory is not None:
            return self._sampling_params_factory(**kwargs)
        from vllm import SamplingParams

        return SamplingParams(**kwargs)


def _extract_text(output) -> str:
    candidates = getattr(output, "outputs", None) or []
    if not candidates:
        return ""
    return str(getattr(candidates[0], "text", "")).strip()


def _extract_logprobs(output):
    candidates = getattr(output, "outputs", None) or []
    if not candidates:
        return None
    return getattr(candidates[0], "logprobs", None)
