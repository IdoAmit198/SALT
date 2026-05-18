import os
import sys
import warnings
from types import SimpleNamespace

from salt_benchmark.generation import (
    GenerationConfig,
    VLLMGenerator,
    load_generation_cache,
    render_hf_prompts,
    save_generation_cache,
)
from salt_benchmark.tutorial_utils import configure_quiet_generation_logs, quiet_call


class FakeTokenizer:
    chat_template = "{% if system %}system{% endif %} user"

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        roles = ",".join(message["role"] for message in chat)
        content = chat[-1]["content"]
        suffix = "<assistant>" if add_generation_prompt else ""
        return f"roles={roles}; {content} {suffix}"


class FallbackTokenizer:
    chat_template = "template"

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        if any(message["role"] == "system" for message in chat):
            raise ValueError("system role unsupported")
        return f"user-only: {chat[-1]['content']}"


class NoTemplateTokenizer:
    chat_template = None


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params):
        self.calls.append((prompts, sampling_params))
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=f" answer for {prompt} ", logprobs=[{"token": -0.1}])])
            for prompt in prompts
        ]


def test_render_hf_prompts_uses_chat_template():
    prompts = render_hf_prompts(FakeTokenizer(), ["solve this"])
    assert prompts == ["roles=system,user; solve this <assistant>"]


def test_render_hf_prompts_retries_without_system():
    prompts = render_hf_prompts(FallbackTokenizer(), ["solve this"])
    assert prompts == ["user-only: solve this"]


def test_render_hf_prompts_without_template_returns_raw_instruction():
    prompts = render_hf_prompts(NoTemplateTokenizer(), ["raw prompt"])
    assert prompts == ["raw prompt"]


def test_vllm_generator_with_fake_backend():
    fake_llm = FakeLLM()
    generator = VLLMGenerator(
        "fake/model",
        llm=fake_llm,
        tokenizer=NoTemplateTokenizer(),
        sampling_params_factory=lambda **kwargs: kwargs,
    )
    results = generator.generate(
        ["prompt one", "prompt two"],
        inputs=["input one", "input two"],
        config=GenerationConfig(max_tokens=16, temperature=0.0, logprobs=3),
    )
    assert [result.input for result in results] == ["input one", "input two"]
    assert results[0].text == "answer for prompt one"
    assert results[0].logprobs == [{"token": -0.1}]
    assert fake_llm.calls[0][1]["max_tokens"] == 16
    assert fake_llm.calls[0][1]["logprobs"] == 3


def test_generation_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "generation.json"
    payload = {"model": "fake/model", "responses": [{"text": "hello"}]}
    save_generation_cache(cache_path, payload)
    assert load_generation_cache(cache_path) == payload
    assert load_generation_cache(tmp_path / "missing.json") is None


def test_quiet_call_suppresses_python_and_file_descriptor_output(capfd):
    def action():
        print("python stdout")
        print("python stderr", file=sys.stderr)
        os.write(1, b"fd stdout\n")
        os.write(2, b"fd stderr\n")
        return "ok"

    assert quiet_call(action, "demo") == "ok"
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_configure_quiet_generation_logs_suppresses_tqdm_iprogress_warning():
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("default")
        configure_quiet_generation_logs()
        warnings.warn("IProgress not found. Please update jupyter and ipywidgets.", Warning)

    assert caught_warnings == []
