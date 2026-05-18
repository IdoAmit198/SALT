# SALT

Code for the paper "[Evaluating LLM Uncertainty in Long-Form Generation Using Deterministic Ground Truth](https://icml.cc/virtual/2026/poster/61198)" (ICML 2026).

SALT, Single-answer Atomic Long-form Target, is a benchmark for evaluating long-form model generations with deterministic ground truth. The repository provides procedural dataset creation, task prompt construction, fenced generation helpers, and atomic evaluation utilities for comparing generated answers against known references at `atom`, `line`, and `generation` granularity.

## What Is Included

- Reproduction datasets under [data/reproduction](data/reproduction)
- HuggingFace `Dataset` creators for DNA Translation, Code, First-Order Logic, Matrix Multiplication, Kronecker Product, and Words Collection tasks
- Task classes with prompt builders, delimiters, regex metadata, and deterministic solvers
- Fenced-answer extraction and unit-level evaluation helpers
- Logprob-based confidence functions and uncertainty metrics for ECE, AUROC, and PRR
- A small vLLM wrapper for regular SALT prompts
- Versioned JSON generation cache helpers
- A tutorial notebook with an Advanced section on adding custom tasks: [tutorial.ipynb](tutorial.ipynb)

## Repository Layout

```text
SALT/
  data/reproduction/          # Datasets used for paper reproduction
  salt_benchmark/
    datasets.py               # Public dataset creation API
    evaluation/               # Fenced extraction, atomic evaluation, uncertainty metrics
    generation/                # Chat-template rendering, vLLM wrapper, logprob confidence, JSON cache
    tasks/                    # Task classes, prompt builders, and solvers
    tutorial_utils.py          # Display helpers used by the tutorial notebook
  tests/                      # Unit tests
  local_tests/                 # Optional gitignored local backend regression tests
  tutorial.ipynb              # End-to-end tutorial
  camera_ready.tex            # Paper source
  SALT_Benchmark_paper.pdf    # Paper PDF
  README.md
  LICENSE
  requirements.txt
  pyproject.toml
```

## Backend Package And Tutorial

The importable backend package is [salt_benchmark](salt_benchmark). The package contains the reusable code used by scripts, tests, and the notebook:

- [salt_benchmark/tasks](salt_benchmark/tasks): `Problem` classes, prompt builders, task solvers, and atom-decomposition metadata
- [salt_benchmark/datasets.py](salt_benchmark/datasets.py): HuggingFace `Dataset` factory functions built from those task solvers
- [salt_benchmark/evaluation](salt_benchmark/evaluation): fenced-answer extraction, unit decomposition, atomic comparison, and uncertainty metrics
- [salt_benchmark/generation](salt_benchmark/generation): chat-template rendering, vLLM generation, logprob-to-unit confidence alignment, and JSON cache helpers
- [salt_benchmark/tutorial_utils.py](salt_benchmark/tutorial_utils.py): display and quiet-execution helpers used only by the tutorial notebook

[tutorial.ipynb](tutorial.ipynb) is a runnable tour over that backend. Its Advanced section includes "Adding Your Own Tasks", which demonstrates the same extension path used by backend tasks: define a `Problem` subclass, choose `delimiter` or `regex_exp` for atom decomposition, provide a deterministic solver, and create rows with `task`, `input`, and `reference` columns. The notebook's `Decimal-Hexa-Translation` example is intentionally self-contained for teaching; to make a new task part of the package, promote the class and solver into [salt_benchmark/tasks](salt_benchmark/tasks), add a dataset creator in [salt_benchmark/datasets.py](salt_benchmark/datasets.py), register it in [salt_benchmark/tasks/__init__.py](salt_benchmark/tasks/__init__.py), and add focused tests.

## Installation

Install the project as `salt-benchmark`. In Python code, import it as `salt_benchmark` because Python module names cannot contain hyphens.

Install all dependencies, including notebook and vLLM support:

```bash
pip install -e .[all]
```

Install the repository without optional generation dependencies:

```bash
pip install -e .[dev]
```

Install only generation support in addition to the base package:

```bash
pip install -e .[generation]
```

Equivalent non-editable install:

```bash
pip install .[all]
```

If you are running scripts from the repository root and only need the dependencies, use:

```bash
pip install -r requirements.txt
```

## Reproduction Data

The reproduction datasets are saved with HuggingFace `Dataset.save_to_disk`:

- `DNA_dataset` (DNA Translation)
- `Code_dataset` (Code)
- `Logic_dataset` (First-Order Logic)
- `Matrix_multiplication_dataset` (Matrix Multiplication)
- `Kronecker_product_dataset` (Kronecker Product)
- `NeedlesInHaystack_dataset` (Words Collection)

Load any dataset with `datasets.load_from_disk`:

```python
from datasets import load_from_disk

dataset = load_from_disk("data/reproduction/DNA_dataset")
print(dataset.column_names)
print(dataset[0])
```

## Tutorial

[tutorial.ipynb](tutorial.ipynb) is the notebook entry point. It introduces the benchmark motivation, evaluation units, precision, redundant units, ECE, AUROC, PRR, dataset construction, reproduction data loading, prompt construction, vLLM generation, atomic evaluation, logprob-based uncertainty evaluation, and custom task creation.

The vLLM section is controlled by `RUN_REAL_VLLM`. Set it to `False` for CPU-only notebook execution, or use `True` in an environment with a GPU and model access.

## Dataset Creation

All dataset creators return HuggingFace `Dataset` objects with `task`, `input`, and `reference` columns. The Code dataset also includes `variant`, and the Words Collection dataset also includes `subject`.

```python
from salt_benchmark.datasets import (
    create_dna_dataset,
    create_logic_dataset,
    create_matrix_multiplication_dataset,
)

dna = create_dna_dataset(num_samples=5, min_codons=10, max_codons=12, seed=0)
logic = create_logic_dataset(num_samples=2, num_variables_range=(4, 6), num_initial_assignments=2, seed=0)
matrix = create_matrix_multiplication_dataset(
    num_samples=2,
    output_lengths=[4],
    min_inner_dim=2,
    max_inner_dim=2,
    seed=0,
)

print(dna[0])
print(logic[0])
print(matrix[0])
```

Save generated datasets with `save=True` and `save_dir`:

```python
from salt_benchmark.datasets import create_matrix_multiplication_dataset

create_matrix_multiplication_dataset(
    num_samples=20,
    output_lengths=[6, 8, 10, 12],
    seed=42,
    save=True,
    save_dir="data/my_matrix_dataset",
)
```

## Prompt Construction

Task classes own the prompt format and few-shot examples. Each prompt asks the model to place the final answer inside asterisk fences so it can be extracted deterministically.

```python
from salt_benchmark.datasets import create_dna_dataset
from salt_benchmark.tasks import DNAProblem, problem_from_task

dataset = create_dna_dataset(num_samples=1, max_codons=4, seed=0)
problem = DNAProblem()
prompt = problem.construct_prompt(dataset[0]["input"], few_shot_num=2)

print(prompt)

same_problem = problem_from_task(dataset[0])
assert isinstance(same_problem, DNAProblem)
```

## Generation With vLLM

`VLLMGenerator` renders SALT prompts through the model tokenizer chat template and calls vLLM.

```python
from salt_benchmark.datasets import create_dna_dataset
from salt_benchmark.generation import GenerationConfig, VLLMGenerator
from salt_benchmark.tasks import DNAProblem

dataset = create_dna_dataset(num_samples=1, max_codons=4, seed=0)
problem = DNAProblem()
prompt = problem.construct_prompt(dataset[0]["input"], few_shot_num=2)

generator = VLLMGenerator("Qwen/Qwen3-4B-Instruct-2507")
results = generator.generate(
    [prompt],
    inputs=[dataset[0]["input"]],
    config=GenerationConfig(max_tokens=512, temperature=0.0, top_p=1.0, logprobs=20),
)

print(results[0].text)
print(results[0].logprobs)
```

SALT uses 20 top logprobs in the tutorial to match the OpenAI and Gemini API limit.

The regular unit tests use fake backends and do not load a model.

## Generation Cache

`save_generation_cache` and `load_generation_cache` write and read versioned JSON cache files. Cache payloads should contain JSON-serializable values such as model names, prompts, inputs, response text, and simplified logprob records.

```python
from salt_benchmark.generation import load_generation_cache, save_generation_cache

payload = {"model": "demo", "responses": [{"text": "********\nTCT CGA\n********"}]}
save_generation_cache("cache/demo_generation.json", payload)
loaded = load_generation_cache("cache/demo_generation.json")

assert loaded == payload
```

## Atomic Evaluation

Atomic evaluation extracts the fenced answer, decomposes the response and reference into units, and compares units position by position. Comparison statuses include `correct`, `incorrect`, `missing`, `redundant`, and `selective`.

```python
from salt_benchmark.evaluation import evaluate_atomic_response
from salt_benchmark.tasks import DNAProblem

problem = DNAProblem()
response = "thinking...\n********\nTCT CGA AAA\n********"
reference = "TCT CGA"

result = evaluate_atomic_response(response, reference, problem, unit="atom")

print(result.response_units)
print(result.reference_units)
print([comparison.status for comparison in result.comparisons])
print(result.metrics["num_correct"])
print(result.metrics["num_redundant"])
```

Supported unit modes are:

- `atom`: task-specific atomic units, using each problem's delimiter or regex metadata
- `line`: non-empty output lines
- `generation`: the whole extracted answer as one unit

For atom decomposition, `delimiter` is used when it is set. Set `delimiter=None` to use `regex_exp` instead. Simple translation and flat-sequence tasks such as DNA Translation, Matrix Multiplication, Reverse Words, Next Permutation, and Rotate Array use delimiters. More structured Code variants use regex atoms: Edit Distance extracts the scalar numeric answer, Rotate Matrix extracts whole matrix rows, and Insert Interval extracts whole intervals. Those regex boundaries prevent punctuation inside structured values from being mistaken for atom separators.

## Uncertainty Evaluation

SALT includes pure-Python ports of the benchmark uncertainty metrics and logprob confidence functions. The metric helpers include `ECE_calc`, `AUROC`, `auroc_score`, and PRR helpers such as `prr_from_correctness` and `prr_from_scores`. The notebook-facing helper `evaluate_uncertainty_scores` reports Precision, ECE, AUROC, and PRR for confidence scores aligned with generated units.

Confidence functions have explicit direction metadata in `CONFIDENCE_SPECS`. Probability-style scores such as `mean_probability`, `median_probability`, and `max_probability` are already confidence-style. Scores such as `perplexity`, `log_perplexity`, `mean_entropy`, and `max_entropy` are uncertainty-style, so lower raw values indicate higher confidence. Pass `score_name` or `higher_is_confident=False` when evaluating ranking metrics.

```python
from salt_benchmark.evaluation import evaluate_atomic_response, evaluate_uncertainty_scores, zscore_sigmoid_calibration
from salt_benchmark.generation import CONFIDENCE_SPECS
from salt_benchmark.tasks import MatrixMultiplicationProblem

problem = MatrixMultiplicationProblem()
response = "********\n5 4\n6 7 8\n********"
reference = "5 4\n5 7"

evaluation = evaluate_atomic_response(response, reference, problem, unit="atom")
confidence_scores = [1.0, 1.0, 0.8, 0.8, 0.8]
uncertainty = evaluate_uncertainty_scores(evaluation, confidence_scores)

print(uncertainty.precision)
print(uncertainty.ece)
print(uncertainty.auroc)
print(uncertainty.prr)

perplexities = [1.0, 1.0, 1.8, 1.8, 1.8]
calibrated = zscore_sigmoid_calibration(perplexities, perplexities, higher_is_confident=False)
uncertainty_from_ppl = evaluate_uncertainty_scores(
  evaluation,
  perplexities,
  score_name="perplexity",
  calibration_confidences=calibrated,
)
```

The logprob confidence functions include mean probability, median probability, max probability, log-probability sum, max log-probability, log-perplexity, perplexity, mean entropy, and max entropy.

## Tests

```bash
python -m pytest
```

The default tests do not require a GPU or model download.