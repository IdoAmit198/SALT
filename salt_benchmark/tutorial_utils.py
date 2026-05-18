from __future__ import annotations

import contextlib
import html
import io
import logging
import os
import tempfile
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from IPython.display import display

from .generation import GenerationConfig, VLLMGenerator


def ensure_repository_root(path: str | Path | None = None) -> Path:
    """Return the SALT repository root or fail with an actionable message.

    Args:
        path: Candidate working directory. Defaults to the current working directory.

    Returns:
        The resolved repository root path.
    """
    repo_root = Path.cwd() if path is None else Path(path)
    if not (repo_root / "salt_benchmark").exists():
        raise RuntimeError("Run this notebook from the SALT repository root.")
    return repo_root


def configure_quiet_generation_logs() -> None:
    """Reduce routine logging from optional model-generation libraries."""
    warnings.filterwarnings("ignore", message=r"IProgress not found.*", category=Warning)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    for logger_name in ("vllm", "transformers"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def clip(text: object, max_chars: int = 1200) -> str:
    """Return a string shortened to ``max_chars`` characters for display."""
    value = str(text)
    return value if len(value) <= max_chars else f"{value[:max_chars].rstrip()}..."


def md_safe(text: object) -> str:
    """Escape values that would otherwise break a Markdown table cell."""
    if text is None:
        return ""
    return str(text).replace("|", "&#124;").replace("\n", "<br>")


def show_text(title: str, text: object, max_chars: int = 1600) -> None:
    """Display a titled fenced text block in a notebook."""
    clipped = clip(text, max_chars)
    _display_html(
        f"<h3>{_html_safe(title)}</h3><pre>{_html_safe(clipped)}</pre>",
        f"{title}\n\n{clipped}",
    )


def show_dataset_overview(datasets_by_name: Mapping[str, Any]) -> None:
    """Display row counts and column names for a collection of datasets."""
    table_rows: list[list[object]] = []
    plain_rows = ["| Dataset | Rows | Columns |", "|---|---:|---|"]
    for name, dataset in datasets_by_name.items():
        columns = ", ".join(dataset.column_names)
        table_rows.append([name, len(dataset), columns])
        plain_rows.append(f"| {name} | {len(dataset)} | `{columns}` |")
    _display_html(_html_table(["Dataset", "Rows", "Columns"], table_rows), "\n".join(plain_rows))


def show_row(
    title: str,
    row: Mapping[str, object],
    fields: tuple[str, ...] = ("input", "reference"),
    max_chars: int = 1200,
) -> None:
    """Display selected fields from one dataset row as titled text blocks."""
    html_blocks = [f"<h3>{_html_safe(title)}</h3>"]
    plain_blocks = [title]
    for field in fields:
        if field in row:
            clipped = clip(row[field], max_chars)
            html_blocks.append(f"<h4>{_html_safe(field.title())}</h4><pre>{_html_safe(clipped)}</pre>")
            plain_blocks.append(f"{field.title()}\n\n{clipped}")
    _display_html("".join(html_blocks), "\n\n".join(plain_blocks))


def unit_precision(evaluation) -> float:
    """Compute unit precision as correct generated units over generated units."""
    return float(evaluation.metrics["precision"])


def unit_recall(evaluation) -> float:
    """Compute unit recall as correct generated units over reference units."""
    return float(evaluation.metrics["recall"])


def redundant_rate(evaluation) -> float:
    """Compute the fraction of generated units that are redundant."""
    response_units = evaluation.metrics["num_response_units"]
    redundant_units = evaluation.metrics.get("num_redundant", evaluation.metrics["num_extra"])
    return redundant_units / response_units if response_units else 0.0


def show_metrics_by_granularity(evaluations: Mapping[str, Any]) -> None:
    """Display summary metrics for atom, line, and generation evaluations."""
    headers = [
        "Granularity",
        "Response units",
        "Reference units",
        "Correct",
        "Precision",
        "Recall",
        "Redundant units",
        "Redundant rate",
    ]
    table_rows: list[list[object]] = []
    plain_rows = [
        "| Granularity | Response units | Reference units | Correct | Precision | Recall | Redundant units | Redundant rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for unit, evaluation in evaluations.items():
        metrics = evaluation.metrics
        row = [
            unit,
            metrics["num_response_units"],
            metrics["num_reference_units"],
            metrics["num_correct"],
            f"{unit_precision(evaluation):.3f}",
            f"{unit_recall(evaluation):.3f}",
            metrics.get("num_redundant", metrics["num_extra"]),
            f"{redundant_rate(evaluation):.3f}",
        ]
        table_rows.append(row)
        plain_rows.append(
            f"| `{unit}` | {metrics['num_response_units']} | {metrics['num_reference_units']} | "
            f"{metrics['num_correct']} | {unit_precision(evaluation):.3f} | {unit_recall(evaluation):.3f} | "
            f"{metrics.get('num_redundant', metrics['num_extra'])} | {redundant_rate(evaluation):.3f} |"
        )
    _display_html(_html_table(headers, table_rows), "\n".join(plain_rows))


def show_uncertainty_table(uncertainty_by_granularity: Mapping[str, Any]) -> None:
    """Display precision, calibration, ranking, and rejection metrics by granularity."""
    headers = ["Granularity", "Precision", "ECE", "AUROC", "PRR"]
    table_rows: list[list[object]] = []
    plain_rows = [
        "| Granularity | Precision | ECE | AUROC | PRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for unit, result in uncertainty_by_granularity.items():
        row = [
            unit,
            _format_metric(result.precision),
            _format_metric(result.ece),
            _format_metric(result.auroc),
            _format_metric(result.prr),
        ]
        table_rows.append(row)
        plain_rows.append(
            f"| `{unit}` | {_format_metric(result.precision)} | {_format_metric(result.ece)} | "
            f"{_format_metric(result.auroc)} | {_format_metric(result.prr)} |"
        )
    _display_html(_html_table(headers, table_rows), "\n".join(plain_rows))


def show_comparisons(title: str, evaluation, limit: int = 12) -> None:
    """Display per-unit response/reference comparisons from an evaluation result."""
    headers = ["#", "Response unit", "Reference unit", "Status"]
    table_rows: list[list[object]] = []
    plain_rows = ["| # | Response unit | Reference unit | Status |", "|---:|---|---|---|"]
    for comparison in evaluation.comparisons[:limit]:
        table_rows.append(
            [comparison.index + 1, comparison.response_unit or "", comparison.reference_unit or "", comparison.status]
        )
        plain_rows.append(
            f"| {comparison.index + 1} | {md_safe(comparison.response_unit)} | "
            f"{md_safe(comparison.reference_unit)} | {comparison.status} |"
        )
    if len(evaluation.comparisons) > limit:
        remaining = len(evaluation.comparisons) - limit
        table_rows.append(["...", "...", "...", f"{remaining} more units"])
        plain_rows.append(f"| ... | ... | ... | {remaining} more units |")
    _display_html(f"<h3>{_html_safe(title)}</h3>{_html_table(headers, table_rows)}", "\n".join([title, *plain_rows]))


def _format_metric(value: object) -> str:
    if value is None:
        return "None"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return md_safe(value)
    if numeric != numeric:
        return "nan"
    return f"{numeric:.3f}"


def _html_safe(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _display_html(html_body: str, plain_text: str) -> None:
    display({"text/html": html_body, "text/plain": plain_text}, raw=True)


def _html_table(headers: list[str], rows: list[list[object]]) -> str:
    header_html = "".join(f"<th>{_html_safe(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_html_safe(cell).replace(chr(10), '<br>')}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def quiet_call(action: Callable[[], Any], label: str) -> Any:
    """Run an action while suppressing noisy stdout, stderr, and warnings.

    Captured logs are printed only if the action raises, so failures remain
    debuggable without cluttering successful notebook execution.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    fd_capture = tempfile.TemporaryFile(mode="w+b")
    saved_fds: list[tuple[int, int]] = []
    caught_exception: Exception | None = None
    traceback = None
    result = None
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            warnings.simplefilter("ignore")
            saved_fds = _redirect_standard_fds(fd_capture.fileno())
            try:
                result = action()
            except Exception as exc:
                caught_exception = exc
                traceback = exc.__traceback__
            finally:
                _restore_standard_fds(saved_fds)
        if caught_exception is None:
            return result
        captured = _captured_output(stdout, stderr, fd_capture)
        if captured:
            print(f"{label} produced logs before failing:\n{captured}")
        raise caught_exception.with_traceback(traceback)
    finally:
        fd_capture.close()


def _redirect_standard_fds(target_fd: int) -> list[tuple[int, int]]:
    saved_fds: list[tuple[int, int]] = []
    for fd in (1, 2):
        saved_fd = None
        try:
            saved_fd = os.dup(fd)
            os.dup2(target_fd, fd)
        except OSError:
            if saved_fd is not None:
                os.close(saved_fd)
            continue
        saved_fds.append((fd, saved_fd))
    return saved_fds


def _restore_standard_fds(saved_fds: list[tuple[int, int]]) -> None:
    for fd, saved_fd in reversed(saved_fds):
        try:
            os.dup2(saved_fd, fd)
        finally:
            os.close(saved_fd)


def _captured_output(stdout: io.StringIO, stderr: io.StringIO, fd_capture) -> str:
    fd_capture.flush()
    fd_capture.seek(0)
    fd_output = fd_capture.read().decode("utf-8", errors="replace").strip()
    parts = [stdout.getvalue().strip(), stderr.getvalue().strip(), fd_output]
    return "\n".join(part for part in parts if part)


def run_vllm_generation(model_name: str, prompt: str, input_text: str, config: GenerationConfig):
    """Generate one completion for a SALT prompt with vLLM."""

    def action():
        generator = VLLMGenerator(model_name, llm_kwargs={"disable_log_stats": True})
        return generator.generate([prompt], inputs=[input_text], config=config)[0]

    return quiet_call(action, f"vLLM generation with {model_name}")
