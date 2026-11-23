"""Offline benchmark aggregation and Markdown/JSON report generator."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import BaseModel, ConfigDict

from malay_tool_bench.models import RunRecord
from malay_tool_bench.pricing import has_known_price
from malay_tool_bench.runner import RunMeta


class FailureSample(BaseModel):
    """Concrete failing case excerpt for debugging and write-ups."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    trial: int
    category: str
    language: str
    failure_kind: str
    failure_detail: str
    emitted_tools: list[str]


class ModelSummary(BaseModel):
    """Aggregated benchmark metrics for one model configuration."""

    model_config = ConfigDict(extra="forbid")

    model_label: str
    model_id: str
    reasoning_effort: str | None
    total_records: int
    ok_records: int
    emitted_tool_records: int
    trials_present: list[int]
    complete_trials: list[int]
    per_trial_pass_rates: dict[int, float]
    pass_rate_median: float
    pass_rate_min: float
    pass_rate_max: float
    tool_name_acc: float
    schema_valid_rate: float | None
    safety_violations: int
    api_errors: int
    timeouts: int
    length_truncations: int
    matched_triplet_count: int
    triplet_en_my: float | None
    triplet_bm: float | None
    triplet_manglish: float | None
    bm_minus_en_pp: float | None
    manglish_minus_en_pp: float | None
    by_category: dict[str, float]
    by_domain: dict[str, float]
    failure_counts: dict[str, int]
    latency_p50_ms: float
    latency_p95_ms: float
    known_price: bool
    total_cost_myr: float
    cost_per_100_calls_myr: float | None
    total_prompt_tokens: int
    total_cached_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int
    top_failure_examples: list[FailureSample]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _rate(passed: int, total: int) -> float:
    return round((passed / total) * 100.0, 2) if total > 0 else 0.0


def load_run_records(raw_jsonl: Path) -> list[RunRecord]:
    """Load serialized RunRecord rows from raw.jsonl."""
    records: list[RunRecord] = []
    with raw_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(RunRecord.model_validate_json(stripped))
    return records


def summarize_records(
    records: list[RunRecord],
    expected_cases_per_trial: int | None = None,
) -> list[ModelSummary]:
    """Aggregate RunRecord rows into per-model summaries with spread and parity."""
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.model_label].append(rec)

    summaries: list[ModelSummary] = []
    for label, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: (r.trial, r.case_id))
        first = rows_sorted[0]
        by_trial: dict[int, list[RunRecord]] = defaultdict(list)
        by_cat: dict[str, list[RunRecord]] = defaultdict(list)
        by_dom: dict[str, list[RunRecord]] = defaultdict(list)
        triplets_by_key: dict[tuple[int, str], dict[str, bool]] = defaultdict(dict)
        failures: Counter[str] = Counter()
        latencies: list[float] = []
        examples: list[FailureSample] = []

        for r in rows_sorted:
            by_trial[r.trial].append(r)
            by_cat[r.score.category].append(r)
            by_dom[r.score.domain].append(r)
            if r.score.triplet_group and r.score.category.startswith("triplet_"):
                triplets_by_key[(r.trial, r.score.triplet_group)][r.score.language] = r.score.passed
            if r.response.status == "ok" and r.response.latency_ms > 0:
                latencies.append(r.response.latency_ms)
            if not r.score.passed and r.score.failure_kind:
                failures[r.score.failure_kind] += 1
                if len(examples) < 10:
                    examples.append(
                        FailureSample(
                            case_id=r.case_id,
                            trial=r.trial,
                            category=r.score.category,
                            language=r.score.language,
                            failure_kind=r.score.failure_kind,
                            failure_detail=r.score.failure_detail,
                            emitted_tools=r.score.emitted_tools,
                        )
                    )

        trials_present = sorted(by_trial.keys())
        per_trial_rates = {
            t: _rate(sum(1 for x in by_trial[t] if x.score.passed), len(by_trial[t]))
            for t in trials_present
        }
        if expected_cases_per_trial is not None:
            complete_trials = [t for t in trials_present if len(by_trial[t]) >= expected_cases_per_trial]
        else:
            complete_trials = trials_present
        rate_vals = [per_trial_rates[t] for t in (complete_trials or trials_present)] or [0.0]

        cat_rates = {
            k: _rate(sum(1 for x in v if x.score.passed), len(v))
            for k, v in sorted(by_cat.items())
        }
        dom_rates = {
            k: _rate(sum(1 for x in v if x.score.passed), len(v))
            for k, v in sorted(by_dom.items())
        }

        complete_triplets = [
            langs
            for langs in triplets_by_key.values()
            if {"en_my", "bm", "manglish"}.issubset(langs.keys())
        ]
        matched_groups = {
            gid
            for (_, gid), langs in triplets_by_key.items()
            if {"en_my", "bm", "manglish"}.issubset(langs.keys())
        }
        en_rate: float | None = None
        bm_rate: float | None = None
        mg_rate: float | None = None
        bm_delta: float | None = None
        mg_delta: float | None = None
        if complete_triplets:
            n_trip = len(complete_triplets)
            en_val = _rate(sum(1 for t in complete_triplets if t["en_my"]), n_trip)
            bm_val = _rate(sum(1 for t in complete_triplets if t["bm"]), n_trip)
            mg_val = _rate(sum(1 for t in complete_triplets if t["manglish"]), n_trip)
            en_rate, bm_rate, mg_rate = en_val, bm_val, mg_val
            bm_delta = round(bm_val - en_val, 2)
            mg_delta = round(mg_val - en_val, 2)

        total = len(rows_sorted)
        ok_records = sum(1 for x in rows_sorted if x.response.status == "ok")
        emitted_rows = [x for x in rows_sorted if x.response.status == "ok" and len(x.response.tool_calls) > 0]
        schema_rate = (
            _rate(sum(1 for x in emitted_rows if x.score.schema_valid), len(emitted_rows))
            if emitted_rows
            else None
        )
        known_price = has_known_price(first.model_id)
        total_cost = sum(r.response.cost_myr for r in rows_sorted)
        cost_per_100 = (
            round((total_cost / ok_records) * 100.0, 4) if (known_price and ok_records > 0) else None
        )

        summaries.append(
            ModelSummary(
                model_label=label,
                model_id=first.model_id,
                reasoning_effort=first.reasoning_effort,
                total_records=total,
                ok_records=ok_records,
                emitted_tool_records=len(emitted_rows),
                trials_present=trials_present,
                complete_trials=complete_trials,
                per_trial_pass_rates=per_trial_rates,
                pass_rate_median=round(float(median(rate_vals)), 2),
                pass_rate_min=round(min(rate_vals), 2),
                pass_rate_max=round(max(rate_vals), 2),
                tool_name_acc=_rate(sum(1 for x in rows_sorted if x.score.tool_name_correct), total),
                schema_valid_rate=schema_rate,
                safety_violations=sum(1 for x in rows_sorted if x.score.safety_violation),
                api_errors=sum(1 for x in rows_sorted if x.response.status == "api_error"),
                timeouts=sum(1 for x in rows_sorted if x.response.status == "timeout"),
                length_truncations=sum(1 for x in rows_sorted if x.response.finish_reason == "length"),
                matched_triplet_count=len(matched_groups),
                triplet_en_my=en_rate,
                triplet_bm=bm_rate,
                triplet_manglish=mg_rate,
                bm_minus_en_pp=bm_delta,
                manglish_minus_en_pp=mg_delta,
                by_category=cat_rates,
                by_domain=dom_rates,
                failure_counts=dict(failures.most_common()),
                latency_p50_ms=round(_percentile(latencies, 50.0), 1),
                latency_p95_ms=round(_percentile(latencies, 95.0), 1),
                known_price=known_price,
                total_cost_myr=round(total_cost, 6),
                cost_per_100_calls_myr=cost_per_100,
                total_prompt_tokens=sum(r.response.usage.prompt_tokens for r in rows_sorted),
                total_cached_tokens=sum(r.response.usage.cached_tokens for r in rows_sorted),
                total_completion_tokens=sum(r.response.usage.completion_tokens for r in rows_sorted),
                total_reasoning_tokens=sum(r.response.usage.reasoning_tokens for r in rows_sorted),
                top_failure_examples=examples,
            )
        )

    summaries.sort(key=lambda s: (-s.pass_rate_median, s.safety_violations, s.latency_p50_ms))
    return summaries


def _fmt_opt(val: float | None, suffix: str = "%", signed: bool = False, digits: int = 1) -> str:
    if val is None:
        return "n/a"
    if signed:
        return f"{val:+.{digits}f}{suffix}"
    return f"{val:.{digits}f}{suffix}"


def render_markdown_report(summaries: list[ModelSummary], meta: RunMeta | None = None) -> str:
    """Render a clean GFM benchmark report with trial spread, parity, safety, and cost."""
    lines: list[str] = ["# malay-tool-bench Report", ""]
    if meta is not None:
        lines.extend(
            [
                f"- Run ID: `{meta.run_id}`",
                f"- Dataset: `{meta.dataset}` ({meta.case_count} cases x {meta.trials} interleaved trials)",
                f"- Total records: {meta.records_written} | Total spend: RM {meta.total_cost_myr:.4f} (cap RM {meta.budget_myr:.2f}, budget stopped: {meta.budget_stopped})",
                f"- Host load at end: {meta.load_avg} ({meta.cpu_count} CPUs)",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall results (median and range across complete trials)",
            "",
            "| Model | Pass median (range) | Tool name | Schema valid (emitted) | Safety violations | Err / TO / Len | p50 / p95 ms (ok) | RM / 100 ok calls | Total RM |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s in summaries:
        trial_n = len(s.complete_trials) if s.complete_trials else len(s.trials_present)
        spread = (
            f"**{s.pass_rate_median:.1f}%** ({s.pass_rate_min:.1f} to {s.pass_rate_max:.1f}%)"
            if trial_n > 1
            else f"**{s.pass_rate_median:.1f}%** (1 trial)"
        )
        cost_100_str = _fmt_opt(s.cost_per_100_calls_myr, suffix="", digits=4)
        total_rm_str = f"{s.total_cost_myr:.4f}" if s.known_price else "n/a"
        lines.append(
            f"| `{s.model_label}` | {spread} | {s.tool_name_acc:.1f}% | {_fmt_opt(s.schema_valid_rate)} "
            f"| {s.safety_violations} | {s.api_errors}/{s.timeouts}/{s.length_truncations} "
            f"| {s.latency_p50_ms:.0f} / {s.latency_p95_ms:.0f} | {cost_100_str} | {total_rm_str} |"
        )

    max_triplets = max((s.matched_triplet_count for s in summaries), default=0)
    lines.extend(
        [
            "",
            f"## Matched language parity (same {max_triplets} tasks in EN-MY vs BM vs Manglish)",
            "",
            "| Model | EN-MY | BM | Manglish | BM vs EN | Manglish vs EN |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s in summaries:
        lines.append(
            f"| `{s.model_label}` | {_fmt_opt(s.triplet_en_my)} | {_fmt_opt(s.triplet_bm)} "
            f"| {_fmt_opt(s.triplet_manglish)} | {_fmt_opt(s.bm_minus_en_pp, ' pp', signed=True)} "
            f"| {_fmt_opt(s.manglish_minus_en_pp, ' pp', signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## Breakdown by category",
            "",
            "| Model | Triplet EN | Triplet BM | Triplet Manglish | Multi-tool & Complex | Confirmation Guardrail | Abstention & Ambiguity |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s in summaries:
        c = s.by_category
        lines.append(
            f"| `{s.model_label}` | {_fmt_opt(c.get('triplet_en_my'))} | {_fmt_opt(c.get('triplet_bm'))} "
            f"| {_fmt_opt(c.get('triplet_manglish'))} | {_fmt_opt(c.get('complex_multi_tool'))} "
            f"| {_fmt_opt(c.get('confirmation_guardrail'))} | {_fmt_opt(c.get('abstention_ambiguity'))} |"
        )

    lines.extend(
        [
            "",
            "## Token usage and cache hits",
            "",
            "| Model | Prompt tokens | Cached tokens | Cache hit % | Completion tokens | Reasoning tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s in summaries:
        cache_pct = (
            (s.total_cached_tokens / s.total_prompt_tokens) * 100.0
            if s.total_prompt_tokens > 0
            else 0.0
        )
        lines.append(
            f"| `{s.model_label}` | {s.total_prompt_tokens:,} | {s.total_cached_tokens:,} "
            f"| {cache_pct:.1f}% | {s.total_completion_tokens:,} | {s.total_reasoning_tokens:,} |"
        )

    lines.extend(["", "## Failure taxonomy", ""])
    for s in summaries:
        if not s.failure_counts:
            lines.append(f"- `{s.model_label}`: 0 failures")
        else:
            parts = ", ".join(f"{k}: {v}" for k, v in s.failure_counts.items())
            lines.append(f"- `{s.model_label}`: {parts}")

    lines.extend(["", "## Sample failures (first 5 per model)", ""])
    for s in summaries:
        lines.append(f"### `{s.model_label}`")
        if not s.top_failure_examples:
            lines.append("No failures recorded.")
            lines.append("")
            continue
        for ex in s.top_failure_examples[:5]:
            lines.append(
                f"- `{ex.case_id}` (t{ex.trial}, {ex.category}/{ex.language}): "
                f"`{ex.failure_kind}` ({ex.failure_detail}) emitted={ex.emitted_tools}"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_report_artifacts(run_dir: Path) -> tuple[list[ModelSummary], Path, Path]:
    """Recompute summary.json and REPORT.md from raw.jsonl in run_dir."""
    raw_path = run_dir / "raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")
    records = load_run_records(raw_path)
    meta: RunMeta | None = None
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = RunMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))

    summaries = summarize_records(
        records,
        expected_cases_per_trial=meta.case_count if meta is not None else None,
    )
    summary_payload: dict[str, Any] = {
        "meta": meta.model_dump() if meta else None,
        "models": [s.model_dump() for s in summaries],
    }
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "REPORT.md"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown_report(summaries, meta), encoding="utf-8")
    return summaries, report_path, summary_path
