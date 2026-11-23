"""CLI entrypoint for malay-tool-bench."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from malay_tool_bench.report import write_report_artifacts
from malay_tool_bench.runner import load_dataset, parse_model_spec, run_benchmark
from malay_tool_bench.toolsets import get_tool_map

app = typer.Typer(add_completion=False, help="Malaysian BM/English/Manglish tool-calling benchmark.")


@app.command()
def validate(
    dataset: Annotated[Path, typer.Argument(help="Path to dataset JSONL.")] = Path("datasets/v1.jsonl"),
) -> None:
    """Validate dataset JSONL against Pydantic schemas and domain toolsets."""
    cases = load_dataset(dataset)
    seen_ids: set[str] = set()
    for case in cases:
        if case.id in seen_ids:
            raise typer.BadParameter(f"Duplicate case id: {case.id}")
        seen_ids.add(case.id)
        tool_map = get_tool_map(case.domain)
        for ft in case.expect.forbidden_tools:
            if ft not in tool_map:
                raise typer.BadParameter(f"{case.id}: unknown forbidden tool {ft!r} in {case.domain}")
        if case.expect.kind == "call_tools":
            for call in case.expect.calls:
                if call.name not in tool_map:
                    raise typer.BadParameter(f"{case.id}: unknown tool {call.name!r} in {case.domain}")
                props = tool_map[call.name]["parameters"]["properties"]
                for arg_key in call.args:
                    if arg_key not in props:
                        raise typer.BadParameter(
                            f"{case.id}: unknown arg {arg_key!r} for {call.name}"
                        )
    typer.echo(f"OK: {len(cases)} cases validated in {dataset}")


@app.command()
def run(
    dataset: Annotated[Path, typer.Option(help="Path to benchmark JSONL dataset.")] = Path("datasets/v1.jsonl"),
    models: Annotated[
        str,
        typer.Option(help="Comma-separated model specs (e.g. ilmu-v3.1@none,ilmu-mini-v3.3)."),
    ] = "ilmu-mini-v3.3,ilmu-glm-5.3-flash@low,ilmu-v3.1@none,ilmu-glm-5.3@low",
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible API base URL.")] = "https://api.ilmu.ai/v1",
    api_key_env: Annotated[str, typer.Option(help="Env var holding the API key.")] = "ILMU_API_KEY",
    trials: Annotated[int, typer.Option(min=1, help="Number of interleaved trials per model.")] = 3,
    concurrency: Annotated[int, typer.Option(min=1, help="Concurrent in-flight API requests.")] = 4,
    budget_myr: Annotated[float, typer.Option(min=0.01, help="Hard spend cap in MYR.")] = 5.0,
    timeout_s: Annotated[float, typer.Option(min=1.0, help="Per-request timeout in seconds.")] = 45.0,
    limit: Annotated[int | None, typer.Option(min=1, help="Optional max cases (for smoke tests).")] = None,
    categories: Annotated[str | None, typer.Option(help="Optional comma-separated category filter.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Output directory (defaults to results/<ts>).")] = None,
) -> None:
    """Run interleaved live evals and write raw.jsonl, summary.json, and REPORT.md."""
    cases = load_dataset(dataset)
    if categories:
        allowed = {c.strip() for c in categories.split(",") if c.strip()}
        cases = [c for c in cases if c.category in allowed]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise typer.BadParameter("No cases matched the selected filters.")

    targets = [
        parse_model_spec(spec, base_url=base_url, api_key_env=api_key_env)
        for spec in models.split(",")
        if spec.strip()
    ]
    if not targets:
        raise typer.BadParameter("At least one model must be provided.")

    if out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = Path("results") / stamp

    _, meta = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            cases=cases,
            targets=targets,
            out_dir=out_dir,
            trials=trials,
            concurrency=concurrency,
            budget_myr=budget_myr,
            timeout_s=timeout_s,
        )
    )
    summaries, report_path, _ = write_report_artifacts(out_dir)
    typer.echo(
        f"Done: {meta.records_written} records, RM {meta.total_cost_myr:.4f} spent "
        f"(budget_stopped={meta.budget_stopped}). Report: {report_path}"
    )
    for s in summaries:
        typer.echo(
            f"  {s.model_label:24s} median={s.pass_rate_median:5.1f}% "
            f"({s.pass_rate_min:.1f}..{s.pass_rate_max:.1f}%) "
            f"safety_viol={s.safety_violations} p50={s.latency_p50_ms:.0f}ms RM={s.total_cost_myr:.4f}"
        )


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Directory containing raw.jsonl")],
) -> None:
    """Regenerate REPORT.md and summary.json offline from raw.jsonl."""
    summaries, report_path, summary_path = write_report_artifacts(run_dir)
    typer.echo(f"Wrote {report_path} and {summary_path} ({len(summaries)} models)")
