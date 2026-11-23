"""Provider-agnostic interleaved async benchmark runner for malay-tool-bench."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from malay_tool_bench.models import (
    EvalCase,
    NormalizedResponse,
    RawToolCall,
    RunRecord,
    TokenUsage,
)
from malay_tool_bench.pricing import compute_cost_myr
from malay_tool_bench.scorer import score_case
from malay_tool_bench.toolsets import TOOLSETS, build_system_prompt

RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


class ModelTarget(BaseModel):
    """Target model endpoint configuration."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    label: str
    reasoning_effort: str | None = None
    base_url: str = "https://api.ilmu.ai/v1"
    api_key_env: str = "ILMU_API_KEY"


class RunMeta(BaseModel):
    """Top-level run metadata written to results/<run-id>/meta.json."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset: str
    case_count: int
    trials: int
    models: list[ModelTarget]
    started_at: str
    finished_at: str
    total_cost_myr: float
    budget_myr: float
    budget_stopped: bool
    records_written: int
    load_avg: list[float]
    cpu_count: int


def parse_model_spec(
    spec: str,
    *,
    base_url: str = "https://api.ilmu.ai/v1",
    api_key_env: str = "ILMU_API_KEY",
) -> ModelTarget:
    """Parse 'model-id' or 'model-id@effort' into a ModelTarget."""
    clean = spec.strip()
    if "@" in clean:
        model_id, effort = clean.split("@", 1)
        model_id = model_id.strip()
        effort = effort.strip()
        return ModelTarget(
            model_id=model_id,
            label=f"{model_id}@{effort}",
            reasoning_effort=effort or None,
            base_url=base_url.rstrip("/"),
            api_key_env=api_key_env,
        )
    return ModelTarget(
        model_id=clean,
        label=clean,
        reasoning_effort=None,
        base_url=base_url.rstrip("/"),
        api_key_env=api_key_env,
    )


def load_dataset(path: Path) -> list[EvalCase]:
    """Load and validate EvalCase rows from a JSONL file."""
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                cases.append(EvalCase.model_validate(data))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid EvalCase: {exc}") from exc
    return cases


def build_interleaved_schedule(
    cases: list[EvalCase],
    targets: list[ModelTarget],
    trials: int,
) -> list[tuple[int, EvalCase, ModelTarget]]:
    """Build interleaved (trial, case, target) tasks rotating start offset per case."""
    schedule: list[tuple[int, EvalCase, ModelTarget]] = []
    n_targets = len(targets)
    if n_targets == 0:
        return schedule
    for trial in range(1, trials + 1):
        for case_idx, case in enumerate(cases):
            offset = (case_idx + trial - 1) % n_targets
            for step in range(n_targets):
                target = targets[(offset + step) % n_targets]
                schedule.append((trial, case, target))
    return schedule


def _parse_usage(payload_usage: Any) -> TokenUsage:
    if not isinstance(payload_usage, dict):
        return TokenUsage()
    prompt_tokens = int(payload_usage.get("prompt_tokens") or 0)
    completion_tokens = int(payload_usage.get("completion_tokens") or 0)
    prompt_details = payload_usage.get("prompt_tokens_details") or {}
    completion_details = payload_usage.get("completion_tokens_details") or {}
    cached_tokens = int(
        prompt_details.get("cached_tokens") or payload_usage.get("cached_tokens") or 0
    )
    reasoning_tokens = int(
        completion_details.get("reasoning_tokens") or payload_usage.get("reasoning_tokens") or 0
    )
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _parse_tool_calls(raw_calls: Any) -> list[RawToolCall]:
    if not isinstance(raw_calls, list):
        return []
    parsed: list[RawToolCall] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        name = str(fn.get("name") or "")
        args_raw = fn.get("arguments")
        if args_raw is None:
            args_raw_str = ""
        elif isinstance(args_raw, str):
            args_raw_str = args_raw
        else:
            args_raw_str = json.dumps(args_raw)

        args_dict: dict[str, Any] | None = None
        json_err: str | None = None
        try:
            loaded = json.loads(args_raw_str) if args_raw_str.strip() else {}
            if isinstance(loaded, dict):
                args_dict = loaded
            else:
                json_err = f"expected JSON object, got {type(loaded).__name__}"
        except json.JSONDecodeError as exc:
            json_err = str(exc)

        parsed.append(
            RawToolCall(
                id=str(item.get("id") or ""),
                name=name,
                arguments_raw=args_raw_str,
                arguments=args_dict,
                json_error=json_err,
            )
        )
    return parsed


async def call_model_once(
    client: httpx.AsyncClient,
    target: ModelTarget,
    case: EvalCase,
    api_key: str,
    *,
    timeout_s: float = 45.0,
    max_retries: int = 2,
    retry_base_delay_s: float = 0.75,
) -> NormalizedResponse:
    """Invoke one OpenAI-compatible chat completion with defensive parsing and retries."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(case.context_note)},
        *[{"role": m.role, "content": m.content} for m in case.messages],
    ]
    payload: dict[str, Any] = {
        "model": target.model_id,
        "messages": messages,
        "tools": TOOLSETS[case.domain],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 4096 if target.reasoning_effort and target.reasoning_effort != "none" else 1024,
    }
    if target.reasoning_effort is not None:
        payload["reasoning_effort"] = target.reasoning_effort

    url = f"{target.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = ""
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout_s)
            latency_ms = (time.perf_counter() - started) * 1000.0
        except httpx.TimeoutException as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            last_error = f"timeout after {latency_ms:.0f}ms: {exc}"
            if attempt < max_retries:
                await asyncio.sleep(retry_base_delay_s * (2**attempt))
                continue
            return NormalizedResponse(
                status="timeout",
                error_detail=last_error,
                latency_ms=latency_ms,
            )
        except httpx.HTTPError as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            last_error = f"http error: {exc}"
            if attempt < max_retries:
                await asyncio.sleep(retry_base_delay_s * (2**attempt))
                continue
            return NormalizedResponse(
                status="api_error",
                error_detail=last_error,
                latency_ms=latency_ms,
            )

        if resp.status_code in RETRYABLE_STATUS_CODES:
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if attempt < max_retries:
                await asyncio.sleep(retry_base_delay_s * (2**attempt))
                continue
            return NormalizedResponse(
                status="api_error",
                error_detail=last_error,
                latency_ms=latency_ms,
            )

        if resp.status_code >= 400:
            return NormalizedResponse(
                status="api_error",
                error_detail=f"HTTP {resp.status_code}: {resp.text[:400]}",
                latency_ms=latency_ms,
            )

        try:
            data = resp.json()
        except Exception as exc:
            return NormalizedResponse(
                status="api_error",
                error_detail=f"invalid JSON body: {exc}",
                latency_ms=latency_ms,
            )

        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return NormalizedResponse(
                status="api_error",
                error_detail="missing choices[0] in response",
                latency_ms=latency_ms,
            )

        choice0 = choices[0]
        message = choice0.get("message") or {}
        content = message.get("content")
        content_str = content if isinstance(content, str) else ""
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        usage = _parse_usage(data.get("usage"))
        cost_myr = compute_cost_myr(target.model_id, usage)

        return NormalizedResponse(
            status="ok",
            content=content_str,
            tool_calls=tool_calls,
            finish_reason=choice0.get("finish_reason"),
            latency_ms=latency_ms,
            usage=usage,
            cost_myr=cost_myr,
        )

    return NormalizedResponse(status="api_error", error_detail=last_error or "unknown error")


async def run_benchmark(
    *,
    dataset_path: Path,
    cases: list[EvalCase],
    targets: list[ModelTarget],
    out_dir: Path,
    trials: int = 3,
    concurrency: int = 4,
    budget_myr: float = 5.0,
    timeout_s: float = 45.0,
    retry_base_delay_s: float = 0.75,
    api_keys: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[RunRecord], RunMeta]:
    """Execute interleaved benchmark, stream raw.jsonl, and return (records, meta)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name
    started_at = datetime.now(UTC).isoformat()

    resolved_keys: dict[str, str] = {}
    for target in targets:
        if api_keys and target.api_key_env in api_keys:
            resolved_keys[target.api_key_env] = api_keys[target.api_key_env]
        else:
            env_val = os.environ.get(target.api_key_env, "").strip()
            if not env_val:
                raise RuntimeError(
                    f"Missing API key in env var {target.api_key_env} for model {target.label}"
                )
            resolved_keys[target.api_key_env] = env_val

    schedule = build_interleaved_schedule(cases, targets, trials)
    sem = asyncio.Semaphore(max(1, concurrency))
    per_model_limit = max(1, (concurrency + max(1, len(targets)) - 1) // max(1, len(targets)))
    model_sems = {t.label: asyncio.Semaphore(per_model_limit) for t in targets}
    state_lock = asyncio.Lock()

    spent_myr = 0.0
    reserved_myr = 0.0
    budget_stopped = False
    records: list[RunRecord] = []
    raw_path = out_dir / "raw.jsonl"

    async with httpx.AsyncClient(transport=transport) as client:
        with raw_path.open("w", encoding="utf-8") as raw_fh:

            async def _worker(trial: int, case: EvalCase, target: ModelTarget) -> None:
                nonlocal spent_myr, reserved_myr, budget_stopped
                # Conservative per-call reservation (~0.02 MYR) avoids concurrent cap overshoot.
                reserve = min(0.02, budget_myr)
                async with model_sems[target.label], sem:
                    async with state_lock:
                        if spent_myr + reserved_myr >= budget_myr and (spent_myr > 0 or reserved_myr > 0):
                            budget_stopped = True
                            return
                        reserved_myr += reserve
                    try:
                        resp = await call_model_once(
                            client,
                            target,
                            case,
                            resolved_keys[target.api_key_env],
                            timeout_s=timeout_s,
                            retry_base_delay_s=retry_base_delay_s,
                        )
                    except BaseException:
                        async with state_lock:
                            reserved_myr = max(0.0, reserved_myr - reserve)
                        raise
                    score = score_case(case, resp)
                    rec = RunRecord(
                        run_id=run_id,
                        trial=trial,
                        model_id=target.model_id,
                        model_label=target.label,
                        reasoning_effort=target.reasoning_effort,
                        case_id=case.id,
                        response=resp,
                        score=score,
                    )
                    async with state_lock:
                        reserved_myr = max(0.0, reserved_myr - reserve)
                        spent_myr += resp.cost_myr
                        if spent_myr >= budget_myr:
                            budget_stopped = True
                        records.append(rec)
                        raw_fh.write(rec.model_dump_json() + "\n")
                        raw_fh.flush()

            await asyncio.gather(*(_worker(t, c, m) for t, c, m in schedule))

    finished_at = datetime.now(UTC).isoformat()
    try:
        load_avg = [round(v, 2) for v in os.getloadavg()]
    except OSError:
        load_avg = []

    meta = RunMeta(
        run_id=run_id,
        dataset=str(dataset_path),
        case_count=len(cases),
        trials=trials,
        models=targets,
        started_at=started_at,
        finished_at=finished_at,
        total_cost_myr=round(spent_myr, 6),
        budget_myr=budget_myr,
        budget_stopped=budget_stopped,
        records_written=len(records),
        load_avg=load_avg,
        cpu_count=os.cpu_count() or 1,
    )
    (out_dir / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return records, meta
