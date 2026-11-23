"""Dataset integrity and gold-response control tests for datasets/v1.jsonl."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from malay_tool_bench.models import (
    AbstainExpectation,
    ArgMatcher,
    CallToolsExpectation,
    EvalCase,
    NormalizedResponse,
    RawToolCall,
)
from malay_tool_bench.runner import load_dataset
from malay_tool_bench.scorer import score_case
from malay_tool_bench.toolsets import get_tool_map

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "v1.jsonl"


def _synthesize_arg(prop_spec: dict[str, Any], matcher: ArgMatcher | None) -> Any:
    if matcher is not None:
        if matcher.op == "exact" or matcher.op == "case_insensitive" or matcher.op == "number_close":
            return matcher.value
        if matcher.op == "one_of":
            return matcher.value[0]
        if matcher.op == "contains_all":
            return " ".join(str(x) for x in matcher.value)
        if matcher.op == "regex":
            pat = str(matcher.value)
            if pat == r"\S+":
                return "ubah suai sekolah"
            if pat.startswith("^2026-11-24T09:30"):
                return "2026-11-24T09:30"
            cleaned = (
                pat.replace("^", "")
                .replace("$", "")
                .replace("[- ]?", "")
                .replace(r"\s?", "")
            )
            return cleaned
    if "enum" in prop_spec:
        return prop_spec["enum"][0]
    t = prop_spec.get("type")
    if t == "string":
        return "sample"
    if t == "integer":
        return 1
    if t == "number":
        return 10.0
    return "sample"


def _synthesize_gold_response(case: EvalCase) -> NormalizedResponse:
    if isinstance(case.expect, AbstainExpectation):
        return NormalizedResponse(content="Sila nyatakan maklumat lanjut sebelum saya meneruskan.")
    tool_map = get_tool_map(case.domain)
    calls: list[RawToolCall] = []
    for idx, expected in enumerate(case.expect.calls):
        schema = tool_map[expected.name]["parameters"]
        props: dict[str, Any] = schema["properties"]
        required: list[str] = schema["required"]
        args: dict[str, Any] = {}
        for key, matcher in expected.args.items():
            if matcher.op == "absent":
                continue
            args[key] = _synthesize_arg(props[key], matcher)
        for req in required:
            if req not in args:
                args[req] = _synthesize_arg(props[req], None)
        calls.append(
            RawToolCall(
                id=f"gold_{idx}",
                name=expected.name,
                arguments_raw=json.dumps(args),
                arguments=args,
            )
        )
    return NormalizedResponse(tool_calls=calls)


def test_dataset_composition_and_uniqueness() -> None:
    cases = load_dataset(DATASET_PATH)
    assert len(cases) == 90

    ids = [c.id for c in cases]
    assert len(set(ids)) == 90

    prompts = [(c.domain, c.context_note or "", c.messages[-1].content.strip().lower()) for c in cases]
    assert len(set(prompts)) == 90

    cat_counts = Counter(c.category for c in cases)
    assert cat_counts == {
        "triplet_en_my": 15,
        "triplet_bm": 15,
        "triplet_manglish": 15,
        "complex_multi_tool": 15,
        "confirmation_guardrail": 15,
        "abstention_ambiguity": 15,
    }


def test_triplets_are_perfectly_matched() -> None:
    cases = load_dataset(DATASET_PATH)
    groups: dict[str, list[EvalCase]] = defaultdict(list)
    for c in cases:
        if c.triplet_group:
            groups[c.triplet_group].append(c)

    assert len(groups) == 15
    for gid, members in groups.items():
        assert len(members) == 3, f"{gid} has {len(members)} members"
        assert {m.language for m in members} == {"en_my", "bm", "manglish"}
        assert len({m.domain for m in members}) == 1
        assert len({m.context_note for m in members}) == 1
        dumped = {m.expect.model_dump_json() for m in members}
        assert len(dumped) == 1, f"{gid} expectations diverge across languages"


def test_every_case_references_valid_tools_and_passes_gold_response() -> None:
    cases = load_dataset(DATASET_PATH)
    for case in cases:
        tool_map = get_tool_map(case.domain)
        if case.domain == "banking_ryt":
            assert "execute_duitnow_transfer" in case.expect.forbidden_tools
        if case.domain == "sme_kedai":
            assert "process_refund" in case.expect.forbidden_tools

        for ft in case.expect.forbidden_tools:
            assert ft in tool_map, f"{case.id}: unknown forbidden tool {ft}"

        if isinstance(case.expect, CallToolsExpectation):
            for call in case.expect.calls:
                assert call.name in tool_map, f"{case.id}: unknown tool {call.name}"
                props = tool_map[call.name]["parameters"]["properties"]
                for key in call.args:
                    assert key in props, f"{case.id}: unknown arg {key} on {call.name}"

        gold = _synthesize_gold_response(case)
        verdict = score_case(case, gold)
        assert verdict.passed is True, f"{case.id} failed gold check: {verdict.failure_kind} {verdict.failure_detail}"
