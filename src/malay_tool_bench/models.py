"""Typed domain models for malay-tool-bench.

All benchmark inputs, tool call expectations, normalized model outputs, and
scoring verdicts are modeled as strict Pydantic v2 types so invalid cases fail
at load time instead of silently distorting benchmark numbers.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LanguageTag = Literal["en_my", "bm", "manglish"]
DomainName = Literal["banking_ryt", "govtech_my", "sme_kedai", "daily_rojak"]
CategoryName = Literal[
    "triplet_en_my",
    "triplet_bm",
    "triplet_manglish",
    "complex_multi_tool",
    "confirmation_guardrail",
    "abstention_ambiguity",
]
FailureKind = Literal[
    "api_error",
    "timeout",
    "invalid_json_args",
    "unexpected_tool_call",
    "missed_tool_call",
    "wrong_tool",
    "unsafe_direct_execution",
    "wrong_call_count",
    "missing_required_arg",
    "wrong_arg_value",
    "unexpected_extra_arg",
    "empty_abstention",
]


class ArgMatcher(BaseModel):
    """Deterministic matcher for one tool-call argument."""

    model_config = ConfigDict(extra="forbid")

    op: Literal[
        "exact",
        "case_insensitive",
        "number_close",
        "one_of",
        "contains_all",
        "regex",
        "absent",
    ]
    value: Any = None
    tolerance: float = 0.01

    @model_validator(mode="after")
    def _validate_value(self) -> ArgMatcher:
        if self.op == "absent":
            return self
        if self.value is None:
            raise ValueError(f"ArgMatcher op={self.op!r} requires a non-null value")
        if self.op == "number_close" and not isinstance(self.value, int | float):
            raise ValueError("number_close requires numeric value")
        if self.op in ("one_of", "contains_all") and not isinstance(self.value, list):
            raise ValueError(f"{self.op} requires a list value")
        if self.op == "regex" and not isinstance(self.value, str):
            raise ValueError("regex requires a string pattern")
        return self


class ExpectedToolCall(BaseModel):
    """Expected tool invocation and argument constraints."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, ArgMatcher] = Field(default_factory=dict)
    allow_extra_args: bool = True


class CallToolsExpectation(BaseModel):
    """Model must invoke specific tool(s) and never invoke forbidden ones."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["call_tools"] = "call_tools"
    calls: list[ExpectedToolCall] = Field(min_length=1)
    ordered: bool = False
    forbidden_tools: list[str] = Field(default_factory=list)
    allow_extra_calls: bool = False


class AbstainExpectation(BaseModel):
    """Model must emit zero tool calls and answer/clarify/refuse in text."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["abstain"] = "abstain"
    forbidden_tools: list[str] = Field(default_factory=list)
    require_non_empty_text: bool = True
    reason: str = ""


Expectation = Annotated[
    CallToolsExpectation | AbstainExpectation,
    Field(discriminator="kind"),
]


class ChatMessage(BaseModel):
    """One message in an eval case conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class EvalCase(BaseModel):
    """One self-contained benchmark case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3)
    domain: DomainName
    category: CategoryName
    language: LanguageTag
    triplet_group: str | None = None
    context_note: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    expect: Expectation
    notes: str = ""

    @model_validator(mode="after")
    def _validate_triplet(self) -> EvalCase:
        is_triplet_cat = self.category.startswith("triplet_")
        if is_triplet_cat and not self.triplet_group:
            raise ValueError(f"Case {self.id} has triplet category but missing triplet_group")
        if is_triplet_cat:
            expected_lang = self.category.removeprefix("triplet_")
            if self.language != expected_lang:
                raise ValueError(
                    f"Case {self.id} category {self.category} disagrees with language {self.language}"
                )
        return self


class RawToolCall(BaseModel):
    """Normalized tool call extracted from an OpenAI-compatible API response."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    name: str
    arguments_raw: str
    arguments: dict[str, Any] | None = None
    json_error: str | None = None


class TokenUsage(BaseModel):
    """Token counts reported by the model API."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


class NormalizedResponse(BaseModel):
    """Boundary-normalized response from any OpenAI-compatible provider."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "api_error", "timeout"] = "ok"
    error_detail: str | None = None
    content: str = ""
    tool_calls: list[RawToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    latency_ms: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_myr: float = 0.0


class CaseScore(BaseModel):
    """Pure deterministic score for one (case, trial, model) execution."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    domain: DomainName
    category: CategoryName
    language: LanguageTag
    triplet_group: str | None = None
    passed: bool
    failure_kind: FailureKind | None = None
    failure_detail: str = ""
    tool_name_correct: bool
    args_correct: bool
    schema_valid: bool
    safety_violation: bool
    emitted_tools: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    """One serialized JSONL row in results/<run-id>/raw.jsonl."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    trial: int
    model_id: str
    model_label: str
    reasoning_effort: str | None = None
    case_id: str
    response: NormalizedResponse
    score: CaseScore
