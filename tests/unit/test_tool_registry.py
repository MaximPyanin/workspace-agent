import json

import pytest
from pydantic import BaseModel, Field

from workplace_agent.llm.types import ToolUseBlock
from workplace_agent.tools.registry import ToolRegistry, ToolSpec


class _AddParams(BaseModel):
    a: int = Field(..., description="left addend")
    b: int = Field(..., description="right addend")


class _AddResult(BaseModel):
    sum: int


class _NoParams(BaseModel):
    pass


class _PingResult(BaseModel):
    pong: bool = True


async def _add(p: _AddParams) -> _AddResult:
    return _AddResult(sum=p.a + p.b)


async def _ping(_: _NoParams) -> _PingResult:
    return _PingResult()


def _add_spec() -> ToolSpec[_AddParams, _AddResult]:
    return ToolSpec(
        name="math_add",
        description="Add two ints",
        params_model=_AddParams,
        result_model=_AddResult,
        handler=_add,
    )


def _ping_spec() -> ToolSpec[_NoParams, _PingResult]:
    return ToolSpec(
        name="ping",
        description="Reply with pong",
        params_model=_NoParams,
        result_model=_PingResult,
        handler=_ping,
    )


def test_registry_rejects_duplicate_name() -> None:
    reg = ToolRegistry()
    reg.register(_add_spec())
    with pytest.raises(ValueError, match="duplicate tool name"):
        reg.register(_add_spec())


def test_list_definitions_emits_strict_anthropic_shape() -> None:
    reg = ToolRegistry()
    reg.register(_add_spec())
    reg.register(_ping_spec())
    defs = reg.list_definitions()
    assert [d.name for d in defs] == ["math_add", "ping"]
    add_def = defs[0]
    assert add_def.strict is False
    schema = add_def.input_schema
    assert schema["type"] == "object"
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]
    assert set(schema["required"]) == {"a", "b"}
    assert schema["additionalProperties"] is False


class _NestedChild(BaseModel):
    label: str
    qty: int = 0


class _NestedParent(BaseModel):
    name: str
    child: _NestedChild | None = None
    tags: list[str] = Field(default_factory=list)


async def _identity(_: _NestedParent) -> _PingResult:
    return _PingResult()


class _ConstrainedParams(BaseModel):
    duration_minutes: int = Field(..., gt=0, lt=10000, description="positive int")
    label: str = Field(..., min_length=1, max_length=80, pattern=r"^[a-z]+$")
    tags: list[str] = Field(default_factory=list, min_length=0, max_length=5)


async def _constrained_handler(_: _ConstrainedParams) -> _PingResult:
    return _PingResult()


def test_strict_schema_drops_unsupported_validation_keywords() -> None:
    spec: ToolSpec[_ConstrainedParams, _PingResult] = ToolSpec(
        name="constrained",
        description="exercises validation keywords",
        params_model=_ConstrainedParams,
        result_model=_PingResult,
        handler=_constrained_handler,
    )
    reg = ToolRegistry()
    reg.register(spec)
    schema = reg.list_definitions()[0].input_schema
    serialized = json.dumps(schema)
    for forbidden in (
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
    ):
        assert forbidden not in serialized, f"strict schema must not include {forbidden}"

    assert schema["properties"]["duration_minutes"]["type"] == "integer"
    assert schema["properties"]["label"]["type"] == "string"
    assert schema["properties"]["tags"]["type"] == "array"


def test_strict_schema_is_anthropic_compliant_for_nested_optional_fields() -> None:
    spec: ToolSpec[_NestedParent, _PingResult] = ToolSpec(
        name="nested",
        description="exercises nested + optional fields",
        params_model=_NestedParent,
        result_model=_PingResult,
        handler=_identity,
    )
    reg = ToolRegistry()
    reg.register(spec)
    schema = reg.list_definitions()[0].input_schema

    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"name", "child", "tags"}
    nested = schema["properties"]["child"]
    inlined = next(branch for branch in nested["anyOf"] if branch.get("type") == "object")
    assert inlined["additionalProperties"] is False
    assert set(inlined["required"]) == {"label", "qty"}


async def test_dispatch_happy_path_returns_result_json() -> None:
    reg = ToolRegistry()
    reg.register(_add_spec())
    block = ToolUseBlock(id="tu_1", name="math_add", input={"a": 2, "b": 3})

    result_block = await reg.dispatch(block)

    assert result_block.tool_use_id == "tu_1"
    assert result_block.is_error is False
    assert json.loads(result_block.content) == {"sum": 5}


async def test_dispatch_validates_arguments_and_returns_structured_error() -> None:
    reg = ToolRegistry()
    reg.register(_add_spec())
    block = ToolUseBlock(id="tu_2", name="math_add", input={"a": "two", "b": 3})

    result_block = await reg.dispatch(block)

    assert result_block.is_error is True
    payload = json.loads(result_block.content)
    assert payload["ok"] is False
    assert payload["error"] == "invalid_arguments"
    assert payload["detail"]["tool"] == "math_add"
    assert isinstance(payload["detail"]["errors"], list)


async def test_dispatch_unknown_tool_returns_unknown_tool_error() -> None:
    reg = ToolRegistry()
    block = ToolUseBlock(id="tu_3", name="nope", input={})
    result_block = await reg.dispatch(block)
    assert result_block.is_error is True
    payload = json.loads(result_block.content)
    assert payload["error"] == "unknown_tool"
    assert payload["detail"]["tool"] == "nope"


async def test_dispatch_handler_exception_becomes_tool_error_result() -> None:
    async def boom(_: _NoParams) -> _PingResult:
        raise RuntimeError("kaboom")

    spec: ToolSpec[_NoParams, _PingResult] = ToolSpec(
        name="boom",
        description="raises",
        params_model=_NoParams,
        result_model=_PingResult,
        handler=boom,
    )
    reg = ToolRegistry()
    reg.register(spec)
    block = ToolUseBlock(id="tu_4", name="boom", input={})

    result_block = await reg.dispatch(block)

    assert result_block.is_error is True
    payload = json.loads(result_block.content)
    assert payload["error"] == "handler_exception"
    assert payload["detail"]["type"] == "RuntimeError"
    assert payload["detail"]["message"] == "kaboom"
