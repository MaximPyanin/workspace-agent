from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from workplace_agent.llm.types import ToolDefinition, ToolResultBlock, ToolUseBlock


class ToolErrorResult(BaseModel):
    ok: bool = False
    error: str
    detail: dict[str, Any] = Field(default_factory=dict)


Handler = Callable[[Any], Awaitable[BaseModel]]


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    raw.pop("title", None)
    schema = _normalize(raw, defs)
    schema.setdefault("type", "object")
    return schema


UNSUPPORTED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
        "format",
    }
)


def _normalize(node: Any, defs: dict[str, Any]) -> Any:

    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            if ref.startswith("#/$defs/"):
                target = defs.get(ref[len("#/$defs/") :])
                if target is not None:
                    merged = {k: v for k, v in node.items() if k != "$ref"}
                    for k, v in target.items():
                        merged.setdefault(k, v)
                    return _normalize(merged, defs)
        out: dict[str, Any] = {
            k: _normalize(v, defs) for k, v in node.items() if k not in UNSUPPORTED_SCHEMA_KEYWORDS
        }
        if out.get("type") == "object" and isinstance(out.get("properties"), dict):
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
            for prop in out["properties"].values():
                if isinstance(prop, dict):
                    prop.pop("title", None)
                    prop.pop("default", None)
        return out
    if isinstance(node, list):
        return [_normalize(item, defs) for item in node]
    return node


@dataclass(frozen=True)
class ToolSpec[P: BaseModel, R: BaseModel]:
    name: str
    description: str
    params_model: type[P]
    result_model: type[R]
    handler: Callable[[P], Awaitable[R | ToolErrorResult]]

    def definition(self) -> ToolDefinition:

        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=_strict_schema(self.params_model),
            strict=False,
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec[Any, Any]] = {}

    def register(self, spec: ToolSpec[Any, Any]) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool name: {spec.name!r}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec[Any, Any] | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def list_definitions(self) -> list[ToolDefinition]:
        return [spec.definition() for spec in self._tools.values()]

    async def dispatch(self, tool_use: ToolUseBlock) -> ToolResultBlock:
        spec = self._tools.get(tool_use.name)
        if spec is None:
            err = ToolErrorResult(
                error="unknown_tool",
                detail={"tool": tool_use.name},
            )
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=err.model_dump_json(),
                is_error=True,
            )

        try:
            params = spec.params_model.model_validate(tool_use.input)
        except ValidationError as exc:
            err = ToolErrorResult(
                error="invalid_arguments",
                detail={"tool": tool_use.name, "errors": exc.errors()},
            )
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=err.model_dump_json(),
                is_error=True,
            )

        try:
            result = await spec.handler(params)
        except Exception as exc:
            err = ToolErrorResult(
                error="handler_exception",
                detail={"tool": tool_use.name, "type": type(exc).__name__, "message": str(exc)},
            )
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=err.model_dump_json(),
                is_error=True,
            )

        if isinstance(result, ToolErrorResult):
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=result.model_dump_json(),
                is_error=True,
            )

        if not isinstance(result, spec.result_model):
            err = ToolErrorResult(
                error="invalid_handler_result",
                detail={
                    "tool": tool_use.name,
                    "expected": spec.result_model.__name__,
                    "got": type(result).__name__,
                },
            )
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=err.model_dump_json(),
                is_error=True,
            )

        return ToolResultBlock(
            tool_use_id=tool_use.id,
            content=result.model_dump_json(),
            is_error=False,
        )
