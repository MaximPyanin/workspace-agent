from typing import Annotated, Literal, Self, get_args

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ErrorType = Literal["rate_limit", "not_found", "timeout"]
_VALID_ERROR_TYPES: frozenset[str] = frozenset(get_args(ErrorType))

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_BETA_HEADER = "structured-outputs-2025-11-13"
DEFAULT_MOCK_API_HOST = "127.0.0.1"
DEFAULT_MOCK_API_PORT = 8765
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_AGENT_MAX_STEPS = 12
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0
DEFAULT_LLM_MAX_RETRIES = 3

MISSING_API_KEY_ERROR = (
    "ANTHROPIC_API_KEY is required when AGENT_USE_FAKE_LLM is not set. "
    "Set ANTHROPIC_API_KEY in your environment or .env file, or set "
    "AGENT_USE_FAKE_LLM=1 to use the scripted fake LLM."
)


class ErrorInjectionRule(BaseModel):
    service: str
    action: str | None = None
    error_type: ErrorType

    def matches(self, service: str, action: str) -> bool:
        if self.service != service:
            return False
        return self.action is None or self.action == action


def _parse_rule(raw: str) -> ErrorInjectionRule:
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) == 2:
        service, error_type = parts
        action: str | None = None
    elif len(parts) == 3:
        service, action, error_type = parts
    else:
        raise ValueError(
            f"invalid error injection rule {raw!r}: expected 'service:type' or "
            "'service:action:type'"
        )
    if not service or not error_type or (action is not None and not action):
        raise ValueError(f"invalid error injection rule {raw!r}: empty component")
    if error_type not in _VALID_ERROR_TYPES:
        raise ValueError(
            f"invalid error injection rule {raw!r}: error_type must be one of "
            f"{sorted(_VALID_ERROR_TYPES)}"
        )
    return ErrorInjectionRule.model_validate(
        {"service": service, "action": action, "error_type": error_type}
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    anthropic_api_key: str = ""
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    anthropic_beta_header: str = DEFAULT_ANTHROPIC_BETA_HEADER

    mock_api_host: str = DEFAULT_MOCK_API_HOST
    mock_api_port: int = DEFAULT_MOCK_API_PORT
    mock_api_error_injection: Annotated[list[ErrorInjectionRule], NoDecode] = Field(
        default_factory=list
    )

    log_level: str = DEFAULT_LOG_LEVEL
    log_file: str | None = None

    agent_max_steps: int = Field(default=DEFAULT_AGENT_MAX_STEPS, ge=1)
    llm_timeout_seconds: float = Field(default=DEFAULT_LLM_TIMEOUT_SECONDS, gt=0)
    llm_max_retries: int = Field(default=DEFAULT_LLM_MAX_RETRIES, ge=0)
    agent_use_fake_llm: bool = False

    @field_validator("mock_api_error_injection", mode="before")
    @classmethod
    def _parse_error_injection(cls, value: object) -> object:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [_parse_rule(part) for part in value.split(",") if part.strip()]
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("log_file", mode="before")
    @classmethod
    def _empty_log_file_is_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def _require_api_key_unless_fake(self) -> Self:
        if not self.agent_use_fake_llm and not self.anthropic_api_key:
            raise ValueError(MISSING_API_KEY_ERROR)
        return self
