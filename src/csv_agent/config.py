import os
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

CONFIGURATION_HINTS = {
    "thinking_conflict": "Conflicting thinking controls; set MODEL_DISABLE_THINKING=false to use MODEL_THINKING, or leave MODEL_THINKING empty to disable thinking",
    "thinking_protocol": "Explicit thinking controls require Anthropic Messages; use MODEL_PROTOCOL=anthropic or omit both thinking controls",
    "spark_temperature": "This Spark service only supports temperature=1 or omission; set MODEL_TEMPERATURE=1 or leave it empty",
}


def configuration_error_message(error: ValidationError) -> str:
    """Return only allowlisted hints, never validation inputs or exception details."""
    for item in error.errors(
        include_input=False, include_context=False, include_url=False
    ):
        if item["type"] in CONFIGURATION_HINTS:
            return CONFIGURATION_HINTS[item["type"]]
    return "Invalid model configuration or run limits; check environment settings"


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    protocol: Literal["anthropic", "openai"] = "anthropic"
    endpoint: str
    name: str = Field(min_length=1)
    api_key_env: str = "MODEL_API_KEY"
    temperature: float | None = Field(default=None, ge=0, le=2)
    timeout_seconds: float = Field(default=90, gt=0, le=600)
    max_tokens: int = Field(default=4096, gt=0, le=32768)
    disable_thinking: bool = False
    thinking: Literal["enabled", "adaptive"] | None = None
    instruction_mode: Literal["structured", "tools"] = "structured"

    @model_validator(mode="after")
    def validate_provider_options(self) -> "ModelConfig":
        if self.disable_thinking and self.thinking is not None:
            raise PydanticCustomError(
                "thinking_conflict", CONFIGURATION_HINTS["thinking_conflict"]
            )
        if self.protocol != "anthropic" and (
            self.disable_thinking or self.thinking is not None
        ):
            raise PydanticCustomError(
                "thinking_protocol", CONFIGURATION_HINTS["thinking_protocol"]
            )
        if (
            urlsplit(self.endpoint).hostname == "maas-api.cn-huabei-1.xf-yun.com"
            and self.name in {"spark-x2.6", "spark-x2.6-4b", "spark-x2.6-preview"}
            and self.temperature not in (None, 1)
        ):
            raise PydanticCustomError(
                "spark_temperature", CONFIGURATION_HINTS["spark_temperature"]
            )
        return self

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        # Accessing port validates syntax/range instead of deferring to the HTTP client.
        if parsed.port is not None and parsed.port == 0:
            raise ValueError("Use a valid HTTPS port")
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Use a full HTTPS request URL without credentials or query parameters"
            )
        return value

    @classmethod
    def from_env(cls) -> "ModelConfig":
        values = {
            field: os.environ[key]
            for field, key in {
                "protocol": "MODEL_PROTOCOL",
                "endpoint": "MODEL_ENDPOINT",
                "name": "MODEL_NAME",
                "api_key_env": "MODEL_API_KEY_ENV",
                "temperature": "MODEL_TEMPERATURE",
                "timeout_seconds": "MODEL_TIMEOUT_SECONDS",
                "max_tokens": "MODEL_MAX_TOKENS",
                "disable_thinking": "MODEL_DISABLE_THINKING",
                "thinking": "MODEL_THINKING",
                "instruction_mode": "MODEL_INSTRUCTION_MODE",
            }.items()
            if os.environ.get(key)
        }
        return cls.model_validate(values)


class RunLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_executions: int = Field(default=3, ge=1, le=20)
    max_format_repairs: int = Field(default=1, ge=0, le=10)
    max_model_retries: int = Field(default=0, ge=0, le=10)
    max_model_calls: int | None = Field(default=None, ge=1, le=40)
    task_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    image: str = "csv-agent-python:0.1"
    timeout_seconds: float = Field(default=20, gt=0, le=600)
    memory_mb: int = Field(default=512, ge=64, le=8192)
    cpus: float = Field(default=1, gt=0, le=8)
    pids: int = Field(default=64, ge=8, le=512)
    output_bytes: int = Field(default=1048576, ge=16384, le=16777216)
    input_max_bytes: int = Field(default=10485760, ge=1, le=104857600)

    @property
    def model_call_budget(self) -> int:
        if self.max_model_calls is not None:
            return self.max_model_calls
        return min(
            40, self.max_executions + self.max_format_repairs + self.max_model_retries
        )

    def for_single_generation(self) -> "RunLimits":
        return self.model_copy(
            update={
                "max_executions": 1,
                "max_format_repairs": 0,
                "max_model_retries": 0,
                "max_model_calls": 1,
            }
        )

    @classmethod
    def from_env(cls, *, exploration: bool = False) -> "RunLimits":
        keys = {
            "max_executions": "MAX_EXECUTIONS",
            "max_format_repairs": "MAX_FORMAT_REPAIRS",
            "max_model_retries": "MAX_MODEL_RETRIES",
            "max_model_calls": "MAX_MODEL_CALLS",
            "task_timeout_seconds": "TASK_TIMEOUT_SECONDS",
            "image": "EXECUTION_IMAGE",
            "timeout_seconds": "EXECUTION_TIMEOUT_SECONDS",
            "memory_mb": "EXECUTION_MEMORY_MB",
            "cpus": "EXECUTION_CPUS",
            "pids": "EXECUTION_PIDS",
            "output_bytes": "EXECUTION_OUTPUT_BYTES",
            "input_max_bytes": "INPUT_MAX_BYTES",
        }
        defaults: dict[str, int | float | str] = (
            {"max_model_calls": 16, "max_executions": 12, "task_timeout_seconds": 1200}
            if exploration
            else {}
        )
        return cls.model_validate(
            {
                **defaults,
                **{f: os.environ[k] for f, k in keys.items() if os.environ.get(k)},
            }
        )
