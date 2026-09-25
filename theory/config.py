from __future__ import annotations
import json
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ConfigurationError
from .paths import CONFIG_PATH


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_budget_usd: float = Field(default=100.0, gt=0)
    openai_model: str = "gpt-5.6-sol"
    anthropic_model: str = "claude-opus-5-5"

    @classmethod
    def load(cls) -> "Config":
        if not CONFIG_PATH.exists():
            raise ConfigurationError(
                f"Workspace config is missing at {CONFIG_PATH}. Recreate it or run `theory init`."
            )
        try:
            return cls.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Invalid workspace config at {CONFIG_PATH}: {exc}") from exc

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CONFIG_PATH.with_suffix(".json.tmp")
        temporary.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(CONFIG_PATH)
