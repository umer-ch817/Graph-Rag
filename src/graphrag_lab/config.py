"""Validated local settings; credentials come from process environment only."""

import re
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class LocalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)


class Neo4jConfig(LocalModel):
    uri: str = "bolt://127.0.0.1:7687"
    database: str = Field(default="neo4j", min_length=1, pattern=r"^[A-Za-z0-9.-]+$")

    @field_validator("uri")
    @classmethod
    def local_bolt_uri(cls, value: str) -> str:
        uri = urlsplit(value)
        if (
            uri.scheme not in {"bolt", "neo4j"}
            or uri.hostname not in {"localhost", "127.0.0.1", "::1"}
            or uri.username is not None
            or uri.password is not None
            or uri.path
            or uri.query
            or uri.fragment
            or uri.port is None
            or uri.port == 0
            or any(character.isspace() for character in value)
        ):
            raise ValueError("Neo4j URI must use a local Bolt host and port without credentials")
        return value


class BudgetConfig(LocalModel):
    max_calls: int | None = Field(default=20, gt=0)
    max_total_tokens: int | None = Field(default=20_000, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("max_calls", "max_total_tokens", mode="before")
    @classmethod
    def integer_limit(cls, value):
        # Environment strings are supported; booleans/floats are not integer budgets.
        if value is not None and not (
            type(value) is int or (isinstance(value, str) and value.isascii() and value.isdigit())
        ):
            raise ValueError("Budget limit must be a positive integer")
        return value

    @field_validator("max_cost_usd", mode="before")
    @classmethod
    def numeric_cost(cls, value):
        if isinstance(value, bool):
            raise ValueError("Cost must be a number, not a boolean")
        return value

    @model_validator(mode="after")
    def bounded_without_pricing(self) -> Self:
        if self.max_calls is None and self.max_total_tokens is None:
            raise ValueError("A call or token cap is required while provider pricing is unknown")
        return self


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRAPHRAG_",
        env_nested_delimiter="__",
        extra="forbid",
        hide_input_in_errors=True,
        frozen=True,
    )

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str | None = None
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    neo4j_password: SecretStr | None = Field(default=None, exclude=True, repr=False)
    llm_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment overrides file defaults. No implicit dotenv or secrets-directory reads.
        return env_settings, init_settings

    @field_validator("embedding_model", "reranker_model")
    @classmethod
    def repository_model_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("Model id must be an organization/name repository id")
        return value

    @field_validator("llm_model")
    @classmethod
    def llm_model_id(cls, value: str | None) -> str | None:
        if value is not None and (not value or any(character.isspace() for character in value)):
            raise ValueError("LLM model id must be nonempty and contain no whitespace")
        return value

    @field_validator("neo4j_password", "llm_api_key")
    @classmethod
    def nonempty_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("Credential must be nonempty when supplied")
        return value

    def require_llm_ready(self) -> None:
        """Later adapters must call this before making an LLM request."""
        if self.llm_model is None or self.llm_api_key is None:
            raise ValueError("LLM model and API key must be configured before requests")


def load_config(path: str | Path) -> AppConfig:
    """Read non-secret YAML, then apply GRAPHRAG_* environment overrides."""
    content = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if content is None:
        content = {}
    if not isinstance(content, dict):
        raise ValueError("Configuration root must be a mapping")
    if {"neo4j_password", "llm_api_key"}.intersection(content):
        raise ValueError("Credentials must be supplied through process environment variables")
    return AppConfig(**content)
