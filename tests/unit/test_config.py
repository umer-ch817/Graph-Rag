"""Configuration contracts: local scope, safe budgets, and secret handling."""

import json

import pytest
from pydantic import ValidationError

from graphrag_lab.config import AppConfig, BudgetConfig, load_config


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    import os

    for key in os.environ:
        if key.startswith("GRAPHRAG_"):
            monkeypatch.delenv(key)


def test_defaults_are_local_and_do_not_enable_llm_calls():
    config = AppConfig()
    assert config.neo4j.uri == "bolt://127.0.0.1:7687"
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    with pytest.raises(ValueError, match="LLM"):
        config.require_llm_ready()


@pytest.mark.parametrize("value", ["", "   ", "model with spaces", "../model"])
def test_invalid_embedding_model_is_rejected(value):
    with pytest.raises(ValidationError):
        AppConfig(embedding_model=value)


@pytest.mark.parametrize(
    "uri",
    [
        "bolt://example.com:7687",
        "bolt://0.0.0.0:7687",
        "https://localhost:7687",
        "bolt://user:password@localhost:7687",
        "bolt://localhost:7687/private",
    ],
)
def test_nonlocal_or_credential_bearing_uri_is_rejected(uri):
    with pytest.raises(ValidationError):
        AppConfig(neo4j={"uri": uri})


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_call_cap_must_be_a_positive_integer(value):
    with pytest.raises(ValidationError):
        BudgetConfig(max_calls=value)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_cost_cap_must_be_positive_and_finite(value):
    with pytest.raises(ValidationError):
        BudgetConfig(max_cost_usd=value)


def test_unknown_price_requires_a_call_or_token_cap():
    with pytest.raises(ValidationError, match="call or token cap"):
        BudgetConfig(max_calls=None, max_total_tokens=None, max_cost_usd=5)


def test_yaml_load_and_environment_override(tmp_path, monkeypatch):
    path = tmp_path / "settings.yaml"
    path.write_text("budget:\n  max_calls: 7\n", encoding="utf-8")
    monkeypatch.setenv("GRAPHRAG_BUDGET__MAX_CALLS", "3")
    assert load_config(path).budget.max_calls == 3


def test_nested_environment_override_preserves_other_yaml_fields(tmp_path, monkeypatch):
    path = tmp_path / "settings.yaml"
    path.write_text("budget:\n  max_calls: 7\n  max_total_tokens: 900\n", encoding="utf-8")
    monkeypatch.setenv("GRAPHRAG_BUDGET__MAX_CALLS", "3")
    config = load_config(path)
    assert (config.budget.max_calls, config.budget.max_total_tokens) == (3, 900)


def test_local_dotenv_is_not_implicitly_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GRAPHRAG_LLM_API_KEY=not-for-python\n", encoding="utf-8")
    assert AppConfig().llm_api_key is None


@pytest.mark.parametrize("value", ["", "  ", "model with spaces"])
def test_invalid_llm_model_is_rejected(value):
    with pytest.raises(ValidationError):
        AppConfig(llm_model=value)


def test_empty_secret_is_rejected_without_echoing_input(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_LLM_API_KEY", "   ")
    with pytest.raises(ValidationError) as error:
        AppConfig()
    assert "input_value=" not in str(error.value)


def test_credentials_only_come_from_environment(tmp_path, monkeypatch):
    path = tmp_path / "settings.yaml"
    path.write_text("llm_api_key: do-not-save-this\n", encoding="utf-8")
    with pytest.raises(ValueError, match="environment"):
        load_config(path)
    monkeypatch.setenv("GRAPHRAG_LLM_API_KEY", "test-secret-value")
    monkeypatch.setenv("GRAPHRAG_LLM_MODEL", "test-model")
    config = AppConfig()
    config.require_llm_ready()
    assert "test-secret-value" not in repr(config)
    assert "test-secret-value" not in json.dumps(config.model_dump(mode="json"))


def test_neo4j_password_is_not_serialized(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_NEO4J_PASSWORD", "test-password-value")
    config = AppConfig()
    assert "test-password-value" not in config.model_dump_json()
    assert "test-password-value" not in repr(config)


def test_unknown_configuration_fields_are_rejected(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("embedding_modle: typo\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("content", ["- invalid\n", "false\n"])
def test_yaml_root_must_be_a_mapping(tmp_path, content):
    path = tmp_path / "settings.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
