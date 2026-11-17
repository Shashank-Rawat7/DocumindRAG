"""Tests for app/core/chat_config.py (C9.1): pure, no database, no network."""

import dataclasses
import importlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys

import pytest

from app.core import chat_config
from app.core.chat_config import (
    DEFAULT_MODEL,
    ENV_DAILY_TOKEN_LIMIT,
    ENV_GROQ_API_KEY,
    ENV_MODEL,
    LEASE_MARGIN_SECONDS,
    LEASE_SECONDS,
    MAX_COMPLETION_TOKENS,
    MAX_DAILY_TOKEN_LIMIT,
    MAX_REQUEST_RESERVATION,
    PER_REQUEST_TOKEN_CEILING,
    PROMPT_TOKEN_BUDGET,
    PROVIDER_TIMEOUT_SECONDS,
    RETRY_AFTER_MAX_SECONDS,
    TOP_K,
    ChatConfigError,
    ChatSettings,
    load_chat_settings,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
KEY = "gsk_test_key_value_0123456789"


def good_env(**overrides):
    env = {ENV_DAILY_TOKEN_LIMIT: "100000", ENV_GROQ_API_KEY: KEY}
    env.update(overrides)
    return {k: v for k, v in env.items() if v is not None}


class TestValidLoad:
    def test_loads_required_values_with_default_model(self):
        s = load_chat_settings(good_env())
        assert isinstance(s, ChatSettings)
        assert s.daily_token_limit == 100000
        assert s.model == DEFAULT_MODEL == "openai/gpt-oss-20b"
        assert s.groq_api_key.get_secret_value() == KEY

    def test_model_override(self):
        assert load_chat_settings(
            good_env(**{ENV_MODEL: "openai/gpt-oss-120b"})
        ).model == ("openai/gpt-oss-120b")

    def test_does_not_mutate_the_mapping(self):
        env = good_env()
        before = dict(env)
        load_chat_settings(env)
        assert env == before

    def test_default_source_is_the_process_environment_read_at_call_time(
        self, monkeypatch
    ):
        monkeypatch.delenv(ENV_DAILY_TOKEN_LIMIT, raising=False)
        monkeypatch.delenv(ENV_GROQ_API_KEY, raising=False)
        with pytest.raises(ChatConfigError):
            load_chat_settings()
        monkeypatch.setenv(ENV_DAILY_TOKEN_LIMIT, "250000")
        monkeypatch.setenv(ENV_GROQ_API_KEY, KEY)
        assert load_chat_settings().daily_token_limit == 250000

    def test_settings_are_frozen_and_slotted(self):
        s = load_chat_settings(good_env())
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.daily_token_limit = 1  # type: ignore[misc]
        assert not hasattr(s, "__dict__")


class TestDailyLimit:
    def test_missing_is_an_error_not_a_default(self):
        with pytest.raises(ChatConfigError, match=ENV_DAILY_TOKEN_LIMIT):
            load_chat_settings(good_env(**{ENV_DAILY_TOKEN_LIMIT: None}))

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            " 100000",
            "100000 ",
            "100000\n",
            "100_000",
            "100,000",
            "100000.0",
            "1e5",
            "0x186a0",
            "+100000",
            "-100000",
            "0",
            "000",
            "0100000",
            "True",
            "none",
            "\u0665\u0660\u0660\u0660\u0660",  # Arabic-Indic digits: str.isdigit() accepts them
            "\uff11\uff10\uff10\uff10\uff10\uff10",  # full-width digits
        ],
    )
    def test_rejects_anything_but_plain_ascii_positive_integers(self, raw):
        with pytest.raises(ChatConfigError, match=ENV_DAILY_TOKEN_LIMIT):
            load_chat_settings(good_env(**{ENV_DAILY_TOKEN_LIMIT: raw}))

    def test_non_string_value_is_rejected(self):
        with pytest.raises(ChatConfigError, match=ENV_DAILY_TOKEN_LIMIT):
            load_chat_settings(good_env(**{ENV_DAILY_TOKEN_LIMIT: 100000}))

    def test_lower_bound_is_the_largest_single_reservation(self):
        assert (
            load_chat_settings(
                good_env(**{ENV_DAILY_TOKEN_LIMIT: str(MAX_REQUEST_RESERVATION)})
            ).daily_token_limit
            == MAX_REQUEST_RESERVATION
        )
        with pytest.raises(ChatConfigError, match="at least"):
            load_chat_settings(
                good_env(**{ENV_DAILY_TOKEN_LIMIT: str(MAX_REQUEST_RESERVATION - 1)})
            )

    def test_upper_bound_is_int4(self):
        assert MAX_DAILY_TOKEN_LIMIT == 2**31 - 1
        assert (
            load_chat_settings(
                good_env(**{ENV_DAILY_TOKEN_LIMIT: str(MAX_DAILY_TOKEN_LIMIT)})
            ).daily_token_limit
            == MAX_DAILY_TOKEN_LIMIT
        )
        with pytest.raises(ChatConfigError, match="at most"):
            load_chat_settings(
                good_env(**{ENV_DAILY_TOKEN_LIMIT: str(MAX_DAILY_TOKEN_LIMIT + 1)})
            )

    def test_huge_digit_string_is_rejected_without_error_noise(self):
        with pytest.raises(ChatConfigError, match="at most"):
            load_chat_settings(good_env(**{ENV_DAILY_TOKEN_LIMIT: "9" * 400}))


class TestApiKey:
    def test_missing(self):
        with pytest.raises(ChatConfigError, match=ENV_GROQ_API_KEY):
            load_chat_settings(good_env(**{ENV_GROQ_API_KEY: None}))

    def test_empty(self):
        with pytest.raises(ChatConfigError, match="must not be empty"):
            load_chat_settings(good_env(**{ENV_GROQ_API_KEY: ""}))

    @pytest.mark.parametrize(
        "raw", [KEY + "\n", " " + KEY, KEY + " ", "gsk with space", "gsk\ttab"]
    )
    def test_whitespace_is_rejected(self, raw):
        with pytest.raises(ChatConfigError, match="whitespace"):
            load_chat_settings(good_env(**{ENV_GROQ_API_KEY: raw}))

    def test_non_string_value_is_rejected(self):
        with pytest.raises(ChatConfigError, match=ENV_GROQ_API_KEY):
            load_chat_settings(good_env(**{ENV_GROQ_API_KEY: 12345}))

    def test_secret_is_masked_in_repr_and_str(self):
        s = load_chat_settings(good_env())
        assert KEY not in repr(s)
        assert KEY not in str(s)
        assert KEY not in repr(s.groq_api_key)


class TestModel:
    @pytest.mark.parametrize(
        "raw",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "a",
            "m:1.5_b",
        ],
    )
    def test_accepts_plausible_ids(self, raw):
        assert load_chat_settings(good_env(**{ENV_MODEL: raw})).model == raw

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            " ",
            " openai/x",
            "openai/x ",
            "a b",
            "x" * 101,
            "/x",
            "-x",
            "x\n",
            "モデル",
        ],
    )
    def test_rejects_malformed_ids_instead_of_falling_back_to_the_default(self, raw):
        with pytest.raises(ChatConfigError, match=ENV_MODEL):
            load_chat_settings(good_env(**{ENV_MODEL: raw}))

    def test_non_string_value_is_rejected(self):
        with pytest.raises(ChatConfigError, match=ENV_MODEL):
            load_chat_settings(good_env(**{ENV_MODEL: 7}))


class TestErrorReporting:
    def test_every_problem_is_reported_in_one_error(self):
        with pytest.raises(ChatConfigError) as info:
            load_chat_settings({ENV_MODEL: "bad model"})
        message = str(info.value)
        assert ENV_DAILY_TOKEN_LIMIT in message
        assert ENV_GROQ_API_KEY in message
        assert ENV_MODEL in message

    def test_messages_never_contain_secret_or_bad_values(self):
        env = good_env(
            **{
                ENV_DAILY_TOKEN_LIMIT: "12 3456",
                ENV_GROQ_API_KEY: "sk-secret value",
                ENV_MODEL: "bad model",
            }
        )
        with pytest.raises(ChatConfigError) as info:
            load_chat_settings(env)
        message = str(info.value)
        for leaked in ("12 3456", "sk-secret", "bad model"):
            assert leaked not in message

    def test_dotenv_file_is_not_consulted(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            f"{ENV_DAILY_TOKEN_LIMIT}=100000\n{ENV_GROQ_API_KEY}={KEY}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ChatConfigError):
            load_chat_settings({})


class TestConstants:
    def test_approved_values(self):
        assert TOP_K == 4
        assert MAX_COMPLETION_TOKENS == 1024
        assert PROMPT_TOKEN_BUDGET == 3500
        assert PROVIDER_TIMEOUT_SECONDS == 30
        assert LEASE_SECONDS == 60
        assert RETRY_AFTER_MAX_SECONDS == 15
        assert (
            MAX_REQUEST_RESERVATION
            == PROMPT_TOKEN_BUDGET + MAX_COMPLETION_TOKENS
            == 4524
        )

    def test_lease_outlasts_the_provider_timeout_with_margin(self):
        assert LEASE_SECONDS >= PROVIDER_TIMEOUT_SECONDS + LEASE_MARGIN_SECONDS

    def test_reservation_fits_the_per_request_ceiling(self):
        assert MAX_REQUEST_RESERVATION <= PER_REQUEST_TOKEN_CEILING

    def test_ceiling_is_documented_as_not_a_tpm_safeguard(self):
        doc = chat_config.__doc__ or ""
        assert "not a safeguard for Groq's tokens-per-minute" in doc
        assert "MVP limitation" in doc

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("LEASE_SECONDS", 44),
            ("PROMPT_TOKEN_BUDGET", 5000),
            ("MAX_REQUEST_RESERVATION", PER_REQUEST_TOKEN_CEILING + 1),
        ],
    )
    def test_invariant_check_fires_when_violated(self, monkeypatch, name, value):
        monkeypatch.setattr(chat_config, name, value)
        if name == "PROMPT_TOKEN_BUDGET":
            monkeypatch.setattr(
                chat_config, "MAX_REQUEST_RESERVATION", value + MAX_COMPLETION_TOKENS
            )
        with pytest.raises(RuntimeError):
            chat_config._check_constants()


def _run_python(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CHAT_", "GROQ_"))}
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestImportIsolation:
    def test_import_needs_no_environment_and_pulls_in_no_app_settings_or_providers(
        self,
    ):
        code = (
            "import json, sys\n"
            "import app.core.chat_config\n"
            "print(json.dumps({m: m in sys.modules for m in "
            "('app.core.config', 'app.core.database', 'groq', 'langgraph')}))\n"
        )
        result = _run_python(code)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            "app.core.config": False,
            "app.core.database": False,
            "groq": False,
            "langgraph": False,
        }


class TestDependencies:
    @pytest.mark.parametrize(
        ("package", "version"), [("groq", "1.7.0"), ("langgraph", "1.2.22")]
    )
    def test_pinned_version_is_installed(self, package, version):
        assert importlib.metadata.version(package) == version

    @pytest.mark.parametrize("module", ["groq", "langgraph", "langgraph.graph"])
    def test_importable(self, module):
        importlib.import_module(module)

    def test_requirements_pin_both_exactly(self):
        lines = {
            line.strip()
            for line in (PROJECT_ROOT / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        }
        assert "groq==1.7.0" in lines
        assert "langgraph==1.2.22" in lines
