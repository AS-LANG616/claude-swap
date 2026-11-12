"""Tests for gateway (custom-endpoint) account support.

A gateway is its own account ``kind``: a custom Anthropic-compatible base URL
plus a raw API key of *any* prefix. It activates via a ``settings.json``
overlay (``apiKeyHelper`` + ``env.ANTHROPIC_BASE_URL``) and never touches the
live ``.credentials.json``. Covers URL validation, the settings overlay and its
refusals, the credential helper, the add/switch/remove lifecycle, and the JSON
surface.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import gateway
from claude_swap.exceptions import ConfigError, SwitchError
from claude_swap.json_output import (
    USAGE_GATEWAY,
    usage_fields,
    account_row,
)
from claude_swap.models import Platform
from claude_swap.paths import get_credentials_path
from claude_swap.switcher import ClaudeAccountSwitcher

# Deliberately not an ``sk-ant-…`` prefix: a gateway key can be any string.
GATEWAY_KEY = "gw-key-0123456789abcdefghijklmnop"
LIVE_MARKER = json.dumps({"claudeAiOauth": {"accessToken": "live", "refreshToken": "keepme"}})


def _linux_switcher() -> ClaudeAccountSwitcher:
    s = ClaudeAccountSwitcher()
    s.platform = Platform.LINUX
    s._setup_directories()
    s._init_sequence_file()
    return s


def _settings(s: ClaudeAccountSwitcher) -> dict:
    path = s.gateway_settings_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ---------------------------------------------------------------------------
# URL validation + record fields (pure)
# ---------------------------------------------------------------------------


class TestValidateUrl:
    @pytest.mark.parametrize(
        "value",
        [
            "http://gw.example.com",
            "http://remote.example.com/v1",
            "https://user:pass@gw.example.com",  # embedded credentials
            "https://gw.example.com?x=1",  # query
            "https://gw.example.com#frag",  # fragment
            "ftp://gw.example.com",
            "",
        ],
    )
    def test_rejected(self, value):
        with pytest.raises(ValueError):
            gateway.validate_url(value)
        with pytest.raises(ValueError):
            gateway.account_fields(value)

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("https://gw.example.com", "https://gw.example.com"),
            ("https://gw.example.com/", "https://gw.example.com"),  # trailing slash
            ("https://gw.example.com/base", "https://gw.example.com/base"),
            # Trailing /v1 is normalised away only for a base URL.
            ("https://gw.example.com/v1", "https://gw.example.com"),
            ("https://gw.example.com/api/v1/", "https://gw.example.com/api"),
            ("http://localhost:8080/v1", "http://localhost:8080"),  # localhost http ok
            ("http://127.0.0.1:9000/v1", "http://127.0.0.1:9000"),
        ],
    )
    def test_normalized(self, value, expected):
        assert gateway.validate_url(value, base=True) == expected

    def test_error_does_not_echo_url(self):
        with pytest.raises(ValueError) as exc:
            gateway.validate_url("http://gw.example.com?secret=q")
        assert "secret" not in str(exc.value)

    def test_account_fields(self):
        f = gateway.account_fields("https://gw.example.com/v1/", auth="x-api-key")
        assert f == {"kind": "gateway", "baseUrl": "https://gw.example.com", "auth": "x-api-key"}

    def test_account_fields_default_bearer(self):
        assert gateway.account_fields("https://gw.example.com")["auth"] == "bearer"

    def test_bad_auth_rejected(self):
        with pytest.raises(ValueError):
            gateway.account_fields("https://gw.example.com", auth="basic")


class TestOwnedSlot:
    def test_owned(self):
        for prog in ("cswap", "claude-swap"):
            assert gateway.owned_slot(f"{prog} gateway-credential 7") == "7"

    def test_windows_exe(self):
        assert gateway.owned_slot("/usr/bin/cswap.exe gateway-credential 3") == "3"

    def test_windows_exe_uppercase_extension(self):
        # uv on Windows creates cswap.EXE — must still read as owned,
        # or our own overlay is refused as foreign on the next switch.
        assert gateway.owned_slot(r"C:\Users\x\.local\bin\cswap.EXE gateway-credential 4") == "4"
        assert gateway.owned_slot(r"C:\Users\x\.local\bin\CLAUDE-SWAP.EXE gateway-credential 4") == "4"

    def test_windows_backslashed_path(self):
        # POSIX shlex eats the backslashes of a Windows path (escape
        # handling) — the non-POSIX re-parse must recover our own helper,
        # otherwise the second-ever gateway switch refuses our own overlay.
        value = r"C:\Users\LANG616\.local\bin\cswap.EXE gateway-credential 3"
        assert shlex.split(value) != [
            r"C:\Users\LANG616\.local\bin\cswap.EXE", "gateway-credential", "3"
        ]  # sanity: POSIX mode really does mangle it
        assert gateway.owned_slot(value) == "3"
        # Quoted form (a spaced dir) also stays owned.
        assert gateway.owned_slot(r'"C:\Program Files\cswap\cswap.exe" gateway-credential 5') == "5"

    @pytest.mark.parametrize(
        "value",
        [
            None,
            5,
            "some-other-cli switch --flag",
            "cswap other-subcommand 7",
            "cswap gateway-credential",  # missing slot
            "cswap gateway-credential 0",  # slot must be > 0
            "cswap gateway-credential x",  # non-numeric
        ],
    )
    def test_not_owned(self, value):
        assert gateway.owned_slot(value) is None


class TestWriteSettings:
    @pytest.fixture(autouse=True)
    def _exec(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(gateway, "executable", lambda: "/fake/bin/cswap")

    def _record(self) -> dict:
        return gateway.account_fields("https://gw.example.com")

    def test_activate(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        gateway.write_settings(path, "1", self._record())
        s = json.loads(path.read_text())
        assert s["env"]["ANTHROPIC_BASE_URL"] == "https://gw.example.com"
        assert "gateway-credential" in s["apiKeyHelper"] and s["apiKeyHelper"].endswith("1")

    def test_deactivate_sets_first_party_url(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        gateway.write_settings(path, "1", self._record())
        gateway.write_settings(path, None)
        s = json.loads(path.read_text())
        assert "apiKeyHelper" not in s
        assert s["env"]["ANTHROPIC_BASE_URL"] == gateway.FIRST_PARTY_URL

    def test_refuses_foreign_helper(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"apiKeyHelper": "env ANTHROPIC_API_KEY"}))
        with pytest.raises(ValueError):
            gateway.write_settings(path, "1", self._record())
        # The foreign file is left alone.
        assert json.loads(path.read_text())["apiKeyHelper"] == "env ANTHROPIC_API_KEY"

    def test_refuses_auth_override(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"env": {"ANTHROPIC_API_KEY": "sk-ant-xxx"}}))
        with pytest.raises(ValueError):
            gateway.write_settings(path, "1", self._record())


class TestModelEnv:
    """Model-id mapping: a gateway knows its own names, not `sonnet`/`haiku`.

    The main model is carried in the top-level "model" setting (so in-session
    /model still works); env model knobs are stashed/overridden and all model
    state is scrubbed on deactivation.
    """

    @pytest.fixture(autouse=True)
    def _exec(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(gateway, "executable", lambda: "/fake/bin/cswap")

    def _gw(self, **kw) -> dict:
        return gateway.account_fields("https://gw.example.com", **kw)

    def test_account_fields_model_defaults_small(self):
        f = gateway.account_fields("https://gw.example.com", model="big-1")
        assert f["model"] == "big-1"
        assert f["smallFastModel"] == "big-1"  # small/fast defaults to main

    def test_account_fields_explicit_small(self):
        f = gateway.account_fields("https://gw.example.com", model="big-1", small_fast_model="fast-1")
        assert f["model"] == "big-1"
        assert f["smallFastModel"] == "fast-1"

    def test_account_fields_no_model(self):
        assert "model" not in gateway.account_fields("https://gw.example.com")

    def test_account_fields_empty_model_rejected(self):
        with pytest.raises(ValueError):
            gateway.account_fields("https://gw.example.com", model="  ")

    def test_activate_switches_top_level_model_and_stashes(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"model": "haiku"}))
        gateway.write_settings(path, "1", self._gw(model="qwen3.8-27b", small_fast_model="sovereign-mini"))
        s = json.loads(path.read_text())
        assert s["model"] == "qwen3.8-27b"
        assert s[gateway.PREV_MODEL_KEY] == "haiku"
        env = s["env"]
        assert "ANTHROPIC_MODEL" not in env  # never pinned via env
        assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "sovereign-mini"
        assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sovereign-mini"

    def test_activate_without_model_leaves_selection_untouched(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"model": "haiku", "env": {"FOO": "bar"}}))
        gateway.write_settings(path, "1", gateway.account_fields("https://gw.example.com"))
        s = json.loads(path.read_text())
        assert s["model"] == "haiku"
        assert gateway.PREV_MODEL_KEY not in s
        assert s["env"]["FOO"] == "bar"
        assert "ANTHROPIC_SMALL_FAST_MODEL" not in s["env"]

    def test_roundtrip_restores_model(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(
            {"model": "haiku", "env": {"CLAUDE_CODE_SUBAGENT_MODEL": "haiku"}}
        ))
        gateway.write_settings(path, "1", self._gw(model="qwen3.8-27b", small_fast_model="sovereign-mini"))
        gateway.write_settings(path, None)
        s = json.loads(path.read_text())
        assert s["model"] == "haiku"
        assert gateway.PREV_MODEL_KEY not in s
        env = s["env"]
        for name in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"):
            assert name not in env
        assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "haiku"  # restored from stash

    def test_deactivate_drops_model_when_user_had_none(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        gateway.write_settings(path, "1", self._gw(model="qwen3.8-27b"))
        gateway.write_settings(path, None)
        s = json.loads(path.read_text())
        assert "model" not in s  # not left on the gateway's model id
        assert gateway.PREV_MODEL_KEY not in s

    def test_stale_env_model_equal_to_gateway_is_dropped(self, tmp_path: Path):
        # The residue state that actually happened: a gateway session left
        # ANTHROPIC_MODEL on the gateway id itself. Re-activating must drop it
        # (not stash it), and deactivating must not resurrect it.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(
            {"model": "haiku", "env": {"ANTHROPIC_MODEL": "qwen3.8-27b"}}
        ))
        gateway.write_settings(path, "1", self._gw(model="qwen3.8-27b", small_fast_model="sovereign-mini"))
        s = json.loads(path.read_text())
        assert "ANTHROPIC_MODEL" not in s["env"]
        assert gateway._prev_env_key(gateway.MODEL_ENV) not in s["env"]
        gateway.write_settings(path, None)
        s = json.loads(path.read_text())
        assert "ANTHROPIC_MODEL" not in s["env"]
        assert s["model"] == "haiku"

    def test_user_env_model_stashed_and_restored(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(
            {"model": "haiku", "env": {"ANTHROPIC_MODEL": "claude-sonnet-4-5"}}
        ))
        gateway.write_settings(path, "1", self._gw(model="qwen3.8-27b"))
        s = json.loads(path.read_text())
        assert "ANTHROPIC_MODEL" not in s["env"]
        gateway.write_settings(path, None)
        s = json.loads(path.read_text())
        assert s["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# JSON surface
# ---------------------------------------------------------------------------


def test_usage_fields_gateway():
    assert usage_fields(USAGE_GATEWAY, None) == ("gateway", None)


def test_account_row_gateway_fields():
    row = account_row(
        3, "gw@test.local", "gw", "", True, USAGE_GATEWAY, gateway={"baseUrl": "https://gw.example.com", "auth": "bearer"},
    )
    assert row["usageStatus"] == "gateway"
    assert row["kind"] == "gateway"
    assert row["gateway"] == {"baseUrl": "https://gw.example.com", "auth": "bearer"}
    # Non-gateway rows are untouched: no kind/gateway keys.
    plain = account_row(2, "a@b.c", "", "", False, None)
    assert "kind" not in plain and "gateway" not in plain


# ---------------------------------------------------------------------------
# Add lifecycle (temp home, file-backed credentials)
# ---------------------------------------------------------------------------


class TestAddGateway:
    def test_add_normalizes_and_stores_raw_key(self, temp_home: Path, capsys):
        s = _linux_switcher()
        s.add_account_from_token(
            GATEWAY_KEY, email="gw@test.local", base_url="https://gw.example.com/v1/",
        )
        data = s._get_sequence_data()
        rec = data["accounts"]["1"]
        assert rec["kind"] == "gateway"
        assert rec["baseUrl"] == "https://gw.example.com"  # /v1 stripped
        assert rec["auth"] == "bearer"
        assert rec["email"] == "gw@test.local"
        # The raw key (any prefix) is stored verbatim as the backup credential.
        assert s._read_account_credentials("1", "gw@test.local") == GATEWAY_KEY
        # A tag rendered the host · auth.
        assert "gw.example.com" in capsys.readouterr().out

    def test_synthetic_email_and_alias(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token(GATEWAY_KEY, base_url="https://gw.example.com", alias="mygw")
        rec = s._get_sequence_data()["accounts"]["1"]
        assert rec["email"] == "mygw-1@token.local"
        assert rec["organizationName"] == "mygw"
        assert rec["alias"] == "mygw"

    def test_add_does_not_touch_live_credentials(self, temp_home: Path):
        s = _linux_switcher()
        live = get_credentials_path()
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(LIVE_MARKER, encoding="utf-8")
        s.add_account_from_token(GATEWAY_KEY, base_url="https://gw.example.com")
        assert live.read_text(encoding="utf-8") == LIVE_MARKER

    def test_same_email_gateway_refresh_in_place(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token(GATEWAY_KEY, email="gw@test.local", base_url="https://old.example.com")
        s.add_account_from_token("rotated-key-42", email="gw@test.local", base_url="https://new.example.com")
        data = s._get_sequence_data()
        assert len(data["sequence"]) == 1
        rec = data["accounts"]["1"]
        assert rec["baseUrl"] == "https://new.example.com"
        assert s._read_account_credentials("1", "gw@test.local") == "rotated-key-42"

    def test_occupied_slot_rejected(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token(GATEWAY_KEY, email="a@test.local", slot=1, base_url="https://gw.example.com")
        with pytest.raises(ConfigError):
            s.add_account_from_token(GATEWAY_KEY, email="b@test.local", slot=1, base_url="https://gw2.example.com")

    def test_missing_base_url_is_oauth_path(self, temp_home: Path):
        # Without --base-url the token is treated as a first-party key/token,
        # never a gateway (no baseUrl ever written).
        s = _linux_switcher()
        s.add_account_from_token("sk-ant-api03-" + "a" * 40)
        rec = s._get_sequence_data()["accounts"]["1"]
        assert rec.get("kind") != "gateway" and "baseUrl" not in rec


# ---------------------------------------------------------------------------
# Switch lifecycle (temp home)
# ---------------------------------------------------------------------------


class TestSwitchGateway:
    @pytest.fixture
    def patched_exec(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(gateway, "executable", lambda: "/fake/bin/cswap")

    def test_switch_writes_overlay_not_live(self, temp_home: Path, capsys, patched_exec):
        s = _linux_switcher()
        live = get_credentials_path()
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(LIVE_MARKER, encoding="utf-8")
        s.add_account_from_token(GATEWAY_KEY, email="gw@test.local", base_url="https://gw.example.com")
        s.switch_to("1")
        cfg = _settings(s)
        assert cfg["apiKeyHelper"] == "/fake/bin/cswap gateway-credential 1"
        assert cfg["env"]["ANTHROPIC_BASE_URL"] == "https://gw.example.com"
        # The live credential (the underlying OAuth login) is untouched.
        assert live.read_text(encoding="utf-8") == LIVE_MARKER
        assert s._get_sequence_data()["activeAccountNumber"] == 1
        assert "Switched to" in capsys.readouterr().out

    def test_switch_already_on_noop_human(self, temp_home: Path, capsys, patched_exec):
        s = _linux_switcher()
        s.add_account_from_token(GATEWAY_KEY, base_url="https://gw.example.com")
        s.switch_to("1")
        capsys.readouterr()
        s.switch_to("1")
        out = capsys.readouterr().out
        assert "Already on" in out

    def test_switch_already_on_noop_json(self, temp_home: Path, patched_exec):
        s = _linux_switcher()
        s.add_account_from_token(GATEWAY_KEY, base_url="https://gw.example.com")
        s.switch_to("1")
        result = s.switch_to("1", json_output=True)
        assert result["switched"] is False
        assert result["reason"] == "already-active"
        assert result["to"]["number"] == 1

    def test_switch_json_result(self, temp_home: Path, patched_exec):
        s = _linux_switcher()
        s.add_account_from_token(GATEWAY_KEY, email="gw@test.local", base_url="https://gw.example.com")
        result = s.switch_to("1", json_output=True)
        assert result["switched"] is True
        assert result["to"]["number"] == 1
        assert _settings(s)["apiKeyHelper"].endswith("1")

    def test_gateway_credential_emits_key(self, temp_home: Path):
        s = _linux_switcher()
        s.add_account_from_token(GATEWAY_KEY, email="gw@test.local", base_url="https://gw.example.com")
        assert s.gateway_credential("1") == GATEWAY_KEY
        with pytest.raises(SwitchError):
            s.gateway_credential("99")


# ---------------------------------------------------------------------------
# Remove / teardown
# ---------------------------------------------------------------------------


class TestTeardown:
    @pytest.fixture
    def patched_exec(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(gateway, "executable", lambda: "/fake/bin/cswap")

    def test_remove_active_gateway_deactivates(self, temp_home: Path, patched_exec):
        s = _linux_switcher()
        s.add_account_from_token(GATEWAY_KEY, email="gw@test.local", base_url="https://gw.example.com")
        s.switch_to("1")
        assert _settings(s)["apiKeyHelper"].endswith("1")
        s.remove_account("1", assume_yes=True)
        cfg = _settings(s)
        assert "apiKeyHelper" not in cfg
        assert cfg["env"]["ANTHROPIC_BASE_URL"] == gateway.FIRST_PARTY_URL
        assert s._get_sequence_data()["sequence"] == []

    def test_refusal_restores_settings(self, temp_home: Path):
        # Refusing activation (auth override) must leave settings as it found it.
        s = _linux_switcher()
        p = s.gateway_settings_path()
        p.write_text(json.dumps({"env": {"ANTHROPIC_API_KEY": "x"}}), encoding="utf-8")
        snap = p.read_text(encoding="utf-8")
        s.add_account_from_token(GATEWAY_KEY, base_url="https://gw.example.com")
        with pytest.raises(SwitchError):
            s.switch_to("1")
        assert p.read_text(encoding="utf-8") == snap


# ---------------------------------------------------------------------------
# CLI argv → dispatch (subcommand + modifier wiring)
# ---------------------------------------------------------------------------


class TestCliGatewayDispatch:
    def test_argv_forwards_gateway_options(self, temp_home: Path, capsys):
        from claude_swap import cli
        from claude_swap.switcher import ClaudeAccountSwitcher

        with patch.object(
            sys, "argv",
            ["cswap", "add-token", "-", "--base-url", "https://gw.example.com",
             "--auth", "x-api-key", "--alias", "gw",
             "--model", "qwen3.8-27b", "--small-model", "sovereign-mini"],
        ), patch.object(
            ClaudeAccountSwitcher, "add_account_from_token",
        ) as mock_add, patch(
            "claude_swap.update_check.check_for_update", return_value=None,
        ):
            cli.main()

        mock_add.assert_called_once_with(
            token="-", email=None, slot=None,
            base_url="https://gw.example.com", auth="x-api-key", alias="gw",
            model="qwen3.8-27b", small_fast_model="sovereign-mini",
        )

    def test_base_url_requires_add_token(self, capsys):
        from claude_swap import cli

        with patch.object(
            sys, "argv", ["cswap", "list", "--base-url", "https://gw.example.com"],
        ):
            with pytest.raises(SystemExit) as excinfo:
                cli.main()
        assert excinfo.value.code == 2
        assert "--base-url can only be used with 'add-token'" in capsys.readouterr().err
