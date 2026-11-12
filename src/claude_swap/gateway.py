"""Gateway account support: a custom Anthropic-compatible endpoint + API key.

A gateway is its own account ``kind``. Unlike an OAuth seat (which cswap
activates by swapping the live credential store) or a managed API-key seat
(which swaps the live credential to a raw ``sk-ant-…`` key), activating a
gateway does **not** touch ``.credentials.json``: the saved Claude login stays
in place and Claude Code is pointed at the gateway through an overlay on its
user-level ``settings.json``.

Activation writes to ``settings.json``:

- ``apiKeyHelper`` — a command that prints the gateway key. Claude Code runs it
  and uses its output as the API key (the key never lands in ``settings.json``;
  it stays in cswap's own credential store).
- ``env.ANTHROPIC_BASE_URL`` — the gateway's base URL.
- the model selection: the top-level ``model`` is pointed at the gateway's
  main model id (the user's previous value is stashed and restored on
  deactivation, and in-session ``/model`` keeps working), while
  ``env.ANTHROPIC_SMALL_FAST_MODEL`` / ``env.CLAUDE_CODE_SUBAGENT_MODEL`` are
  pinned to the gateway's small/fast model so background work also targets a
  model the gateway has.

Claude Code live-reloads ``settings.json`` (verified against Claude Code
2.1.303 in the upstream gateway work), so no session restart is needed on
Linux/macOS/Windows.

Deactivation (returning to a regular seat) removes ``apiKeyHelper`` and sets
``ANTHROPIC_BASE_URL`` **explicitly** back to the first-party URL — deleting an
environment variable does not unset it in an already-running session.

Only the fields a focused build needs are kept here: URL validation, the
``settings.json`` overlay, the credential helper, and slot bookkeeping. The
upstream live-probe / usage-adapter / auto-fallback machinery is intentionally
out of scope.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

FIRST_PARTY_URL = "https://api.anthropic.com"

#: Record fields (besides ``kind``/``email``/identity) that describe a gateway.
RECORD_FIELDS = ("baseUrl", "auth", "model", "smallFastModel")

#: settings.json model knobs a gateway seat may adjust. Claude Code sends
#: whatever model id it is configured with verbatim in each request, and a
#: gateway usually only knows its OWN model names. The MAIN model is carried
#: in the top-level "model" setting (kept in sync with what Claude Code itself
#: uses, so in-session /model selection keeps working); the env vars below
#: cover the knobs that have no top-level setting. All stashed on activation
#: and restored on deactivation.
MODEL_ENV = "ANTHROPIC_MODEL"
SMALL_FAST_MODEL_ENV = "ANTHROPIC_SMALL_FAST_MODEL"
SUBAGENT_MODEL_ENV = "CLAUDE_CODE_SUBAGENT_MODEL"

#: Top-level settings.json key where the user's own "model" selection is
#: stashed while a gateway overlay is active.
PREV_MODEL_KEY = "_cswapGatewayPrevModel"


def _prev_env_key(name: str) -> str:
    return f"_CSWAP_PREV_{name}"

#: Settings environment variables that would double-supply an auth axis and
#: conflict with cswap's ``apiKeyHelper`` overlay. Their presence means the user
#: configured auth manually, so activating a gateway must refuse rather than
#: silently clobber it.
AUTH_OVERRIDE_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
)

# Both installed entry points own the helper command.
_HELPER_PROGRAMS = ("cswap", "claude-swap", "cswap.exe", "claude-swap.exe")


def validate_url(value: str, *, base: bool = False) -> str:
    """Validate and normalize a gateway URL without echoing a credential.

    HTTPS is required (plain HTTP is tolerated only for ``localhost`` /
    ``127.0.0.1``); embedded credentials, a query string, or a fragment are
    rejected. Trailing slashes and a trailing ``/v1`` are normalized away so
    the stored base URL composes cleanly with Claude Code's ``/v1/…`` paths.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A gateway needs a base URL")
    try:
        parts = urlsplit(value.strip())
        if (
            parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or not parts.hostname
            or parts.port == 0
            or (
                parts.scheme != "https"
                and not (
                    parts.scheme == "http"
                    and parts.hostname in ("127.0.0.1", "localhost")
                )
            )
        ):
            raise ValueError
        path = parts.path.rstrip("/")
        if base and path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    except (ValueError, TypeError, AttributeError):
        raise ValueError(
            "Gateway URL must use HTTPS (HTTP is allowed on localhost or "
            "127.0.0.1), without credentials, query, or fragment"
        ) from None


def account_fields(
    base_url: str,
    auth: str = "bearer",
    model: str | None = None,
    small_fast_model: str | None = None,
) -> dict:
    """Build the gateway record fields (``baseUrl`` + ``auth`` [+ models]).

    ``auth`` selects how Claude Code signs requests: ``bearer`` (an
    ``Authorization: Bearer <key>`` header) or ``x-api-key`` (an ``x-api-key``
    header). Both are honored by common Anthropic-compatible gateways.

    ``model`` / ``small_fast_model`` are the gateway's own model ids, written
    to ``ANTHROPIC_MODEL`` / ``ANTHROPIC_SMALL_FAST_MODEL`` on activation so a
    request carries a name the gateway actually knows. The small/fast id
    defaults to the main model. Omitting them leaves the user's Claude Code
    model selection untouched.
    """
    if auth not in ("bearer", "x-api-key"):
        raise ValueError("Gateway auth must be bearer or x-api-key")
    fields: dict = {
        "kind": "gateway",
        "baseUrl": validate_url(base_url, base=True),
        "auth": auth,
    }
    if model is not None:
        if not model.strip():
            raise ValueError("Gateway model must be a non-empty model id")
        fields["model"] = model.strip()
    if small_fast_model is not None:
        if not small_fast_model.strip():
            raise ValueError("Gateway small/fast model must be a non-empty model id")
        fields["smallFastModel"] = small_fast_model.strip()
    if "model" in fields and "smallFastModel" not in fields:
        fields["smallFastModel"] = fields["model"]
    return fields


def executable() -> str:
    """Absolute path to the ``cswap`` executable used by ``apiKeyHelper``.

    Prefer the PATH-resolved launcher: on Windows the uv console-script sets
    ``sys.argv[0]`` to the extension-less path (``…\\bin\\cswap``), which is not
    itself an executable — ``os.path.abspath`` returns it unchanged and Claude
    Code's ``CreateProcess`` fails with ``ERROR_INVALID_NAME``. ``shutil.which``
    yields the real ``cswap.exe``. ``argv[0]`` is kept only as a last resort,
    and only when it is an absolute path to an existing file.
    """
    found = shutil.which("cswap") or shutil.which("claude-swap")
    if found:
        return found
    argv0 = sys.argv[0]
    if (
        argv0
        and Path(argv0).name.lower() in _HELPER_PROGRAMS
        and os.path.isabs(argv0)
        and os.path.isfile(argv0)
    ):
        return os.path.abspath(argv0)
    raise ValueError("Cannot locate the cswap executable for gateway-credential")


def helper(slot: str) -> str:
    """The ``apiKeyHelper`` command that emits the key for ``slot``.

    Quoting follows the shell Claude Code runs the helper under: POSIX
    ``sh -c`` (single-quotes via ``shlex``) on Linux/macOS, double-quotes on
    Windows ``cmd /c``. Tokens without whitespace are left unquoted, so a normal
    install yields a plain ``C:\\…\\cswap.exe gateway-credential 3``.
    """
    cmd = [executable(), "gateway-credential", str(int(slot))]
    if os.name == "nt":
        return " ".join(f'"{tok}"' if " " in tok else tok for tok in cmd)
    return shlex.join(cmd)


def owned_slot(value: object) -> str | None:
    """Return the slot the ``apiKeyHelper`` string at ``value`` belongs to.

    ``None`` for anything that is not a cswap gateway helper — that is what
    lets the switcher tell a cswap-owned overlay from a foreign one.

    Parsing tries POSIX mode first (the form cswap writes on Linux/macOS),
    then non-POSIX: POSIX ``shlex`` treats backslashes as escape characters,
    so a Windows helper (``C:\\Users\\…\\cswap.EXE gateway-credential 3``) is
    mangled in that mode and would make our OWN overlay read as foreign on
    the next switch — the non-POSIX re-parse is what recovers it.
    """
    if not isinstance(value, str):
        return None
    for split in (
        lambda: shlex.split(value),
        lambda: shlex.split(value, posix=False),
    ):
        try:
            parts = split()
        except ValueError:
            continue
        if (
            len(parts) == 3
            # Case-insensitive: Windows paths are, and installers (uv) may
            # create ``cswap.EXE`` vs ``cswap.exe`` — a helper we wrote
            # ourselves must never read as foreign.
            and Path(parts[0]).name.lower() in _HELPER_PROGRAMS
            and parts[1] == "gateway-credential"
            and parts[2].isdigit()
            and int(parts[2]) > 0
        ):
            return str(int(parts[2]))
    return None


def read_settings(path: Path) -> dict:
    """Read ``settings.json`` as a dict, or ``{}`` when absent/unreadable."""
    if not path.exists():
        return {}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not isinstance(result.get("env", {}), dict):
            raise ValueError
        return result
    except (ValueError, OSError):
        raise ValueError(
            "Cannot read Claude settings.json; repair it before switching a gateway"
        ) from None


def active_slot(path: Path, accounts: dict) -> str | None:
    """The slot currently pointed at by the settings overlay, or ``None``.

    ``accounts`` maps slot number → its sequence record. Only counts as active
    when the helper names a real gateway slot **and** the base URL still matches
    (a stale helper for a removed/changed slot is not an active gateway).
    """
    try:
        settings = read_settings(path)
    except ValueError:
        return None
    slot = owned_slot(settings.get("apiKeyHelper"))
    record = accounts.get(slot, {})
    if (
        record.get("kind") == "gateway"
        and settings.get("env", {}).get("ANTHROPIC_BASE_URL") == record.get("baseUrl")
    ):
        return slot
    return None


def atomic_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + rename + fsync)."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".cswap-settings-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if path.read_bytes() != data:
            raise ValueError("Claude settings changed while switching; retry the switch")
    finally:
        if os.path.exists(name):
            os.unlink(name)


def restore_settings(path: Path, previous: bytes | None) -> None:
    """Restore a previously captured ``settings.json`` (or remove it)."""
    if previous is None:
        path.resolve().unlink(missing_ok=True)
    else:
        atomic_bytes(path, previous)


def refuse_auth_overrides(settings: dict, path: Path) -> None:
    """Refuse to activate while a foreign auth override is set in settings."""
    if any(settings.get("env", {}).get(name) for name in AUTH_OVERRIDE_ENV_VARS):
        raise ValueError(
            f"Claude settings file {path} contains authentication overrides; "
            "remove them before activating a gateway"
        )


def write_settings(
    path: Path,
    slot: str | None,
    record: dict | None = None,
    *,
    remove_last: bool = False,
) -> None:
    """Point the overlay at ``slot``'s gateway, or (when ``slot`` is None) undo it.

    ``remove_last`` is the teardown used when removing the final gateway: it
    drops only the first-party URL cswap itself left behind on deactivation —
    it never touches a gateway the user has since configured by hand.
    """
    settings = read_settings(path)
    if remove_last:
        env = settings.get("env", {})
        if env.get("ANTHROPIC_BASE_URL") == FIRST_PARTY_URL:
            env.pop("ANTHROPIC_BASE_URL")
        if not env:
            settings.pop("env", None)
        atomic_bytes(path, (json.dumps(settings, indent=2) + "\n").encode())
        return

    old_helper = settings.get("apiKeyHelper")
    if old_helper is not None and owned_slot(old_helper) is None:
        raise ValueError(
            "Claude settings.json has a foreign apiKeyHelper; refusing to replace it"
        )

    env = settings.setdefault("env", {})
    if slot is not None:
        refuse_auth_overrides(settings, path)
        settings["apiKeyHelper"] = helper(slot)
        env["ANTHROPIC_BASE_URL"] = record["baseUrl"]
        # Point the seat's model selection at the gateway's own ids (Claude
        # Code sends whatever model it is configured with verbatim; a gateway
        # usually only knows names like `qwen3.8-27b`, not `sonnet`).
        small = record.get("smallFastModel")
        # Main model: top-level "model" (the setting Claude Code itself uses
        # and persists) — NOT env ANTHROPIC_MODEL, which outranks it and would
        # defeat in-session /model selection. Stash the user's selection first
        # so deactivation can restore it.
        if record.get("model"):
            if PREV_MODEL_KEY not in settings:
                settings[PREV_MODEL_KEY] = settings.get("model")
            settings["model"] = record["model"]
        # env ANTHROPIC_MODEL must never be SET by us: it outranks the
        # top-level "model" and would defeat in-session /model selection.
        # Clear it for the overlay's duration, stashing it only when it is a
        # genuine user pin (not stale residue equal to the gateway model).
        if env.get(MODEL_ENV):
            if env[MODEL_ENV] != record.get("model"):
                env[_prev_env_key(MODEL_ENV)] = env[MODEL_ENV]
            env.pop(MODEL_ENV)
        # Env-model knobs with no top-level equivalent: stash any pre-existing
        # value that differs from the override, then set. A value equal to
        # the override is stale residue from this same gateway and is simply
        # dropped — deactivation discards whatever is not stashed, which is
        # exactly the first-party-safe outcome.
        env_overrides = {}
        if small:
            env_overrides[SMALL_FAST_MODEL_ENV] = small
            env_overrides[SUBAGENT_MODEL_ENV] = small
        for name, value in env_overrides.items():
            if env.get(name):
                if env[name] != value:
                    env[_prev_env_key(name)] = env[name]
                env.pop(name)
            env[name] = value
    else:
        settings.pop("apiKeyHelper", None)
        env["ANTHROPIC_BASE_URL"] = FIRST_PARTY_URL
        # Restore the stashed model state, discarding the rest: whatever model
        # value a gateway session left behind (env var or top-level) would
        # point a first-party seat at a model Anthropic does not have.
        for name in (MODEL_ENV, SMALL_FAST_MODEL_ENV, SUBAGENT_MODEL_ENV):
            prev = env.pop(_prev_env_key(name), None)
            if prev:
                env[name] = prev
            else:
                env.pop(name, None)
        prev_model = settings.pop(PREV_MODEL_KEY, "__absent__")
        if prev_model == "__absent__":
            pass
        elif isinstance(prev_model, str) and prev_model.strip():
            settings["model"] = prev_model
        elif "model" in settings:
            # The user had no top-level model before the gateway: leave them
            # with none (Claude Code's default), not the gateway's model id.
            settings.pop("model", None)
        if not env:
            settings.pop("env", None)
    atomic_bytes(path, (json.dumps(settings, indent=2) + "\n").encode())


def tag(record: dict) -> str:
    """Short display tag, e.g. ``gateway · gateway.example · bearer``."""
    try:
        host = urlsplit(record["baseUrl"]).netloc
    except (KeyError, ValueError):
        host = record.get("baseUrl", "")
    return f"gateway · {host} · {record.get('auth', 'bearer')}"
