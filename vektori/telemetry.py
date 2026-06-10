"""Anonymous usage telemetry for Vektori CLI.

Events are fire-and-forget (daemon thread, 3s timeout).
Skipped entirely when:
  - VEKTORI_NO_TELEMETRY=1  (or DO_NOT_TRACK=1)
  - running inside a known CI/CD environment
  - the PostHog API key is still the placeholder

Replace _POSTHOG_API_KEY with your project's key from PostHog → Project Settings → API Keys.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

_POSTHOG_API_KEY = "phc_REPLACE_WITH_YOUR_KEY"
_POSTHOG_HOST = "https://us.i.posthog.com"
_CONFIG_PATH = Path.home() / ".vektori" / "config.json"

_NOTICE = (
    "[vektori] Anonymous usage stats are collected to improve the project. "
    "Opt out: VEKTORI_NO_TELEMETRY=1  "
    "Details: https://github.com/vektori-ai/vektori#telemetry"
)

# Known CI/CD environment variables — if any are set we skip telemetry entirely.
_CI_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "CIRCLECI",
    "TRAVIS",
    "JENKINS_URL",
    "BUILDKITE",
    "DRONE",
    "GITLAB_CI",
    "TF_BUILD",
    "BITBUCKET_BUILD_NUMBER",
    "TEAMCITY_VERSION",
)


def _is_ci() -> bool:
    return any(os.environ.get(v) for v in _CI_VARS)


def _is_disabled() -> bool:
    return bool(
        os.environ.get("VEKTORI_NO_TELEMETRY")
        or os.environ.get("DO_NOT_TRACK")
        or _POSTHOG_API_KEY == "phc_REPLACE_WITH_YOUR_KEY"
    )


def _load_cfg() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_cfg(data: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _get_or_create_install_id() -> str | None:
    try:
        cfg = _load_cfg()
        if "install_id" not in cfg:
            cfg["install_id"] = str(uuid.uuid4())
            _save_cfg(cfg)
        return cfg["install_id"]
    except Exception:
        return None


def _maybe_show_notice() -> None:
    try:
        cfg = _load_cfg()
        if not cfg.get("telemetry_notice_shown"):
            import typer
            typer.echo(_NOTICE, err=True)
            cfg["telemetry_notice_shown"] = True
            _save_cfg(cfg)
    except Exception:
        pass


def _pkg_version() -> str:
    try:
        from vektori._version import version
        return version
    except Exception:
        return "unknown"


def capture(event: str, properties: dict[str, Any] | None = None) -> None:
    """Fire a telemetry event asynchronously. Never raises."""
    if _is_disabled() or _is_ci():
        return

    _maybe_show_notice()
    install_id = _get_or_create_install_id()
    if not install_id:
        return

    payload = {
        "api_key": _POSTHOG_API_KEY,
        "event": event,
        "distinct_id": install_id,
        "properties": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.system().lower(),
            "pkg_version": _pkg_version(),
            **(properties or {}),
        },
    }

    def _fire() -> None:
        try:
            import httpx
            with httpx.Client(timeout=3.0) as client:
                client.post(f"{_POSTHOG_HOST}/capture/", json=payload)
        except Exception:
            pass

    threading.Thread(target=_fire, daemon=True).start()


def provider(model: str) -> str:
    """Extract provider prefix from a model string, e.g. 'litellm:groq/...' → 'litellm'."""
    return model.split(":")[0] if ":" in model else model
