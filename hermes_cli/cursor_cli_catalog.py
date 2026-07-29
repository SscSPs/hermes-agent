"""Cursor CLI model discovery and model-id normalization for ``cursor-acp``."""

from __future__ import annotations

import shutil
import subprocess
from typing import Literal, Optional

from hermes_cli.external_process_acp import resolve_external_process_command_and_args

CURSOR_ACP_PLACEHOLDER_MODELS = frozenset({"cursor-acp"})

_DEFAULT_CURATED_CURSOR_MODELS = (
    "composer-2.5",
    "auto",
    "default",
    "gpt-5.2",
    "claude-sonnet-4.6",
)


def _list_models_args_from_acp_args(acp_args: list[str]) -> list[str]:
    if acp_args and acp_args[-1] == "acp":
        return acp_args[:-1] + ["--list-models"]
    return ["agent", "--list-models"]


def resolve_cursor_cli_invocation(*, mode: Literal["acp", "list-models"]) -> list[str]:
    """Build argv for Cursor ACP or ``--list-models`` using the same CLI config."""
    command, acp_args = resolve_external_process_command_and_args("cursor-acp")
    resolved = shutil.which(command) or command
    if mode == "acp":
        return [resolved, *acp_args]
    return [resolved, *_list_models_args_from_acp_args(acp_args)]


def resolve_cursor_acp_command() -> str:
    return resolve_cursor_cli_invocation(mode="acp")[0]


def resolve_cursor_acp_args() -> list[str]:
    return resolve_cursor_cli_invocation(mode="acp")[1:]


def _parse_list_models_output(text: str) -> list[str]:
    models: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower() == "available models":
            continue
        model_id = line.split(" - ", 1)[0].strip() if " - " in line else line.split()[0]
        if model_id:
            models.append(model_id)
    return models


def fetch_cursor_agent_model_ids(*, timeout: float = 15.0) -> Optional[list[str]]:
    """Return model ids from the local Cursor CLI ``--list-models`` output."""
    argv = resolve_cursor_cli_invocation(mode="list-models")
    try:
        from hermes_cli._subprocess_compat import windows_hide_flags

        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=windows_hide_flags(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    parsed = _parse_list_models_output(completed.stdout or "")
    return parsed or None


def normalize_cursor_model_id(
    model_id: Optional[str],
    *,
    known_ids: Optional[list[str]] = None,
) -> Optional[str]:
    """Map legacy Hermes placeholder ids to a real Cursor CLI model name."""
    if not model_id:
        return None
    stripped = str(model_id).strip()
    if not stripped:
        return None
    if stripped.lower() not in CURSOR_ACP_PLACEHOLDER_MODELS:
        return stripped
    catalog = list(known_ids or fetch_cursor_agent_model_ids() or _DEFAULT_CURATED_CURSOR_MODELS)
    catalog_set = set(catalog)
    for preferred in ("composer-2.5", "auto", "default"):
        if preferred in catalog_set:
            return preferred
    return catalog[0] if catalog else "composer-2.5"
