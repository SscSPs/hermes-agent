"""CLI resolution for external-process ACP providers (Copilot CLI, Cursor CLI)."""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from typing import Optional

EXTERNAL_PROCESS_ACP_PROVIDER_IDS = frozenset({"copilot-acp", "cursor-acp"})


@dataclass(frozen=True)
class ExternalProcessACPCliSpec:
    provider_id: str
    placeholder_api_key: str
    default_command: str
    command_env_vars: tuple[str, ...]
    args_env_var: str
    default_args: tuple[str, ...]
    missing_cli_code: str
    missing_cli_install_hint: str


_EXTERNAL_PROCESS_ACP_CLI: dict[str, ExternalProcessACPCliSpec] = {
    "copilot-acp": ExternalProcessACPCliSpec(
        provider_id="copilot-acp",
        placeholder_api_key="copilot-acp",
        default_command="copilot",
        command_env_vars=("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"),
        args_env_var="HERMES_COPILOT_ACP_ARGS",
        default_args=("--acp", "--stdio"),
        missing_cli_code="missing_copilot_cli",
        missing_cli_install_hint=(
            "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
        ),
    ),
    "cursor-acp": ExternalProcessACPCliSpec(
        provider_id="cursor-acp",
        placeholder_api_key="cursor-acp",
        default_command="cursor",
        command_env_vars=("HERMES_CURSOR_ACP_COMMAND", "CURSOR_CLI_PATH"),
        args_env_var="HERMES_CURSOR_ACP_ARGS",
        default_args=("agent", "acp"),
        missing_cli_code="missing_cursor_cli",
        missing_cli_install_hint=(
            "Install Cursor CLI or set HERMES_CURSOR_ACP_COMMAND/CURSOR_CLI_PATH."
        ),
    ),
}


def get_external_process_acp_cli_spec(provider_id: str) -> ExternalProcessACPCliSpec:
    spec = _EXTERNAL_PROCESS_ACP_CLI.get(provider_id)
    if spec is None:
        raise KeyError(f"Unknown external-process ACP provider: {provider_id}")
    return spec


def resolve_external_process_command_and_args(provider_id: str) -> tuple[str, list[str]]:
    """Return configured CLI command name/path and argument vector."""
    spec = get_external_process_acp_cli_spec(provider_id)
    command = ""
    for env_var in spec.command_env_vars:
        command = os.getenv(env_var, "").strip()
        if command:
            break
    if not command:
        command = spec.default_command
    raw_args = os.getenv(spec.args_env_var, "").strip()
    args = list(shlex.split(raw_args)) if raw_args else list(spec.default_args)
    return command, args


def resolve_external_process_command_path(provider_id: str) -> Optional[str]:
    command, _ = resolve_external_process_command_and_args(provider_id)
    return shutil.which(command) if command else None


def external_process_cache_fingerprint_parts(provider_id: str) -> list[str]:
    """Env-derived cache fingerprint fragments for ``provider_models_cache.json``."""
    try:
        spec = get_external_process_acp_cli_spec(provider_id)
    except KeyError:
        return []
    parts: list[str] = []
    for env_var in spec.command_env_vars:
        parts.append(f"{env_var}={os.environ.get(env_var, '')}")
    parts.append(f"{spec.args_env_var}={os.environ.get(spec.args_env_var, '')}")
    return parts
