"""Tests for external-process ACP CLI resolution."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.cursor_cli_catalog import resolve_cursor_cli_invocation
from hermes_cli.external_process_acp import (
    resolve_external_process_command_and_args,
)


def test_cursor_acp_list_models_uses_same_command_as_acp(monkeypatch):
    monkeypatch.delenv("HERMES_CURSOR_ACP_ARGS", raising=False)
    monkeypatch.delenv("HERMES_CURSOR_ACP_COMMAND", raising=False)
    monkeypatch.delenv("CURSOR_CLI_PATH", raising=False)

    acp_argv = resolve_cursor_cli_invocation(mode="acp")
    list_argv = resolve_cursor_cli_invocation(mode="list-models")

    assert acp_argv[0] == list_argv[0]
    assert acp_argv[1:] == ["agent", "acp"]
    assert list_argv[1:] == ["agent", "--list-models"]


def test_cursor_custom_args_derive_list_models_from_acp_args(monkeypatch):
    monkeypatch.setenv("HERMES_CURSOR_ACP_ARGS", "agent acp")
    monkeypatch.delenv("HERMES_CURSOR_ACP_COMMAND", raising=False)

    with patch("shutil.which", return_value="/usr/bin/cursor"):
        list_argv = resolve_cursor_cli_invocation(mode="list-models")

    assert list_argv == ["/usr/bin/cursor", "agent", "--list-models"]


def test_copilot_acp_default_args():
    command, args = resolve_external_process_command_and_args("copilot-acp")
    assert command == "copilot"
    assert args == ["--acp", "--stdio"]


def test_unknown_external_process_provider():
    with pytest.raises(KeyError):
        resolve_external_process_command_and_args("not-a-provider")
