"""Regressions for the Cursor ACP shim."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from agent.cursor_acp_client import CursorACPClient


class CursorACPPersistentSessionTests(unittest.TestCase):
    def test_second_prompt_reuses_existing_session(self) -> None:
        client = CursorACPClient(acp_cwd="/tmp")
        rpc_calls: list[str] = []

        def fake_jsonrpc(
            _proc,
            method,
            _params,
            *,
            timeout_seconds,
            text_parts=None,
            reasoning_parts=None,
        ):
            rpc_calls.append(method)
            if method == "initialize":
                return {}
            if method == "session/new":
                return {"sessionId": "sess-1"}
            if method == "session/prompt":
                if text_parts is not None:
                    text_parts.append("ok")
                return {}
            return {}

        with patch.object(client, "_jsonrpc_request", side_effect=fake_jsonrpc):
            with patch.object(
                client,
                "_spawn_persistent_process",
                return_value=MagicMock(poll=MagicMock(return_value=None), stdin=MagicMock()),
            ):
                first = client._run_prompt("one", model_hint="composer-2.5", timeout_seconds=30)
                second = client._run_prompt("two", model_hint="composer-2.5", timeout_seconds=30)

        self.assertEqual(first, ("ok", ""))
        self.assertEqual(second, ("ok", ""))
        self.assertEqual(
            rpc_calls,
            [
                "initialize",
                "session/new",
                "session/prompt",
                "session/prompt",
            ],
        )

    def test_close_tears_down_session(self) -> None:
        client = CursorACPClient(acp_cwd="/tmp")
        client._session_id = "sess-1"
        client._active_process = MagicMock()
        client.close()
        self.assertTrue(client.is_closed)
        self.assertIsNone(client._session_id)
        self.assertIsNone(client._active_process)
