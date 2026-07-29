"""OpenAI-compatible shim that forwards Hermes requests to `cursor agent acp`.

This adapter lets Hermes treat the Cursor CLI's ACP server as a chat-style
backend.  Each request starts a short-lived ACP session, sends the formatted
conversation as a single prompt, collects text chunks, and converts the result
back into the minimal shape Hermes expects from an OpenAI client.

Cursor CLI must be installed and authenticated::

    curl https://cursor.com/install -fsS | bash
    cursor agent login

Models available through this provider include Cursor's own **Composer 2.5**
as well as any models your Cursor subscription grants access to (Claude,
GPT-5, Gemini, etc.).  Set the ``model`` parameter in your config to the
desired model name (e.g. ``composer-2.5``).
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.file_safety import get_read_block_error, get_write_denied_error
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://cursor"
_DEFAULT_TIMEOUT_SECONDS = 900.0
from hermes_cli.cursor_cli_catalog import (
    CURSOR_ACP_PLACEHOLDER_MODELS,
    resolve_cursor_acp_args,
    resolve_cursor_acp_command,
)

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(
    r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}",
    re.DOTALL,
)


def _resolve_command() -> str:
    """Return the Cursor CLI command path."""
    return resolve_cursor_acp_command()


def _resolve_args() -> list[str]:
    """Return arguments for the Cursor ACP server command."""
    return resolve_cursor_acp_args()


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()
        if resolved:
            return resolved
    except Exception:
        pass

    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    """Build environment for the Cursor ACP subprocess.

    Cursor needs its own auth tokens (CURSOR_API_KEY or stored OAuth tokens)
    rather than generic LLM provider credentials.
    """
    env = hermes_subprocess_env(inherit_credentials=False)
    home = _resolve_home_dir()
    env["HOME"] = home
    # Cursor stores auth state under ~/.cursor/ — ensure HOME points there.
    from hermes_constants import apply_subprocess_home_env

    apply_subprocess_home_env(env)
    return env


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _permission_denied(message_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "cancelled",
            }
        },
    }


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are the language-model backend for Hermes. You are NOT the executing agent.",
        "This ACP session cannot run shell, terminal, or filesystem actions — permission "
        "requests are always cancelled.",
        "Never use ACP-native tools or ask to run commands yourself.",
        "When you need to act, emit Hermes tool calls ONLY as "
        "<tool_call>{...}</tool_call> blocks with JSON in OpenAI function-call shape "
        "(id, type, function.name, function.arguments as a JSON string).",
        "Hermes executes those tools (terminal, read_file, web_search, etc.) and returns results.",
        "If no tool is needed, answer in plain text.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_openai_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot ACP response into OpenAI-style stream chunks."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )

    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=delta,
                finish_reason=choice.finish_reason,
            )
        ],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    return [data_chunk, usage_chunk]


def _extract_tool_calls_from_text(text: str) -> tuple[list[ChatCompletionMessageToolCall], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"cursor_acp_call_{len(extracted)+1}"

        extracted.append(
            _build_openai_tool_call(
                call_id=call_id,
                name=fn_name.strip(),
                arguments=fn_args,
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1)
        _try_add_tool_call(raw)
        consumed_spans.append((m.start(), m.end()))

    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            raw = m.group(0)
            _try_add_tool_call(raw)
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned


def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


class _ACPChatCompletions:
    def __init__(self, client: "CursorACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CursorACPClient"):
        self.completions = _ACPChatCompletions(client)


class CursorACPClient:
    """Minimal OpenAI-client-compatible facade for Cursor ACP."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "cursor-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = list(acp_args or args or _resolve_args())
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._session_id: str | None = None
        self._session_cmd_key: tuple[str, ...] | None = None
        self._session_cwd: str | None = None
        self._process_initialized = False
        self._rpc_id = 0
        self._inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._reader_threads: list[threading.Thread] = []

    def _build_acp_command(self, model_hint: str | None) -> list[str]:
        cmd = [self._acp_command, *self._acp_args]
        effective_hint = (model_hint or "").strip()
        if effective_hint.lower() in CURSOR_ACP_PLACEHOLDER_MODELS:
            effective_hint = ""
        if effective_hint and "--model" not in cmd:
            cmd.extend(["--model", effective_hint])
        return cmd

    def _teardown_persistent_locked(self) -> None:
        proc = self._active_process
        self._active_process = None
        self._session_id = None
        self._session_cmd_key = None
        self._session_cwd = None
        self._process_initialized = False
        self._reader_threads = []
        while True:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                break
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def close(self) -> None:
        with self._active_process_lock:
            self._teardown_persistent_locked()
        self.is_closed = True

    def _start_reader_threads(self, proc: subprocess.Popen[str]) -> None:
        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                try:
                    self._inbox.put(json.loads(line))
                except Exception:
                    self._inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                self._stderr_tail.append(line.rstrip("\n"))

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()
        self._reader_threads = [out_thread, err_thread]

    def _spawn_persistent_process(self, cmd: list[str]) -> subprocess.Popen[str]:
        from hermes_cli._subprocess_compat import windows_hide_flags

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._acp_cwd,
                env=_build_subprocess_env(),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Cursor ACP command '{cmd[0]}'. "
                "Install Cursor CLI (curl https://cursor.com/install -fsS | bash) "
                "and authenticate (cursor agent login), or set HERMES_CURSOR_ACP_COMMAND/CURSOR_CLI_PATH."
            ) from exc
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Cursor ACP process did not expose stdin/stdout pipes.")
        return proc

    def _jsonrpc_request(
        self,
        proc: subprocess.Popen[str],
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
        text_parts: list[str] | None = None,
        reasoning_parts: list[str] | None = None,
    ) -> Any:
        self._rpc_id += 1
        request_id = self._rpc_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        if proc.stdin is None:
            raise RuntimeError("Cursor ACP process stdin is closed.")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                msg = self._inbox.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._handle_server_message(
                msg,
                process=proc,
                cwd=self._acp_cwd,
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            ):
                continue

            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg.get("error") or {}
                raise RuntimeError(
                    f"Cursor ACP {method} failed: {err.get('message') or err}"
                )
            return msg.get("result")

        stderr_text = "\n".join(self._stderr_tail).strip()
        if proc.poll() is not None and stderr_text:
            lower = stderr_text.lower()
            if "not authenticated" in lower or "login" in lower:
                raise RuntimeError(
                    "Cursor CLI is not authenticated.\n\n"
                    "Run: cursor agent login\n\n"
                    f"Original error:\n{stderr_text}"
                )
            raise RuntimeError(f"Cursor ACP process exited early: {stderr_text}")
        raise TimeoutError(f"Timed out waiting for Cursor ACP response to {method}.")

    def _ensure_persistent_process(
        self, *, model_hint: str | None, timeout_seconds: float
    ) -> subprocess.Popen[str]:
        cmd = self._build_acp_command(model_hint)
        cmd_key = tuple(cmd)
        cwd = self._acp_cwd
        proc = self._active_process
        if (
            proc is not None
            and proc.poll() is None
            and self._process_initialized
            and self._session_cmd_key == cmd_key
            and self._session_cwd == cwd
        ):
            return proc

        self._teardown_persistent_locked()
        proc = self._spawn_persistent_process(cmd)
        self._active_process = proc
        self._session_cmd_key = cmd_key
        self._session_cwd = cwd
        self.is_closed = False
        self._start_reader_threads(proc)

        self._jsonrpc_request(
            proc,
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {
                    "name": "hermes-agent",
                    "title": "Hermes Agent",
                    "version": "0.0.0",
                },
            },
            timeout_seconds=timeout_seconds,
        )
        self._process_initialized = True
        return proc

    def _new_acp_session(
        self, proc: subprocess.Popen[str], *, timeout_seconds: float
    ) -> str:
        session = (
            self._jsonrpc_request(
                proc,
                "session/new",
                {
                    "cwd": self._acp_cwd,
                    "mcpServers": [],
                },
                timeout_seconds=timeout_seconds,
            )
            or {}
        )
        session_id = str(session.get("sessionId") or "").strip()
        if not session_id:
            raise RuntimeError("Cursor ACP did not return a sessionId.")
        self._session_id = session_id
        return session_id

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text = self._run_prompt(
            prompt_text,
            model_hint=model,
            timeout_seconds=_effective_timeout,
        )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "cursor-acp",
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    def _run_prompt(
        self, prompt_text: str, *, model_hint: str | None, timeout_seconds: float
    ) -> tuple[str, str]:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with self._active_process_lock:
                    proc = self._ensure_persistent_process(
                        model_hint=model_hint,
                        timeout_seconds=timeout_seconds,
                    )
                    session_id = self._new_acp_session(
                        proc, timeout_seconds=timeout_seconds
                    )
                    text_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    self._jsonrpc_request(
                        proc,
                        "session/prompt",
                        {
                            "sessionId": session_id,
                            "prompt": [
                                {
                                    "type": "text",
                                    "text": prompt_text,
                                }
                            ],
                        },
                        timeout_seconds=timeout_seconds,
                        text_parts=text_parts,
                        reasoning_parts=reasoning_parts,
                    )
                return "".join(text_parts), "".join(reasoning_parts)
            except Exception as exc:
                last_error = exc
                with self._active_process_lock:
                    self._teardown_persistent_locked()
                if attempt == 0:
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Cursor ACP prompt failed without a response.")

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
                reasoning_parts.append(chunk_text)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = _permission_denied(message_id)
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    content = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                denied = get_write_denied_error(str(path))
                if denied:
                    raise PermissionError(denied)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""), encoding="utf-8")
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": None,
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True
