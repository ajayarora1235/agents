from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from livekit.agents import llm
from livekit.agents._exceptions import APIError
from livekit.agents.llm.remote_chat_context import RemoteChatContext
from livekit.plugins.openai.realtime.realtime_model import RealtimeSession, _is_fatal_error

pytestmark = pytest.mark.unit


def test_input_transcripts_keep_their_speech_start_time() -> None:
    emitted: list[tuple[str, object]] = []
    session = cast(
        RealtimeSession,
        SimpleNamespace(
            _input_speech_started_at={},
            _remote_chat_ctx=RemoteChatContext(),
            _clear_transcript_accumulator=lambda item_id, content_index: None,
            emit=lambda name, event: emitted.append((name, event)),
        ),
    )

    with patch(
        "livekit.plugins.openai.realtime.realtime_model.time.time",
        side_effect=[100.0, 250.0, 999.0, 999.0],
    ):
        RealtimeSession._handle_input_audio_buffer_speech_started(
            session, SimpleNamespace(item_id="user-1")
        )
        RealtimeSession._handle_input_audio_buffer_speech_started(
            session, SimpleNamespace(item_id="user-2")
        )
        RealtimeSession._handle_conversion_item_input_audio_transcription_completed(
            session,
            SimpleNamespace(
                item_id="user-2",
                content_index=0,
                transcript="second",
                logprobs=None,
            ),
        )
        RealtimeSession._handle_conversion_item_input_audio_transcription_completed(
            session,
            SimpleNamespace(
                item_id="user-1",
                content_index=0,
                transcript="first",
                logprobs=None,
            ),
        )

    transcripts = [
        event for name, event in emitted if name == "input_audio_transcription_completed"
    ]
    assert [(event.item_id, event.created_at) for event in transcripts] == [
        ("user-2", 250.0),
        ("user-1", 100.0),
    ]
    assert session._input_speech_started_at == {}


def test_update_chat_ctx_deletes_empty_remote_items() -> None:
    remote_ctx = RemoteChatContext()
    audio_item = llm.ChatMessage(id="audio_item", role="user", content=[])
    kept_item = llm.ChatMessage(id="assistant_item", role="assistant", content=["kept"])
    remote_ctx.insert(None, audio_item)
    remote_ctx.insert(audio_item.id, kept_item)

    session = cast(RealtimeSession, SimpleNamespace(_remote_chat_ctx=remote_ctx))
    events = RealtimeSession._create_update_chat_ctx_events(
        session,
        llm.ChatContext(items=[kept_item]),
    )

    delete_ids = [
        getattr(event, "item_id", None)
        for event in events
        if getattr(event, "type", None) == "conversation.item.delete"
    ]
    assert delete_ids == ["audio_item"]


# --------------------------------------------------------------------------- #
# fatal error classification: a fatal error must break the recv loop so that
# _main_task stops reconnecting (raised as APIError(retryable=False))
# --------------------------------------------------------------------------- #


def test_is_fatal_error_matches_known_codes() -> None:
    assert _is_fatal_error(SimpleNamespace(code="insufficient_quota"))
    assert _is_fatal_error(SimpleNamespace(code=None, type="invalid_api_key"))
    assert not _is_fatal_error(SimpleNamespace(code="server_error"))
    assert not _is_fatal_error(SimpleNamespace())
    assert not _is_fatal_error(None)


def _handle_error_session(capture: dict[str, object]) -> RealtimeSession:
    return cast(
        RealtimeSession,
        SimpleNamespace(
            _realtime_model=SimpleNamespace(_provider_label="openai"),
            _emit_error=lambda error, recoverable: capture.update(recoverable=recoverable),
        ),
    )


def test_handle_error_raises_on_fatal() -> None:
    # a fatal code is raised (not emitted here): the recv loop re-raises it so
    # _main_task emits it once with recoverable=False and stops reconnecting
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(
        error=SimpleNamespace(message="quota exceeded", code="insufficient_quota")
    )
    with pytest.raises(APIError) as exc_info:
        RealtimeSession._handle_error(session, event)
    assert exc_info.value.retryable is False
    assert captured == {}  # not emitted by the handler; _main_task owns the emit


def test_handle_error_emits_transient_as_recoverable() -> None:
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(error=SimpleNamespace(message="server hiccup", code="server_error"))
    RealtimeSession._handle_error(session, event)
    assert captured["recoverable"] is True


def test_handle_error_ignores_cancellation_failed() -> None:
    captured: dict[str, object] = {}
    event = SimpleNamespace(error=SimpleNamespace(message="Cancellation failed: no response"))
    RealtimeSession._handle_error(_handle_error_session(captured), event)
    assert captured == {}  # early return, nothing emitted


def test_response_done_failed_fatal_raises() -> None:
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(
        response=SimpleNamespace(
            id="resp_1",
            status="failed",
            status_details=SimpleNamespace(
                error=SimpleNamespace(type="insufficient_quota", code="insufficient_quota")
            ),
        )
    )
    with pytest.raises(APIError) as exc_info:
        RealtimeSession._handle_response_done_but_not_complete(session, event)
    assert exc_info.value.retryable is False
    assert captured == {}


def test_response_done_failed_transient_stays_recoverable() -> None:
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(
        response=SimpleNamespace(
            id="resp_1",
            status="failed",
            status_details=SimpleNamespace(
                error=SimpleNamespace(type="invalid_request_error", code="rate_limit_exceeded")
            ),
        )
    )
    RealtimeSession._handle_response_done_but_not_complete(session, event)
    assert captured["recoverable"] is True
