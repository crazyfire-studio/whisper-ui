"""Tests for the centralised logging configuration."""

from __future__ import annotations

import json
import logging

import pytest

from whisper_ui.core.logging_setup import (
    _DEFAULT_REQUEST_ID,
    _DEFAULT_USER_ID,
    JsonFormatter,
    RequestContextFilter,
    _resolve_json,
    collapse_control_chars,
    current_request_id,
    current_user_id,
    mask_username,
    reset_request_context,
    set_request_context,
    setup_logging,
)


@pytest.mark.parametrize(
    ("raw", "masked"),
    [
        ("alice", "a***e"),
        ("bob", "b***b"),
        ("ab", "**"),
        ("x", "**"),
        ("", ""),
    ],
)
def test_mask_username_keeps_only_first_and_last(raw, masked):
    assert mask_username(raw) == masked


def test_mask_username_never_returns_the_full_value():
    secret = "supersecretlogin"
    assert secret not in mask_username(secret)


@pytest.fixture(autouse=True)
def _restore_logging():
    """Reset root logger handlers between tests so dictConfig state does not leak."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)


@pytest.mark.parametrize(
    ("env_value", "expected_level_name"),
    [
        ("DEBUG", "DEBUG"),
        ("INFO", "INFO"),
        ("WARNING", "WARNING"),
        ("ERROR", "ERROR"),
        ("CRITICAL", "CRITICAL"),
        ("info", "INFO"),
        ("  warning  ", "WARNING"),
    ],
)
def test_setup_logging_honours_valid_log_level(monkeypatch, env_value, expected_level_name):
    monkeypatch.setenv("LOG_LEVEL", env_value)

    setup_logging()

    assert logging.getLogger().getEffectiveLevel() == getattr(logging, expected_level_name)


def test_setup_logging_falls_back_to_info_for_invalid_log_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "TRACE")

    setup_logging()

    assert logging.getLogger().getEffectiveLevel() == logging.INFO


def test_setup_logging_defaults_to_info_when_env_unset(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    setup_logging()

    assert logging.getLogger().getEffectiveLevel() == logging.INFO


def test_setup_logging_param_log_level_overrides_env(monkeypatch):
    """A Settings-supplied log_level (from .env, which os.getenv cannot see)
    takes precedence over the process-env fallback."""
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    setup_logging(log_level="DEBUG")

    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG


def test_setup_logging_param_log_json_true_selects_json_formatter(monkeypatch, capsys):
    monkeypatch.delenv("LOG_JSON", raising=False)  # env says text; param must win

    setup_logging(log_json=True)
    logging.getLogger("whisper_ui.test").warning("hello json")

    err = capsys.readouterr().err.strip().splitlines()[-1]
    parsed = json.loads(err)  # would raise if the text formatter were used
    assert parsed["message"] == "hello json"


def test_setup_logging_pins_rq_loggers_to_warning(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    setup_logging()

    assert logging.getLogger("rq").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("rq.worker").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("rq.scheduler").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("uvicorn.access").getEffectiveLevel() == logging.WARNING


def test_rq_worker_logs_keep_reaching_our_handler(monkeypatch):
    """RQ 3 turns propagation off on loggers it gives its own handlers.

    It only adds them when no handler up the hierarchy would catch the record.
    The worker calls setup_logging before ``rq.cli.main``, so the root handler
    is there and ``rq.worker`` records keep going through our formatter.
    """
    from rq.logutils import setup_loghandlers

    monkeypatch.setenv("LOG_LEVEL", "INFO")
    setup_logging()
    worker_logger = logging.getLogger("rq.worker")
    monkeypatch.setattr(worker_logger, "handlers", [])
    monkeypatch.setattr(worker_logger, "propagate", True)

    setup_loghandlers(level="INFO")

    assert worker_logger.handlers == []
    assert worker_logger.propagate is True


def test_setup_logging_is_idempotent(monkeypatch):
    """Calling setup_logging twice must not double-attach handlers."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    setup_logging()
    setup_logging()

    handlers = logging.getLogger().handlers
    assert len(handlers) == 1


def test_setup_logging_emits_records_with_request_context(monkeypatch, capsys):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    setup_logging()
    tokens = set_request_context(request_id="abc12345", user_id="alice")
    try:
        logging.getLogger("whisper_ui.test").info("hello world")
    finally:
        reset_request_context(tokens)

    captured = capsys.readouterr().err
    assert "hello world" in captured
    assert "req=abc12345" in captured
    assert "user=alice" in captured


def test_request_context_filter_defaults_when_unset():
    """A LogRecord processed outside any request must get the default dashes."""
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="msg",
        args=(),
        exc_info=None,
    )

    assert RequestContextFilter().filter(record) is True
    assert record.request_id == _DEFAULT_REQUEST_ID
    assert record.user_id == _DEFAULT_USER_ID


def test_set_and_reset_request_context_round_trip():
    tokens = set_request_context(request_id="r1", user_id="u1")
    try:
        assert current_request_id() == "r1"
        assert current_user_id() == "u1"
    finally:
        reset_request_context(tokens)

    assert current_request_id() == _DEFAULT_REQUEST_ID
    assert current_user_id() == _DEFAULT_USER_ID


def test_request_context_isolated_across_nested_blocks():
    """Nested set/reset must restore the outer value, not the default."""
    outer = set_request_context(request_id="outer", user_id="outer-u")
    try:
        inner = set_request_context(request_id="inner", user_id="inner-u")
        try:
            assert current_request_id() == "inner"
        finally:
            reset_request_context(inner)
        assert current_request_id() == "outer"
    finally:
        reset_request_context(outer)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("  Yes ", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        (None, False),
        ("nope", False),
    ],
)
def test_resolve_json(raw, expected):
    assert _resolve_json(raw) is expected


def test_json_formatter_emits_structured_extra_fields():
    record = logging.LogRecord("w", logging.INFO, "p", 1, "Stage %s finished", ("diarize",), None)
    record.request_id = "req-1"
    record.user_id = "u-1"
    record.event = "stage_finish"
    record.stage = "diarize"
    record.job_id = "abc123"
    record.elapsed_ms = 651060

    payload = json.loads(JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z").format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "w"
    assert payload["request_id"] == "req-1"
    assert payload["user_id"] == "u-1"
    assert payload["message"] == "Stage diarize finished"
    assert payload["event"] == "stage_finish"
    assert payload["stage"] == "diarize"
    assert payload["job_id"] == "abc123"
    assert payload["elapsed_ms"] == 651060


def test_json_formatter_redacts_sensitive_extra_fields():
    """A call site that accidentally passes a token / password / raw URL via
    extra={} must not leak it: sensitive-named fields are redacted in the JSON
    log while ordinary fields pass through."""
    record = logging.LogRecord("w", logging.INFO, "p", 1, "op", (), None)
    record.hf_token = "hf_realsecret"
    record.password = "hunter2"
    record.redis_url = "redis://:pw@host:6379/0"
    record.api_key = "sk-live-123"
    record.job_id = "abc123"  # ordinary field, must survive

    payload = json.loads(JsonFormatter().format(record))

    assert payload["hf_token"] == "***"
    assert payload["password"] == "***"
    assert payload["redis_url"] == "***"
    assert payload["api_key"] == "***"
    assert payload["job_id"] == "abc123"


def test_json_formatter_redaction_matches_whole_words_not_substrings():
    """Boundary-aware matching: sensitive fields (whole-word markers) are
    redacted, but fields that merely *contain* a marker substring
    (curl_command → 'url', tokenizer_name → 'token') are kept."""
    record = logging.LogRecord("w", logging.INFO, "p", 1, "op", (), None)
    # Redacted: whole-word / delimited markers.
    record.source_url = "https://x.com/watch?v=1"
    record.authorization = "Bearer abc"
    record.apikey = "sk-1"
    # Kept: marker only appears as a substring inside another word.
    record.curl_command = "curl https://ok"
    record.curl_exit_code = 0
    record.tokenizer_name = "whisper"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["source_url"] == "***"
    assert payload["authorization"] == "***"
    assert payload["apikey"] == "***"
    assert payload["curl_command"] == "curl https://ok"
    assert payload["curl_exit_code"] == 0
    assert payload["tokenizer_name"] == "whisper"


def test_json_formatter_includes_rendered_exception():
    import sys

    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("w", logging.ERROR, "p", 1, "failed", (), sys.exc_info())

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exc"]


def test_setup_logging_json_mode_outputs_one_json_object_per_line(monkeypatch, capsys):
    monkeypatch.setenv("LOG_JSON", "true")
    setup_logging()

    logging.getLogger("whisper_ui.test").info("hello %s", "world", extra={"event": "x", "elapsed_ms": 5})

    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)  # must be valid JSON
    assert payload["message"] == "hello world"
    assert payload["event"] == "x"
    assert payload["elapsed_ms"] == 5
    assert payload["request_id"] == _DEFAULT_REQUEST_ID  # no request context -> default dash


def test_setup_logging_defaults_to_text_when_log_json_unset(monkeypatch, capsys):
    monkeypatch.delenv("LOG_JSON", raising=False)
    setup_logging()

    logging.getLogger("whisper_ui.test").info("plain line")

    captured = capsys.readouterr().err
    assert "plain line" in captured
    assert "[req=" in captured  # text format, not JSON


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain ascii", "plain ascii"),
        ("第三講 (2026)", "第三講 (2026)"),
        ("Lecture\u00a03", "Lecture\u00a03"),  # NO-BREAK SPACE is ordinary text
        ("a\nb\rc\td", "a b c d"),
        ("a\u2028b\u2029c", "a b c"),
        ("x\u202ey", "x y"),  # RIGHT-TO-LEFT OVERRIDE
        ("\x1b[31mred", " [31mred"),
    ],
)
def test_collapse_control_chars(raw, expected):
    assert collapse_control_chars(raw) == expected


def test_text_log_line_cannot_be_split_by_logged_values(monkeypatch, capsys):
    # A percent-decoded request path is the live case: /x%0A... reaches the
    # access log as a real newline.
    monkeypatch.delenv("LOG_JSON", raising=False)
    setup_logging()

    logging.getLogger("whisper_ui.test").info("path=%s", "/x\n2026-01-01T00:00:00+0000 ERROR forged")

    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("path=/x 2026-01-01T00:00:00+0000 ERROR forged")


def test_text_traceback_keeps_its_line_breaks_indented(monkeypatch, capsys):
    monkeypatch.delenv("LOG_JSON", raising=False)
    setup_logging()

    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("whisper_ui.test").exception("failed")

    lines = capsys.readouterr().err.splitlines()
    assert "    Traceback (most recent call last):" in lines
    assert "    ValueError: boom" in lines
    assert lines[0].endswith("failed")


def test_exception_message_cannot_forge_a_record_or_reach_the_terminal(monkeypatch, capsys):
    monkeypatch.delenv("LOG_JSON", raising=False)
    setup_logging()

    try:
        raise ValueError("bad\x1b[31m\u202eevil\n2026-01-01T00:00:00+0000 ERROR whisper_ui forged")
    except ValueError:
        logging.getLogger("whisper_ui.test").exception("failed")

    err = capsys.readouterr().err
    lines = err.splitlines()
    assert "\x1b" not in err and "\u202e" not in err
    # Only the real record starts at column 0; the forged text is an indented
    # continuation of the traceback.
    assert [line for line in lines if not line.startswith(" ")] == [lines[0]]
    assert "    2026-01-01T00:00:00+0000 ERROR whisper_ui forged" in lines


def test_stack_info_lines_are_indented(monkeypatch, capsys):
    monkeypatch.delenv("LOG_JSON", raising=False)
    setup_logging()

    logging.getLogger("whisper_ui.test").info("with stack", stack_info=True)

    lines = capsys.readouterr().err.splitlines()
    assert lines[1] == "    Stack (most recent call last):"
    assert all(line.startswith("    ") for line in lines[1:])


def test_json_line_cannot_be_split_by_unicode_line_separators(monkeypatch, capsys):
    monkeypatch.setenv("LOG_JSON", "true")
    setup_logging()

    logging.getLogger("whisper_ui.test").info("title=%s", "a\u2028b\u202ec")

    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "title=a b c"
