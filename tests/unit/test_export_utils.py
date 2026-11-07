from __future__ import annotations

import pytest

from whisper_ui.export.utils import format_timestamp, strip_control_chars


def test_format_timestamp_comma():
    assert format_timestamp(0.0, ",") == "00:00:00,000"
    assert format_timestamp(1.5, ",") == "00:00:01,500"
    assert format_timestamp(3661.123, ",") == "01:01:01,123"


def test_format_timestamp_dot():
    assert format_timestamp(0.0, ".") == "00:00:00.000"
    assert format_timestamp(1.5, ".") == "00:00:01.500"
    assert format_timestamp(3661.123, ".") == "01:01:01.123"


def test_format_timestamp_default_separator():
    assert format_timestamp(1.5) == "00:00:01,500"


def test_format_timestamp_clamps_negative_to_zero():
    # A negative offset must not borrow into an invalid "-1:59:59,500" timecode.
    assert format_timestamp(-0.5) == "00:00:00,000"
    assert format_timestamp(-1.2, ".") == "00:00:00.000"


def test_strip_control_chars_removes_xml_incompatible():
    assert strip_control_chars("a\x00b\x08c\x1fd") == "abcd"


def test_strip_control_chars_keeps_legal_whitespace_and_text():
    assert strip_control_chars("line1\nline2\ttab\rcr") == "line1\nline2\ttab\rcr"


def test_strip_control_chars_preserves_cjk_and_fullwidth():
    assert strip_control_chars("逐字稿。全形「測試」") == "逐字稿。全形「測試」"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (1.001, "00:00:01,001"),  # 1.001 % 1 is 0.000999..., which truncated to 000
        (1.003, "00:00:01,003"),
        (59.9996, "00:01:00,000"),  # rounding carries into the next minute
        (3599.9999, "01:00:00,000"),
    ],
)
def test_format_timestamp_keeps_every_millisecond(seconds, expected):
    assert format_timestamp(seconds, ",") == expected


def test_format_timestamp_round_trips_all_millisecond_values():
    for total_ms in range(0, 120_000):
        text = format_timestamp(total_ms / 1000, ",")
        h, m, rest = text.split(":")
        s, ms = rest.split(",")
        assert ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms) == total_ms
