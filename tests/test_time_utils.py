from datetime import UTC, datetime, timedelta, timezone

from vla_eval.time_utils import as_beijing_time, format_beijing_time


def test_format_beijing_time_converts_utc_to_24_hour_display():
    value = datetime(2026, 8, 8, 10, 35, 42, tzinfo=UTC)

    assert format_beijing_time(value) == "2026-08-08 18:35:42（北京时间）"


def test_format_beijing_time_does_not_double_shift_existing_offset():
    value = datetime(
        2026, 8, 8, 18, 35, 42, tzinfo=timezone(timedelta(hours=8))
    )

    assert format_beijing_time(value) == "2026-08-08 18:35:42（北京时间）"


def test_legacy_naive_system_time_is_interpreted_as_utc():
    value = datetime(2026, 8, 8, 10, 35, 42)  # noqa: DTZ001 - legacy SQLite value

    converted = as_beijing_time(value)

    assert converted.utcoffset() == timedelta(hours=8)
    assert converted.hour == 18


def test_missing_time_uses_placeholder():
    assert format_beijing_time(None) == "—"
