from datetime import UTC, datetime
from zoneinfo import ZoneInfo

BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


def as_beijing_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(BEIJING_TIMEZONE)


def format_beijing_time(value: datetime | None) -> str:
    if value is None:
        return "—"
    return f"{as_beijing_time(value):%Y-%m-%d %H:%M:%S}（北京时间）"


def beijing_now() -> datetime:
    return datetime.now(UTC).astimezone(BEIJING_TIMEZONE)
