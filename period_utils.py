"""
Period parsing for FINA's chat context.

Single source of truth for "what does this period string mean." Used by
api.py (context builder) and mac_client.py (date-range filter). Supports:

  - "month"        : current calendar month
  - "year"         : year-to-date
  - "all"          : lifetime
  - "YYYY-MM"      : explicit month
  - "1m" / "3m" / "6m" / "12m" : rolling N months ending today

Also provides a best-effort inference from free-text user questions so the
brain can react to "last 3 months" or "year to date" without the client
having to set the period explicitly.
"""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

logger = logging.getLogger("fina")

_SENTINEL_LIFETIME_START = date(1970, 1, 1)
_NM_PATTERN = re.compile(r"^(\d{1,2})m$", re.IGNORECASE)
_YYYY_MM_PATTERN = re.compile(r"^(\d{4})-(\d{2})$")


@dataclass(frozen=True)
class PeriodSpec:
    raw: str
    label: str
    start: date
    end: date            # inclusive
    months_span: int     # 1 for single month; N for rolling Nm; 12 for year, large for all
    is_calendar_month: bool   # True only for "month" or "YYYY-MM"


def _months_between(start: date, end: date) -> int:
    """Whole months between two dates, minimum 1."""
    return max(1, (end.year - start.year) * 12 + (end.month - start.month) + 1)


def parse_period(period: Optional[str], *, today: Optional[date] = None) -> PeriodSpec:
    """Parse a period string into a PeriodSpec. Falls back to current month."""
    today = today or date.today()
    raw = (period or "month").strip().lower()

    if raw == "month":
        days_in_month = calendar.monthrange(today.year, today.month)[1]
        start = date(today.year, today.month, 1)
        end = date(today.year, today.month, days_in_month)
        return PeriodSpec(
            raw=raw,
            label=f"{today.strftime('%B %Y')} (current month)",
            start=start, end=end,
            months_span=1, is_calendar_month=True,
        )

    if raw == "prev_month":
        # Previous calendar month — preferred for natural-language "last month".
        prev_year = today.year if today.month > 1 else today.year - 1
        prev_month = today.month - 1 if today.month > 1 else 12
        days_in_prev = calendar.monthrange(prev_year, prev_month)[1]
        start = date(prev_year, prev_month, 1)
        end = date(prev_year, prev_month, days_in_prev)
        return PeriodSpec(
            raw=raw,
            label=f"{start.strftime('%B %Y')} (last month)",
            start=start, end=end,
            months_span=1, is_calendar_month=True,
        )

    if raw == "year":
        return PeriodSpec(
            raw=raw,
            label=f"{today.year} year-to-date",
            start=date(today.year, 1, 1), end=today,
            months_span=today.month, is_calendar_month=False,
        )

    if raw == "all":
        return PeriodSpec(
            raw=raw,
            label="all time",
            start=_SENTINEL_LIFETIME_START, end=today,
            months_span=_months_between(_SENTINEL_LIFETIME_START, today),
            is_calendar_month=False,
        )

    nm = _NM_PATTERN.match(raw)
    if nm:
        n = int(nm.group(1))
        if n < 1:
            n = 1
        # Rolling N months: start = today minus N months (same day-of-month
        # when possible, clamped to month length).
        start_year = today.year
        start_month = today.month - n
        while start_month <= 0:
            start_month += 12
            start_year -= 1
        start_day = min(today.day, calendar.monthrange(start_year, start_month)[1])
        start = date(start_year, start_month, start_day)
        return PeriodSpec(
            raw=raw,
            label=f"last {n} month{'s' if n != 1 else ''}",
            start=start, end=today,
            months_span=n, is_calendar_month=False,
        )

    ym = _YYYY_MM_PATTERN.match(raw)
    if ym:
        year, month = int(ym.group(1)), int(ym.group(2))
        if 1 <= month <= 12:
            days_in_month = calendar.monthrange(year, month)[1]
            return PeriodSpec(
                raw=raw,
                label=date(year, month, 1).strftime("%B %Y"),
                start=date(year, month, 1),
                end=date(year, month, days_in_month),
                months_span=1, is_calendar_month=True,
            )

    logger.warning("[period] unrecognized period %r — defaulting to current month", period)
    return parse_period("month", today=today)


# ── Inference from free text ────────────────────────────────────────────────
# Order matters: more specific patterns first so "last 3 months" doesn't
# match the generic "month" branch.
_INFERENCE_RULES: list[tuple[re.Pattern[str], str]] = [
    # "last month" → previous calendar month (resolved before the generic
    # rolling-N-months rule, which would otherwise eat it).
    (re.compile(r"\b(?:last|previous)\s+month\b", re.I), "prev_month"),
    (re.compile(r"\b(?:last|past|previous)\s+(\d{1,2})\s*[- ]?\s*month", re.I), r"\1m"),
    (re.compile(r"\b(\d{1,2})\s*[- ]?\s*month\s*(?:recap|review|summary|overview)", re.I), r"\1m"),
    (re.compile(r"\bhalf\s*[- ]?\s*year\b|\bsix\s*months\b", re.I), "6m"),
    (re.compile(r"\bquarter(?:ly)?\b|\bthree\s*months\b", re.I), "3m"),
    (re.compile(r"\b(?:ytd|year[- ]to[- ]date|this\s+year|so far this year)\b", re.I), "year"),
    (re.compile(r"\b(?:all[- ]time|lifetime|overall|since\s+(?:the\s+)?start)\b", re.I), "all"),
]


def infer_period_from_text(message: str) -> Optional[str]:
    """Best-effort: extract a period token from a user question. None if no hit."""
    if not message:
        return None
    for pattern, replacement in _INFERENCE_RULES:
        m = pattern.search(message)
        if m:
            if r"\1" in replacement:
                n = int(m.group(1))
                if 1 <= n <= 24:
                    return f"{n}m"
            else:
                return replacement
    return None
