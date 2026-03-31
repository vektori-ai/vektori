"""Temporal query parsing — detect time windows in natural language queries.

Used by SearchPipeline to add before_date/after_date filters when a query
contains temporal language like "last week", "3 months ago", "in 2023", etc.

Design:
- Regex-based, no external dependencies, fast.
- Returns a TemporalWindow (after_date, before_date) or None if no signal found.
- Reference date defaults to utcnow(); override for deterministic testing.
- "recently" / "lately" → 7-day window (conservative, avoids false negatives).
- Relative expressions ("last N days") anchor to reference_date.
- Explicit year/month expressions ("in January", "in 2023") produce month/year ranges.
- Unknown or ambiguous expressions return None (no filter applied — safe fallback).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class TemporalWindow:
    after_date: Optional[datetime] = None   # fact event_time >= after_date
    before_date: Optional[datetime] = None  # fact event_time <= before_date


# ── Patterns (ordered from most specific to most general) ──────────────────

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_UNIT_TO_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}

# "last N days/weeks/months/years"
_LAST_N_UNIT = re.compile(
    r"\blast\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

# "last week / last month / last year"
_LAST_UNIT = re.compile(
    r"\blast\s+(week|month|year)\b",
    re.IGNORECASE,
)

# "N days/weeks/months/years ago"
_N_AGO = re.compile(
    r"\b(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago\b",
    re.IGNORECASE,
)

# "yesterday"
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)

# "recently" / "lately"
_RECENTLY = re.compile(r"\b(recently|lately)\b", re.IGNORECASE)

# "in [month]" — e.g. "in January", "in March"
_IN_MONTH = re.compile(
    r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)

# "in [year]" — e.g. "in 2023", "in 2021"
_IN_YEAR = re.compile(r"\bin\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)

# "before [year]"
_BEFORE_YEAR = re.compile(r"\bbefore\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)

# "after [year]"
_AFTER_YEAR = re.compile(r"\bafter\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)


class TemporalQueryParser:
    """Parse natural language queries for temporal window signals.

    Usage:
        parser = TemporalQueryParser()
        window = parser.parse("what did I say last week about Python?")
        # window.after_date = utcnow() - 7 days, window.before_date = None
    """

    def parse(
        self,
        query: str,
        reference_date: datetime | None = None,
    ) -> TemporalWindow | None:
        """Return a TemporalWindow if the query contains temporal language, else None."""
        ref = reference_date or datetime.utcnow()

        # "last N units"
        m = _LAST_N_UNIT.search(query)
        if m:
            n, unit = int(m.group(1)), m.group(2).lower()
            days = n * _UNIT_TO_DAYS[unit]
            return TemporalWindow(after_date=ref - timedelta(days=days))

        # "last week / last month / last year"
        m = _LAST_UNIT.search(query)
        if m:
            unit = m.group(1).lower()
            days = _UNIT_TO_DAYS[unit]
            return TemporalWindow(after_date=ref - timedelta(days=days))

        # "N units ago"
        m = _N_AGO.search(query)
        if m:
            n, unit = int(m.group(1)), m.group(2).lower()
            days = n * _UNIT_TO_DAYS[unit]
            anchor = ref - timedelta(days=days)
            # Window: ±half-unit around the anchor
            half = timedelta(days=max(1, days // 2))
            return TemporalWindow(after_date=anchor - half, before_date=anchor + half)

        # "yesterday"
        if _YESTERDAY.search(query):
            yesterday = ref - timedelta(days=1)
            return TemporalWindow(
                after_date=yesterday.replace(hour=0, minute=0, second=0),
                before_date=yesterday.replace(hour=23, minute=59, second=59),
            )

        # "recently" / "lately" — 30-day window; long-term memory contexts
        # make "recently" mean "past month", not "past week"
        if _RECENTLY.search(query):
            return TemporalWindow(after_date=ref - timedelta(days=30))

        # "before [year]"
        m = _BEFORE_YEAR.search(query)
        if m:
            year = int(m.group(1))
            return TemporalWindow(before_date=datetime(year, 1, 1))

        # "after [year]"
        m = _AFTER_YEAR.search(query)
        if m:
            year = int(m.group(1))
            return TemporalWindow(after_date=datetime(year, 12, 31))

        # "in [year]"
        m = _IN_YEAR.search(query)
        if m:
            year = int(m.group(1))
            return TemporalWindow(
                after_date=datetime(year, 1, 1),
                before_date=datetime(year, 12, 31, 23, 59, 59),
            )

        # "in [month]" — use the most recent occurrence of that month
        m = _IN_MONTH.search(query)
        if m:
            month_name = m.group(1).lower()
            month_num = _MONTHS.get(month_name)
            if month_num:
                year = ref.year if ref.month >= month_num else ref.year - 1
                # Window: entire month
                start = datetime(year, month_num, 1)
                # End: first day of next month minus 1 second
                if month_num == 12:
                    end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
                else:
                    end = datetime(year, month_num + 1, 1) - timedelta(seconds=1)
                return TemporalWindow(after_date=start, before_date=end)

        return None
