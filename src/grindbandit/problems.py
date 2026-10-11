"""The problem bank: hand curated classic LeetCode problems."""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "problems.json"


def load() -> list[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)


def by_id(bank: list[dict], pid: int) -> dict | None:
    for p in bank:
        if p["id"] == pid:
            return p
    return None


def slug(title: str) -> str:
    s = title.lower().replace(" - ", " ")
    s = re.sub(r"[^a-z0-9 -]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    return re.sub(r"-{2,}", "-", s)


def url(problem: dict) -> str:
    return f"https://leetcode.com/problems/{slug(problem['title'])}/"


def all_topics(bank: list[dict]) -> list[str]:
    seen: dict = {}
    for p in bank:
        for t in p["topics"]:
            seen.setdefault(t)
    return sorted(seen)
