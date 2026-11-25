"""The problem bank: hand curated classic LeetCode problems."""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "problems.json"
FAMILIES_PATH = Path(__file__).parent / "data" / "problem_families.json"
FAMILIES_VERSION = "families-v1"


def load() -> list[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)


def load_families(path: str | Path | None = None) -> dict:
    """Load the curated solution-template family layer.

    The file is versioned (``families-v1``): families carry stable
    identifiers, and each assignment names the canonical solution
    template(s) for one bank problem. A missing file raises
    ValueError; a structurally invalid file raises ValueError with
    the problems listed.
    """
    target = Path(path) if path is not None else FAMILIES_PATH
    if not target.exists():
        raise ValueError(f"problem families file not found: {target}")
    with open(target) as f:
        data = json.load(f)
    errors = _validate_structure(data)
    if errors:
        raise ValueError(
            "invalid problem families file "
            f"{target}: " + "; ".join(errors)
        )
    return data


def _validate_structure(data: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["families file must be a JSON object"]
    if data.get("version") != FAMILIES_VERSION:
        errors.append(
            f"version must be {FAMILIES_VERSION!r}, "
            f"got {data.get('version')!r}"
        )
    families = data.get("families")
    if not isinstance(families, list) or not families:
        errors.append("families must be a non-empty list")
        return errors
    seen: set[str] = set()
    for fam in families:
        fid = fam.get("id") if isinstance(fam, dict) else None
        if not isinstance(fid, str) or not fid:
            errors.append("every family needs a non-empty string id")
        elif fid in seen:
            errors.append(f"duplicate family id {fid!r}")
        else:
            seen.add(fid)
    assignments = data.get("assignments")
    if not isinstance(assignments, list):
        errors.append("assignments must be a list")
        return errors
    seen_ids: set[int] = set()
    for row in assignments:
        if not isinstance(row, dict):
            errors.append("every assignment must be an object")
            continue
        pid = row.get("problem_id")
        if pid in seen_ids:
            errors.append(f"duplicate assignment for problem {pid}")
        seen_ids.add(pid)
        tags = row.get("families")
        if not isinstance(tags, list) or not tags:
            errors.append(f"problem {pid} has no families")
            continue
        for tag in tags:
            if tag not in seen:
                errors.append(
                    f"problem {pid} names unknown family {tag!r}"
                )
    return errors


def family_ids(families: dict) -> list[str]:
    return [f["id"] for f in families.get("families", [])]


def families_for(problem_id: int, families: dict) -> list[str]:
    """The family tags for one problem, or [] when it has none."""
    for row in families.get("assignments", []):
        if row.get("problem_id") == problem_id:
            return list(row.get("families", []))
    return []


def primary_family(problem_id: int, families: dict) -> str | None:
    """The first tag for a problem: its primary, canonical template.

    A problem may carry more than one tag, but the first is the
    primary template, and primary families partition the bank, so a
    split by primary family can hold whole families out with no
    primary-template leak. Secondary tags can still co-occur across
    a split; the leakage audit reports that.
    """
    tags = families_for(problem_id, families)
    return tags[0] if tags else None


def validate_families(bank: list[dict], families: dict) -> list[str]:
    """Cross-check the family layer against the bank.

    Returns a list of error strings; empty means valid. Every
    assignment must name a real bank problem, every family tag must
    name a defined family, and every bank problem must carry at
    least one tag, so an untagged problem is visible instead of
    silently treated as family-free.
    """
    errors = _validate_structure(families)
    bank_ids = {p["id"] for p in bank}
    assigned = {a.get("problem_id") for a in families.get("assignments", [])}
    for pid in sorted(assigned - bank_ids):
        errors.append(f"assignment names unknown problem {pid}")
    for p in bank:
        if p["id"] not in assigned:
            errors.append(f"problem {p['id']} has no family tags")
    return errors


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
