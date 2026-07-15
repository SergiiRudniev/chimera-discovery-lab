"""Validation helpers for the append-only public research ledger."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

HYPOTHESIS_ID = re.compile(r"^CHM-(V|C|O|A|N|F)-H\d{3}$")
ALLOWED_STATUSES = {"registered", "running", "accepted", "rejected", "inconclusive"}


def load_research_registry(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("hypotheses"), list):
        raise ValueError("research registry must contain a hypotheses list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_record in payload["hypotheses"]:
        if not isinstance(raw_record, Mapping):
            raise ValueError("every hypothesis record must be a mapping")
        record = dict(raw_record)
        hypothesis_id = record.get("id")
        status = record.get("status")
        if not isinstance(hypothesis_id, str) or not HYPOTHESIS_ID.fullmatch(hypothesis_id):
            raise ValueError(f"invalid hypothesis ID: {hypothesis_id!r}")
        if hypothesis_id in seen:
            raise ValueError(f"duplicate hypothesis ID: {hypothesis_id}")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid status for {hypothesis_id}: {status!r}")
        for required in ("title", "registered_at", "config", "result"):
            if not record.get(required):
                raise ValueError(f"{hypothesis_id} is missing {required}")
        seen.add(hypothesis_id)
        records.append(record)
    return records
