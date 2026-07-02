from __future__ import annotations

import csv
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def write_agency_drift_monitor(
    agencies_config: Path,
    out_dir: Path,
    previous: Path | None = None,
    as_of: str | None = None,
    max_age_days: int = 90,
) -> dict[str, Any]:
    report = build_agency_drift_monitor(
        agencies_config=agencies_config,
        previous=previous,
        as_of=as_of,
        max_age_days=max_age_days,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agency_drift_monitor.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(report["refresh_tasks"], out_dir / "agency_drift_tasks.csv")
    (out_dir / "agency_drift_monitor.md").write_text(_render_report(report), encoding="utf-8")
    return report


def build_agency_drift_monitor(
    agencies_config: Path,
    previous: Path | None = None,
    as_of: str | None = None,
    max_age_days: int = 90,
) -> dict[str, Any]:
    config = json.loads(agencies_config.read_text(encoding="utf-8"))
    current = _current_sources(config)
    previous_rows = _previous_sources(previous)
    tasks = _refresh_tasks(current, previous_rows, _parse_date(as_of) or date.today(), max_age_days)
    return {
        "agencies_config": str(agencies_config),
        "previous": str(previous) if previous else None,
        "as_of": as_of or date.today().isoformat(),
        "max_age_days": max_age_days,
        "retrieved_at": config.get("retrieved_at"),
        "agency_count": len(current),
        "sources": current,
        "refresh_tasks": tasks,
        "summary": {
            "task_count": len(tasks),
            "changed_count": sum(1 for task in tasks if task["reason"] == "source_fingerprint_changed"),
            "stale_count": sum(1 for task in tasks if task["reason"] == "retrieval_stale"),
            "baseline_count": sum(1 for task in tasks if task["reason"] == "baseline_missing"),
        },
        "boundary": (
            "Drift monitor compares local roster metadata only. It creates refresh tasks before "
            "old names, source URLs, or examples are reused; it does not fetch or verify pages."
        ),
    }


def _current_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    retrieved_at = config.get("retrieved_at")
    rows = []
    for agency in config.get("agencies", []):
        if not isinstance(agency, dict):
            continue
        source_urls = [
            str(source.get("url", ""))
            for source in agency.get("official_sources", [])
            if isinstance(source, dict) and source.get("url")
        ]
        public_examples = [str(item) for item in agency.get("public_examples", [])]
        row = {
            "slug": str(agency.get("slug", "")),
            "name": str(agency.get("name", "")),
            "retrieved_at": retrieved_at,
            "source_urls": source_urls,
            "public_examples": public_examples,
        }
        row["source_fingerprint"] = _fingerprint(row)
        rows.append(row)
    return rows


def _previous_sources(previous: Path | None) -> dict[str, dict[str, Any]]:
    if previous is None or not previous.exists():
        return {}
    payload = json.loads(previous.read_text(encoding="utf-8"))
    rows = payload.get("sources", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("slug", "")): row
        for row in rows
        if isinstance(row, dict) and row.get("slug")
    }


def _refresh_tasks(
    current: list[dict[str, Any]],
    previous: dict[str, dict[str, Any]],
    as_of: date,
    max_age_days: int,
) -> list[dict[str, Any]]:
    tasks = []
    for row in current:
        previous_row = previous.get(row["slug"])
        if previous_row is None:
            tasks.append(_task(row, "baseline_missing", "Create a baseline drift snapshot before reuse."))
        elif previous_row.get("source_fingerprint") != row.get("source_fingerprint"):
            tasks.append(_task(row, "source_fingerprint_changed", "Refresh and re-review official roster evidence."))
        age_days = _age_days(row.get("retrieved_at"), as_of)
        if age_days is not None and age_days > max_age_days:
            task = _task(row, "retrieval_stale", f"Refresh source metadata; retrieved_at is {age_days} days old.")
            task["age_days"] = age_days
            tasks.append(task)
    return tasks


def _task(row: dict[str, Any], reason: str, action: str) -> dict[str, Any]:
    return {
        "slug": row["slug"],
        "name": row["name"],
        "reason": reason,
        "action": action,
        "retrieved_at": row.get("retrieved_at"),
        "source_urls": row.get("source_urls", []),
        "source_fingerprint": row.get("source_fingerprint"),
    }


def _fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "slug": row["slug"],
        "name": row["name"],
        "retrieved_at": row.get("retrieved_at"),
        "source_urls": row.get("source_urls", []),
        "public_examples": row.get("public_examples", []),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _age_days(value: Any, as_of: date) -> int | None:
    parsed = _parse_date(value)
    if parsed is None:
        return None
    return (as_of - parsed).days


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _write_csv(tasks: list[dict[str, Any]], path: Path) -> None:
    headers = ["slug", "name", "reason", "retrieved_at", "action", "source_urls"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "slug": task["slug"],
                    "name": task["name"],
                    "reason": task["reason"],
                    "retrieved_at": task.get("retrieved_at"),
                    "action": task["action"],
                    "source_urls": ";".join(task.get("source_urls", [])),
                }
            )


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# agency drift monitor",
        "",
        f"- agencies_config: {report['agencies_config']}",
        f"- previous: {report['previous'] or ''}",
        f"- as_of: {report['as_of']}",
        f"- max_age_days: {report['max_age_days']}",
        f"- agency_count: {report['agency_count']}",
        f"- refresh_tasks: {report['summary']['task_count']}",
        "",
        "## Refresh Tasks",
        "",
        "| agency | reason | retrieved_at | action |",
        "| --- | --- | --- | --- |",
    ]
    if not report["refresh_tasks"]:
        lines.append("|  | none |  | no refresh task |")
    for task in report["refresh_tasks"]:
        lines.append(
            f"| {task['name']} | {task['reason']} | {task.get('retrieved_at') or ''} | {task['action']} |"
        )
    lines.extend(["", "## Boundary", "", report["boundary"], ""])
    return "\n".join(lines)
