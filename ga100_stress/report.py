"""Result collection and pass/fail reporting."""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Result:
    name: str
    passed: bool
    seconds: float = 0.0
    metrics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def print_summary(results: list[Result]) -> bool:
    all_pass = all(r.passed for r in results)
    width = max((len(r.name) for r in results), default=0)
    for r in results:
        print(f"  {r.name:<{width}}  {'PASS' if r.passed else 'FAIL'}")
    print(f"\noverall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def json_report(results: list[Result], gpu: dict, version: str) -> str:
    doc = {
        "tool": "ga100-stress",
        "version": version,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "gpu": gpu,
        "results": [r.as_dict() for r in results],
        "all_pass": all(r.passed for r in results),
    }
    return json.dumps(doc, indent=2)
