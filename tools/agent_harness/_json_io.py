"""Read optional JSON metadata shared by runner and log parsers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
