from __future__ import annotations

import csv
from io import StringIO
from typing import Dict, Iterable, List


def dicts_to_csv(rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()
