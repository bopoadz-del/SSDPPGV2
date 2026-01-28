from __future__ import annotations

import csv
from io import StringIO
from typing import List, Tuple


def parse_histogram(csv_text: str) -> List[Tuple[float, float]]:
    """Parse histogram CSV text.

    Accepts columns: speed_mps, prob OR speed_mps, count.
    Returns list of (speed_mps, probability).
    """
    reader = csv.DictReader(StringIO(csv_text.strip()))
    if not reader.fieldnames:
        raise ValueError("CSV requires headers.")
    speeds = []
    values = []
    is_prob = "prob" in reader.fieldnames
    value_key = "prob" if is_prob else "count_or_prob"
    if not is_prob and "count" in reader.fieldnames:
        value_key = "count"
    for row in reader:
        speed = float(row.get("speed_mps", "0"))
        value = float(row.get(value_key, "0"))
        speeds.append(speed)
        values.append(value)
    if not speeds:
        raise ValueError("No data rows provided.")
    total = sum(values)
    if total <= 0:
        raise ValueError("Histogram total must be positive.")
    probs = [v / total for v in values]
    return list(zip(speeds, probs))
