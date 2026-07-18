from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def _minmax(values: list[float | None], *, lower_is_better: bool = False) -> list[float]:
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values)
    lo, hi = min(present), max(present)
    if hi == lo:
        return [0.5 if v is not None else 0.0 for v in values]
    scores = [0.0 if v is None else (v - lo) / (hi - lo) for v in values]
    return [1.0 - x if lower_is_better else x for x in scores]


def rank_rows(rows: list[dict], *, top_n: int = 20, max_per_scaffold: int = 2) -> list[dict]:
    affinity = _minmax([_float(r.get("docking_affinity")) for r in rows], lower_is_better=True)
    cnn_vs = _minmax([_float(r.get("cnn_vs")) for r in rows])
    plip = _minmax([_float(r.get("plip_score")) for r in rows])
    admet = _minmax([_float(r.get("admet_risk")) for r in rows], lower_is_better=True)
    mmgbsa = _minmax([_float(r.get("mmgbsa_delta_total")) for r in rows], lower_is_better=True)
    for i, row in enumerate(rows):
        available = []
        for weight, score, key in [(0.35, affinity[i], "docking_affinity"), (0.20, cnn_vs[i], "cnn_vs"),
                                   (0.20, plip[i], "plip_score"), (0.15, admet[i], "admet_risk"),
                                   (0.10, mmgbsa[i], "mmgbsa_delta_total")]:
            if row.get(key) not in (None, ""):
                available.append((weight, score))
        row["final_score"] = round(sum(w * s for w, s in available) / sum(w for w, _ in available), 6) if available else 0.0
    ordered = sorted(rows, key=lambda r: (-float(r["final_score"]), str(r.get("source_id", ""))))
    selected, counts = [], defaultdict(int)
    for row in ordered:
        scaffold = row.get("scaffold") or f"__{row.get('source_id')}"
        if counts[scaffold] >= max_per_scaffold:
            continue
        counts[scaffold] += 1
        row["rank"] = len(selected) + 1
        selected.append(row)
        if len(selected) >= top_n:
            break
    return selected


def rank_csv(input_csv: Path, output_csv: Path, *, top_n: int = 20) -> list[dict]:
    with input_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ranked = rank_rows(rows, top_n=top_n)
    if ranked:
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranked[0])); writer.writeheader(); writer.writerows(ranked)
    return ranked


def _float(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
