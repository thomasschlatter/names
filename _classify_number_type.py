# -*- coding: utf-8 -*-
"""What kind of number does each source actually carry? Measured, not assumed.

Adds to audits/sources_lookup.csv:

  number_type     absolute counts | rank only | percentage or share
                  | presence only (no quantity) | mixed or unclear | file missing
  number_evidence the measurement behind the verdict
  n_rows, value_min, value_max, has_year

The verdict comes from the values in the file, not from what a document says the
source publishes, because the two can disagree: a rank-ordered top list loaded
into a `count` column still looks like a count until you notice the values run
1, 2, 3 ... with no ties.

Rules, applied per source and checked within (sex, year) groups:

  presence only        every value identical (usually 1): the file records that
                       a name occurs, not how often
  rank only            values are a near-complete run 1..n inside each group,
                       with (almost) no ties. Real frequency counts tie often
                       and skip values; ranks do neither
  percentage or share  non-integer values, none above 100
  absolute counts      integer values with ties and gaps, spanning a wide range
  mixed or unclear     none of the above fits

Usage: python _classify_number_type.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SHARED = Path("H:/My Drive/PROJECTS/_RESOURCES/official_data")
LOOKUP = BASE / "audits" / "sources_lookup.csv"


def describe(path: Path) -> dict:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return {"number_type": "file unreadable",
                "number_evidence": f"{type(exc).__name__}"}
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "count" not in frame.columns:
        return {"number_type": "presence only (no quantity)",
                "number_evidence": f"no count column; columns are {list(frame.columns)[:6]}",
                "n_rows": len(frame)}

    values = pd.to_numeric(frame["count"], errors="coerce").dropna()
    if values.empty:
        return {"number_type": "presence only (no quantity)",
                "number_evidence": "count column present but no numeric values",
                "n_rows": len(frame)}

    out = {"n_rows": int(len(frame)), "value_min": float(values.min()),
           "value_max": float(values.max()),
           "has_year": "year" in frame.columns and frame["year"].notna().any()}

    if values.nunique() == 1:
        out.update(number_type="presence only (no quantity)",
                   number_evidence=f"every value is {values.iloc[0]:g}")
        return out

    non_integer = float((values % 1 != 0).mean())
    if non_integer > 0.2 and values.max() <= 100:
        out.update(number_type="percentage or share",
                   number_evidence=f"{non_integer:.0%} non-integer, max {values.max():g}")
        return out

    # Rank test, within (sex, year) where those exist.
    group_cols = [c for c in ("sex", "year") if c in frame.columns]
    rank_like = 0
    groups = 0
    if group_cols:
        work = frame.copy()
        work["_v"] = pd.to_numeric(work["count"], errors="coerce")
        for _, group in work.groupby(group_cols, dropna=False):
            vals = group["_v"].dropna()
            if len(vals) < 5:
                continue
            groups += 1
            expected = set(range(1, len(vals) + 1))
            overlap = len(set(vals.astype(int)) & expected) / len(expected)
            tie_rate = 1 - vals.nunique() / len(vals)
            if overlap > 0.9 and tie_rate < 0.05:
                rank_like += 1
    if groups and rank_like / groups > 0.8:
        out.update(number_type="rank only",
                   number_evidence=f"{rank_like}/{groups} groups are a 1..n run with no ties")
        return out

    tie_rate = 1 - values.nunique() / len(values)
    out.update(number_type="absolute counts",
               number_evidence=f"integers, range {values.min():g}-{values.max():g}, "
                               f"{tie_rate:.0%} tied values")
    return out


def main() -> None:
    lookup = pd.read_csv(LOOKUP)
    rows = []
    for record in lookup.itertuples():
        name = str(getattr(record, "file", "") or "")
        path = SHARED / name
        if not name or not path.exists():
            rows.append({"number_type": "not a local name list",
                         "number_evidence": "file not in _RESOURCES/official_data"})
            continue
        rows.append(describe(path))
    extra = pd.DataFrame(rows, index=lookup.index)
    for column in ("number_type", "number_evidence", "n_rows",
                   "value_min", "value_max", "has_year"):
        lookup[column] = extra.get(column)
    lookup.to_csv(LOOKUP, index=False, encoding="utf-8")

    local = lookup[lookup.number_type != "not a local name list"]
    print(f"sources in lookup: {len(lookup)}   local name lists measured: {len(local)}\n")
    print("number type:")
    print(local["number_type"].value_counts().to_string())
    for label in ("rank only", "presence only (no quantity)", "percentage or share"):
        subset = local[local.number_type == label]
        if len(subset):
            print(f"\n{label} ({len(subset)}):")
            print(subset[["source_id", "n_rows", "number_evidence"]]
                  .head(12).to_string(index=False))
    print("\ncross-tab of quality tier against number type:")
    print(pd.crosstab(local["quality_tier"], local["number_type"]).to_string())


if __name__ == "__main__":
    main()
