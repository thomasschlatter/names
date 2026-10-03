# -*- coding: utf-8 -*-
"""Rebuild official_data_top30/ from the shared registry harvest.

official_data_top30/ is the derived hand-check subset the regression reads:
the top 30 female and top 30 male names per registry by count. It had no
producer script in this folder, so this file supplies one.

Source (one copy, absolute path, never copied into this project):
    H:/My Drive/PROJECTS/_RESOURCES/official_data/official_*.csv

Usage:
    python _build_top30.py --verify   compare against what is already on disk
    python _build_top30.py --write    regenerate the subset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "official_data_top30"
SRC = Path("H:/My Drive/PROJECTS/_RESOURCES/official_data")
TOP_N = 30


def build_one(path: Path, latest_year_only: bool = True):
    """Top-N distinct names per sex for one registry.

    YEAR HANDLING (decided 2026-09-18). Registries now span 1 to 164 years, so
    pooling every year makes each country contribute names from a different era:
    France pooled 1900-2023 gives Marie and Jean, France 2015+ gives Emma and
    Gabriel. To compare like with like, a registry that HAS years contributes
    only its MOST RECENT year. A registry with no year column contributes all of
    its rows, because it cannot be placed in time and dropping it would remove
    Brazil, Spain and the USA, which together outweigh the rest of the set.

    The full multi-year series stays untouched in _RESOURCES/official_data for
    the diachronic analysis; only this derived subset is restricted.
    """
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if not {"name", "sex", "count"}.issubset(frame.columns):
        return None
    if "year" not in frame.columns:
        frame["year"] = pd.NA
    frame = frame[["name", "sex", "year", "count"]].copy()
    frame["name"] = frame["name"].astype(str).str.strip()
    frame["sex"] = frame["sex"].astype(str).str.upper().str.strip().str[0]
    frame["count"] = pd.to_numeric(frame["count"], errors="coerce")
    frame = frame[(frame["name"] != "") & frame["sex"].isin(["F", "M"])
                  & frame["count"].notna() & (frame["count"] > 0)]
    if frame.empty:
        return None
    year_used = "undated"
    years = pd.to_numeric(frame["year"], errors="coerce")
    if latest_year_only and years.notna().any():
        latest = int(years.max())
        frame = frame[years == latest]
        year_used = str(latest)
        if frame.empty:
            return None
    frame.attrs["year_used"] = year_used
    # Pool a name's counts across years, then take the most frequent per sex.
    pooled = frame.groupby(["name", "sex"], as_index=False)["count"].sum()
    top = (
        pooled.sort_values(["sex", "count", "name"], ascending=[True, False, True])
        .groupby("sex", group_keys=False)
        .head(TOP_N)
    )
    top = top.sort_values(["sex", "count", "name"], ascending=[True, False, True])
    top.attrs["year_used"] = frame.attrs.get("year_used", "undated")
    return top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    # Backups and superseded copies live beside the live files; including them
    # would enter France, England, Sweden and Slovenia twice, once with their
    # pre-re-collection data.
    skip = ("_pre_recollect_", "legacy_", "_legacy", "_backup")
    sources = sorted(p for p in SRC.glob("official_*.csv")
                     if not any(s in p.stem for s in skip))
    print(f"{len(sources)} registries in {SRC}")

    rows, same, differ, new = [], 0, [], []
    for path in sources:
        top = build_one(path)
        if top is None or top.empty:
            continue
        existing = OUT / path.name
        if existing.exists():
            old = pd.read_csv(existing)
            old.columns = [c.strip().lower() for c in old.columns]
            old_names = set(zip(old["name"].astype(str), old["sex"].astype(str)))
            new_names = set(zip(top["name"].astype(str), top["sex"].astype(str)))
            if old_names == new_names:
                same += 1
            else:
                differ.append((path.stem, len(old_names - new_names), len(new_names - old_names)))
        else:
            new.append(path.stem)
        rows.append({"registry": path.stem.replace("official_", ""),
                     "year_used": top.attrs.get("year_used", "undated"),
                     "F": int((top["sex"] == "F").sum()),
                     "M": int((top["sex"] == "M").sum()),
                     "total": len(top)})
        if args.write:
            top.to_csv(existing, index=False, encoding="utf-8")

    print(f"reproduces existing file exactly: {same}")
    print(f"differs from existing:            {len(differ)}")
    for stem, lost, gained in differ[:15]:
        print(f"   {stem}: {lost} names only in old, {gained} only in new")
    print(f"not previously present (new):     {len(new)}")
    for stem in new:
        print(f"   {stem}")
    if args.write:
        man = pd.DataFrame(rows).sort_values("registry")
        man.to_csv(OUT / "_manifest.csv", index=False, encoding="utf-8")
        print(f"\nwrote {len(rows)} subset files and _manifest.csv")


if __name__ == "__main__":
    main()
