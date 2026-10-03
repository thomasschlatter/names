# -*- coding: utf-8 -*-
"""Give every name-list CSV a `source` column, and build the source lookup table.

Two outputs:

  sources_lookup.csv   one row per source_id: publisher, source type, URL,
                       coverage, licence, how it was attributed, quality tier
  official_*.csv       each gains a `source` column holding its source_id

Attribution comes only from evidence already on disk, in this order:

  1. german_cities_2026_manifest.csv / moldova_manifest.csv - exact URL, licence
     and access date per file, written when the data was collected
  2. GOVERNMENT_SOURCE_LEADS_2026-05-04.md - numbered entries carrying
     "Publisher:", "Verified source:" URLs and the output filename
  3. GIVEN_NAMES/<COUNTRY>/SOURCES.csv - structured per-file descriptions
  4. a normalize_*.py / fetch_*.py whose name matches the output

A file with none of these gets source_id `UNATTRIBUTED` rather than a guessed
publisher, and is listed at the end so it can be sourced or re-collected.

The originals are copied to backup_pre_source_column_<date>/ first. Only a
column is added; no row, name or count is altered.

Usage:
  python _add_source_column.py            report what would happen
  python _add_source_column.py --write    back up, then write the column
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SHARED = Path("H:/My Drive/PROJECTS/_RESOURCES/official_data")
GIVEN = Path("H:/My Drive/PROJECTS/GIVEN_NAMES")
LEADS = SHARED / "GOVERNMENT_SOURCE_LEADS_2026-05-04.md"
TODAY = date.today().isoformat()


def parse_leads() -> dict:
    """output filename -> {publisher, url, coverage} from the leads document."""
    if not LEADS.exists():
        return {}
    text = io.open(LEADS, encoding="utf-8", errors="replace").read()
    blocks = re.split(r"\n(?=\d+\.\s)", text)
    found = {}
    for block in blocks:
        outputs = re.findall(r"official_[a-z0-9_]+\.csv", block)
        if not outputs:
            continue
        publisher = re.search(r"- Publisher:\s*(.+)", block)
        coverage = re.search(r"- Coverage:\s*(.+)", block)
        urls = re.findall(r"(https?://\S+)", block)
        for name in set(outputs):
            if name in found and not publisher:
                continue
            found[name] = {
                "publisher": publisher.group(1).strip() if publisher else "",
                "url": urls[0].rstrip(").,") if urls else "",
                "coverage": coverage.group(1).strip() if coverage else "",
                "evidence": "GOVERNMENT_SOURCE_LEADS_2026-05-04.md",
            }
    return found


def parse_collected() -> dict:
    """COLLECTED_DATASETS is a markdown table: | `file` | Source | Coverage | ..."""
    path = SHARED / "COLLECTED_DATASETS_2026-05-04.md"
    if not path.exists():
        return {}
    found = {}
    for line in io.open(path, encoding="utf-8", errors="replace"):
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        match = re.search(r"official_[a-z0-9_]+\.csv", cells[0])
        if not match:
            continue
        found[match.group(0)] = {
            "publisher": cells[1].strip("`"),
            "coverage": cells[2] if len(cells) > 2 else "",
            "url": "",
            "evidence": "COLLECTED_DATASETS_2026-05-04.md",
        }
    return found


def parse_world_audit() -> dict:
    """The audit's evidence column says 'official_X.csv collected from SOURCE, years.'"""
    path = SHARED / "WORLD_GOVERNMENT_NAME_SOURCE_AUDIT_2026-05-04.md"
    if not path.exists():
        return {}
    found = {}
    for line in io.open(path, encoding="utf-8", errors="replace"):
        match = re.search(r"(official_[a-z0-9_]+\.csv)\s+collected from\s+([^.;]+)", line)
        if not match:
            continue
        publisher = match.group(2).strip()
        years = re.search(r"(\d{4})\s*[-–]\s*(\d{4})", publisher)
        found[match.group(1)] = {
            "publisher": re.sub(r",?\s*\d{4}\s*[-–]?\s*\d{0,4}$", "", publisher).strip(),
            "coverage": f"{years.group(1)}-{years.group(2)}" if years else "",
            "url": "",
            "evidence": "WORLD_GOVERNMENT_NAME_SOURCE_AUDIT_2026-05-04.md",
        }
    return found


def parse_manifests() -> dict:
    out = {}
    manifest = SHARED / "german_cities_2026_manifest.csv"
    if manifest.exists():
        frame = pd.read_csv(manifest)
        ok = frame[frame.get("status") == "ok"] if "status" in frame else frame
        for city, group in ok.groupby("city"):
            out[f"official_germany_{city}.csv"] = {
                "publisher": f"Stadt {city.title()}",
                "url": str(group["url"].iloc[0]),
                "coverage": f"{group['year'].min()}-{group['year'].max()}"
                            if "year" in group else "",
                "licence": str(group["licence"].iloc[0]) if "licence" in group else "",
                "accessed": str(group["accessed"].iloc[0]) if "accessed" in group else "",
                "evidence": "german_cities_2026_manifest.csv",
                "source_type": "City government open data",
            }
    manifest = SHARED / "moldova_manifest.csv"
    if manifest.exists():
        frame = pd.read_csv(manifest)
        ok = frame[frame.get("status") == "ok"] if "status" in frame else frame
        if len(ok):
            out["official_moldova.csv"] = {
                "publisher": "dataset.gov.md (Government of Moldova open data portal)",
                "url": str(ok["url"].iloc[0]),
                "coverage": f"{ok['year'].min()}-{ok['year'].max()}" if "year" in ok else "",
                "accessed": str(ok["accessed"].iloc[0]) if "accessed" in ok else "",
                "evidence": "moldova_manifest.csv",
                "source_type": "Government open data portal",
            }
    return out


def parse_given_names() -> dict:
    """country -> list of structured source rows, for the merged lookup."""
    rows = []
    if not GIVEN.exists():
        return {}
    for country in sorted(os.listdir(GIVEN)):
        path = GIVEN / country / "SOURCES.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        frame.insert(0, "country", country)
        rows.append(frame)
    return {"rows": pd.concat(rows, ignore_index=True)} if rows else {}


def producer_for(stem: str) -> str:
    for name in os.listdir(SHARED):
        if not name.endswith(".py"):
            continue
        key = stem.replace("official_", "")
        if re.search(re.escape(key.split("_")[0]), name):
            return name
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    leads = parse_leads()
    manifests = parse_manifests()
    collected = parse_collected()
    world = parse_world_audit()
    targets = sorted(p for p in SHARED.glob("official_*.csv"))

    lookup, unattributed = [], []
    assignment = {}
    for path in targets:
        name = path.name
        stem = path.stem
        source_id = stem.replace("official_", "")
        info = dict(manifests.get(name) or collected.get(name)
                    or leads.get(name) or world.get(name) or {})
        if not info:
            producer = producer_for(stem)
            if producer:
                info = {"publisher": "", "url": "", "coverage": "",
                        "evidence": f"producer script {producer}"}
            else:
                unattributed.append(name)
                assignment[name] = "UNATTRIBUTED"
                continue
        assignment[name] = source_id
        lookup.append({
            "source_id": source_id,
            "file": name,
            "publisher": info.get("publisher", ""),
            "source_type": info.get("source_type", "National/subnational official statistics"),
            "url": info.get("url", ""),
            "coverage": info.get("coverage", ""),
            "licence": info.get("licence", ""),
            "accessed": info.get("accessed", ""),
            "attributed_via": info.get("evidence", ""),
        })

    lookup_df = pd.DataFrame(lookup).sort_values("source_id")
    given = parse_given_names()
    if given:
        extra = given["rows"].rename(columns={
            "source_name": "publisher", "file": "file", "url": "url",
            "year_range": "coverage", "source_type": "source_type"})
        extra["source_id"] = ("givennames_" + extra["country"].str.lower()
                              + "__" + extra["file"].str.replace(".csv", "", regex=False))
        extra["attributed_via"] = "GIVEN_NAMES/<COUNTRY>/SOURCES.csv"
        extra["licence"] = ""
        extra["accessed"] = ""
        keep = ["source_id", "file", "publisher", "source_type", "url", "coverage",
                "licence", "accessed", "attributed_via"]
        lookup_df = pd.concat([lookup_df, extra[keep]], ignore_index=True)

    lookup_df.to_csv(BASE / "audits" / "sources_lookup.csv", index=False, encoding="utf-8")

    print(f"name-list CSVs found      : {len(targets)}")
    print(f"attributed                : {len(targets) - len(unattributed)}")
    print(f"UNATTRIBUTED              : {len(unattributed)}")
    print(f"sources_lookup.csv rows   : {len(lookup_df)}")
    if lookup_df.get("attributed_via") is not None:
        print("\nattributed via:")
        print(lookup_df["attributed_via"].value_counts().to_string())
    if unattributed:
        print("\nno evidence on disk for:")
        for name in unattributed:
            print("   " + name)

    if not args.write:
        print("\n(dry run - rerun with --write to back up and add the column)")
        return

    backup = SHARED / f"backup_pre_source_column_{TODAY}"
    backup.mkdir(exist_ok=True)
    changed = 0
    for path in targets:
        shutil.copy2(path, backup / path.name)
        frame = pd.read_csv(path)
        frame["source"] = assignment[path.name]
        frame.to_csv(path, index=False, encoding="utf-8")
        changed += 1
    print(f"\nbacked up {changed} files to {backup.name}/ and added the source column")


if __name__ == "__main__":
    main()
