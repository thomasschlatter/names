# -*- coding: utf-8 -*-
"""Consolidate the per-country source documentation into this folder.

PROJECTS/GIVEN_NAMES/<COUNTRY>/ is the folder this project was split out of. It
holds the name lists for the countries that have no government registry, and
most country folders carry a SOURCES.csv describing every file: what it is,
where it came from, the year range, whether it has frequencies, and often an
explicit warning about what it must not be used for.

This copies that documentation here and merges it into one table, so the
registry paper can state a source for every country instead of carrying lists
whose origin lives in another folder.

Nothing is rewritten. Where a country documents its sources in prose
(SOURCE_NOTES.txt, source_info.txt) the file is copied verbatim and the row is
marked as needing manual entry, rather than having prose parsed into fields.

QUALITY TIER
Assigned from the recorded fields only, never from an impression of the country:
  A  official statistics office / civil registry, with frequencies
  B  web aggregator (Forebears and similar), with frequencies
  C  name inventory or lexical resource, no frequencies
  D  NER / news corpus, which the source notes themselves warn is not a
     population name list
  ?  source recorded but type not classifiable from the fields

Outputs (in this folder):
  country_sources.csv          one row per data file, all countries
  source_documentation/        verbatim copies of every SOURCES/notes file
"""

from __future__ import annotations

import io
import os
import re
import shutil
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
GIVEN = Path("H:/My Drive/PROJECTS/GIVEN_NAMES")
DOCS_OUT = BASE / "source_documentation"


def tier(row) -> str:
    stype = str(row.get("source_type", "")).lower()
    has_freq = str(row.get("has_frequency", "")).strip().upper()
    if re.search(r"ner|news corpus", stype):
        return "D"
    if re.search(r"official|statistic|registry|census|government|ministry", stype):
        return "A" if has_freq.startswith("YES") else "C"
    if re.search(r"aggregator|genealog", stype):
        return "B" if has_freq.startswith("YES") else "C"
    if re.search(r"lexical|inventory|community|academic|dictionary", stype):
        return "C"
    return "?" if not stype else "C"


def main() -> None:
    DOCS_OUT.mkdir(exist_ok=True)
    countries = sorted(d for d in os.listdir(GIVEN)
                       if (GIVEN / d).is_dir() and d.isupper() and d != "ARCHIVE")

    frames, prose, missing = [], [], []
    for country in countries:
        folder = GIVEN / country
        files = os.listdir(folder)
        structured = [f for f in files if f.lower() == "sources.csv"]
        prose_files = [f for f in files
                       if re.search(r"source", f, re.I) and f.lower() != "sources.csv"]
        data_files = [f for f in files
                      if f.lower().endswith((".csv", ".txt", ".tsv"))
                      and not re.search(r"source", f, re.I)]

        for name in structured + prose_files:
            shutil.copy2(folder / name, DOCS_OUT / f"{country}__{name}")

        if structured:
            try:
                frame = pd.read_csv(folder / structured[0])
                frame.insert(0, "country", country)
                frame["doc"] = f"{country}__{structured[0]}"
                frames.append(frame)
            except Exception as exc:
                missing.append({"country": country, "issue": f"SOURCES.csv unreadable: {exc}",
                                "n_data_files": len(data_files)})
        elif prose_files:
            prose.append({"country": country, "doc": f"{country}__{prose_files[0]}",
                          "n_data_files": len(data_files),
                          "data_files": "; ".join(sorted(data_files)[:8])})
        else:
            missing.append({"country": country, "issue": "no source documentation",
                            "n_data_files": len(data_files),
                            "data_files": "; ".join(sorted(data_files)[:8])})

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(merged):
        merged["quality_tier"] = merged.apply(tier, axis=1)
        merged.to_csv(BASE / "audits" / "country_sources.csv", index=False, encoding="utf-8")

    prose_df = pd.DataFrame(prose)
    missing_df = pd.DataFrame(missing)
    if len(prose_df):
        prose_df.to_csv(BASE / "audits" / "country_sources_prose_only.csv", index=False, encoding="utf-8")
    if len(missing_df):
        missing_df.to_csv(BASE / "audits" / "country_sources_missing.csv", index=False, encoding="utf-8")

    print(f"countries: {len(countries)}")
    print(f"  structured SOURCES.csv : {len(frames)}  ({len(merged)} data files described)")
    print(f"  prose notes only       : {len(prose_df)}")
    print(f"  no documentation       : {len(missing_df)}")
    print(f"  docs copied to         : {DOCS_OUT.name}/ ({len(os.listdir(DOCS_OUT))} files)")
    if len(merged):
        print("\nquality tiers:")
        print(merged["quality_tier"].value_counts().to_string())
        print("\nby source type:")
        print(merged["source_type"].value_counts().head(10).to_string())
        print("\nfiles WITH frequency data:",
              int(merged["has_frequency"].astype(str).str.upper().str.startswith("YES").sum()))
    if len(prose_df):
        print("\nprose-only (need a manual row):")
        print(prose_df[["country", "doc", "n_data_files"]].to_string(index=False))
    if len(missing_df):
        print("\nNO documentation:")
        print(missing_df.to_string(index=False))


if __name__ == "__main__":
    main()
