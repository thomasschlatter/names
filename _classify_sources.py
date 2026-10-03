# -*- coding: utf-8 -*-
"""Classify every source in audits/sources_lookup.csv.

Adds four columns, each derived from text already recorded for that source
(publisher, source_type, coverage) plus the dated caveat documents. Nothing is
classified from an impression of the country.

  publisher_class   who published it
      national statistics office | civil registry | city or regional government
      | open data portal | private survey | web aggregator | NLP corpus
      | community or academic resource | unknown

  data_basis        what the counts are OF, where the record says so
      newborn flow | population stock | sample | unstated

  government        yes / no / unknown - is the publisher a state body

  quality_tier      A  government body, newborn-flow or full-register counts
                    B  government body, but stock / pooled / partial coverage
                    C  non-government but frequency-bearing (survey, aggregator)
                    D  no frequency data, or a corpus the source notes warn is
                       not a population name list
                    ?  not classifiable from what is recorded

Known caveats are applied from EAST_ASIA_SOURCE_STATUS_2026-05-04.md, which
records that Japan is NOT government data (Meiji Yasuda insurance survey),
Taiwan is resident stock rather than births, South Korea is pooled 2008-2019,
and China is a sparse top-name extract.

Usage: python _classify_sources.py
Rewrites audits/sources_lookup.csv with the new columns and prints a summary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
LOOKUP = BASE / "audits" / "sources_lookup.csv"

# Caveats recorded in the dated East Asia status note.
CAVEATS = {
    "japan": ("private survey", "sample", "no",
              "Meiji Yasuda insurance survey, not government (EAST_ASIA_SOURCE_STATUS)"),
    "taiwan": (None, "population stock", None,
               "resident-stock counts for 2023, not newborn flow (EAST_ASIA_SOURCE_STATUS)"),
    "southkorea": (None, "population stock", None,
                   "pooled 2008-2019, not annual (EAST_ASIA_SOURCE_STATUS)"),
    "china": (None, "unstated", None,
              "sparse official top-name extract (EAST_ASIA_SOURCE_STATUS)"),
}

PUBLISHER_RULES = [
    ("NLP corpus", r"\bNER\b|news corpus|masakhane"),
    ("web aggregator", r"forebears|genealog|aggregator|behindthename|behind the name"),
    ("private survey", r"meiji|insurance|survey company"),
    ("civil registry", r"registry of births|registro|registrar|civil regist|RENAPER|RENIEC|"
                       r"births,? deaths|family regist|vital record|state registrar|"
                       r"population regist|DOPA|registre"),
    ("national statistics office", r"statistic|statistik|statistisk|statistiek|INSTAT|ArmStat|"
                                   r"\bCSO\b|INSEE|\bONS\b|\bNSI\b|Statbel|StatCan|\bBFS\b|"
                                   r"ISTAT|\bSSB\b|\bDST\b|\bCBS\b|\bINE\b|\bNSO\b|Hagstofa|"
                                   r"bureau of statistics|census"),
    ("city or regional government", r"^stadt |city of|municipal|comune|ville de|gobierno de la ciudad|"
                                    r"province|state government|canton|prefect"),
    ("open data portal", r"open data|CKAN|Socrata|datastore|data catalogue|dataset\.gov|"
                         r"data\.gov|portal|PxWeb|PxStat|opendata"),
    ("community or academic resource", r"github|wiktionary|community|academic|lexical|"
                                       r"supplementary data|onomastic|dictionary"),
]

FLOW = r"newborn|baby|birth|born|natal|naissance|geburt"
STOCK = r"stock|population|resident|census|living"
SAMPLE = r"sample|survey|policyholder"


def classify_publisher(text: str) -> str:
    low = text.lower()
    for label, pattern in PUBLISHER_RULES:
        if re.search(pattern, low, re.I):
            return label
    return "unknown"


def classify_basis(row) -> str:
    blob = " ".join(str(row.get(c, "")) for c in
                    ("publisher", "source_type", "coverage", "file")).lower()
    if re.search(SAMPLE, blob):
        return "sample"
    if re.search(FLOW, blob):
        return "newborn flow"
    if re.search(STOCK, blob):
        return "population stock"
    return "unstated"


GOV_CLASSES = {"national statistics office", "civil registry",
               "city or regional government", "open data portal"}


def main() -> None:
    frame = pd.read_csv(LOOKUP)
    frame["publisher_class"] = frame.apply(
        lambda r: classify_publisher(f"{r.get('publisher','')} {r.get('source_type','')}"), axis=1)
    frame["data_basis"] = frame.apply(classify_basis, axis=1)
    frame["government"] = frame["publisher_class"].map(
        lambda c: "yes" if c in GOV_CLASSES else ("unknown" if c == "unknown" else "no"))
    # Preserve caveats written by other steps (e.g. needs-recollection flags);
    # blanking the column here silently discarded them on a re-run.
    if "caveat" not in frame.columns:
        frame["caveat"] = ""
    frame["caveat"] = frame["caveat"].fillna("")

    for key, (pub, basis, gov, note) in CAVEATS.items():
        mask = frame["source_id"].astype(str).str.fullmatch(key)
        if not mask.any():
            continue
        if pub:
            frame.loc[mask, "publisher_class"] = pub
        if basis:
            frame.loc[mask, "data_basis"] = basis
        if gov:
            frame.loc[mask, "government"] = gov
        frame.loc[mask, "caveat"] = note

    def tier(row) -> str:
        if row["publisher_class"] == "NLP corpus":
            return "D"
        unattributed = str(row.get("attributed_via", "")).strip() == ""
        if row["publisher_class"] == "unknown" or unattributed:
            return "?"
        if row["government"] == "yes":
            return "A" if row["data_basis"] == "newborn flow" else "B"
        if row["publisher_class"] in ("private survey", "web aggregator"):
            return "C"
        return "D"

    frame["quality_tier"] = frame.apply(tier, axis=1)
    frame.to_csv(LOOKUP, index=False, encoding="utf-8")

    print(f"sources classified: {len(frame)}\n")
    print("publisher class:")
    print(frame["publisher_class"].value_counts().to_string())
    print("\ngovernment:")
    print(frame["government"].value_counts().to_string())
    print("\ndata basis:")
    print(frame["data_basis"].value_counts().to_string())
    print("\nquality tier:")
    print(frame["quality_tier"].value_counts().sort_index().to_string())
    print("\ntier A (government, newborn flow) examples:")
    print(frame[frame.quality_tier == "A"][["source_id", "publisher"]]
          .head(8).to_string(index=False))
    print("\ncaveats applied:")
    print(frame[frame.caveat != ""][["source_id", "publisher_class", "caveat"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
