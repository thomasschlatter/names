# -*- coding: utf-8 -*-
"""Fold the recovered attributions into audits/sources_lookup.csv.

_find_missing_sources.py recovered a publisher URL for 28 of the 34 outputs that
no source document named, by three routes that each found what the others
missed. This writes those into the lookup, adds rows for outputs that were
absent from it entirely, and records the six that remain unattributable along
with the publisher landing page confirmed reachable on 2026-09-18, so the next
collection run knows exactly where to go.

Nothing is invented: a row is only written where a URL was found in a script, a
structured source declaration, or a per-country source note.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SHARED = Path("H:/My Drive/PROJECTS/_RESOURCES/official_data")
LOOKUP = BASE / "audits" / "sources_lookup.csv"

# Publishers confirmed reachable 2026-09-18 for the outputs with no provenance
# record anywhere. These are landing pages, NOT the file that was used, so the
# row is marked needs_recollection: the data on disk cannot be traced to them.
KNOWN_PUBLISHER = {
    "england": ("Office for National Statistics (ONS)",
                "https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/livebirths"),
    "france": ("INSEE, Fichier des prenoms",
               "https://www.insee.fr/fr/statistiques/7633685"),
    "sweden": ("Statistics Sweden (SCB)",
               "https://api.scb.se/OV0104/v1/doris/en/ssd/BE/BE0001"),
    "slovenia": ("Statistical Office of the Republic of Slovenia (SURS)",
                 "https://pxweb.stat.si/SiStatData/api/v1/en/Data"),
    "ukraine": ("data.gov.ua open data portal",
                "https://data.gov.ua"),
    "belarus": ("National Statistical Committee of Belarus (Belstat)",
                "https://www.belstat.gov.by"),
}


def recovered() -> dict:
    """Run the finder and parse its resolved lines."""
    result = subprocess.run([sys.executable, str(BASE / "_find_missing_sources.py")],
                            capture_output=True, text=True, encoding="utf-8")
    found = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s{2}([a-z0-9_]+)\s{2,}(\S.*?)\s{2,}(https?://\S+)", line)
        if match:
            found[match.group(1)] = (match.group(2).strip(), match.group(3).strip())
    return found


def main() -> None:
    lookup = pd.read_csv(LOOKUP)
    by_id = {str(r): i for i, r in enumerate(lookup["source_id"])}
    found = recovered()
    print(f"recovered attributions parsed: {len(found)}")

    added, updated = 0, 0
    new_rows = []
    all_outputs = sorted(p.stem.replace("official_", "") for p in SHARED.glob("official_*.csv"))
    for source_id in all_outputs:
        url, via, publisher, needs = "", "", "", ""
        if source_id in found:
            via, url = found[source_id]
        elif source_id in KNOWN_PUBLISHER:
            publisher, url = KNOWN_PUBLISHER[source_id]
            via = "publisher confirmed reachable 2026-09-18"
            needs = "yes - landing page only, the file on disk is not traceable to it"
        if source_id in by_id:
            index = by_id[source_id]
            if url and not str(lookup.at[index, "url"]).startswith("http"):
                lookup.at[index, "url"] = url
                lookup.at[index, "attributed_via"] = via
                if publisher:
                    lookup.at[index, "publisher"] = publisher
                if needs:
                    lookup.at[index, "caveat"] = needs
                updated += 1
        elif url:
            new_rows.append({"source_id": source_id,
                             "file": f"official_{source_id}.csv",
                             "publisher": publisher, "url": url,
                             "attributed_via": via, "caveat": needs})
            added += 1

    if new_rows:
        lookup = pd.concat([lookup, pd.DataFrame(new_rows)], ignore_index=True)
    lookup = lookup.sort_values("source_id")
    lookup.to_csv(LOOKUP, index=False, encoding="utf-8")

    has_url = lookup["url"].astype(str).str.startswith("http").sum()
    print(f"rows updated with a URL : {updated}")
    print(f"rows added to the lookup: {added}")
    print(f"lookup rows             : {len(lookup)}")
    print(f"rows carrying a real URL: {has_url} ({100*has_url/len(lookup):.0f}%)")
    flagged = lookup[lookup.get("caveat", "").astype(str).str.startswith("yes -")]
    if len(flagged):
        print(f"\nneed re-collection ({len(flagged)}):")
        print(flagged[["source_id", "publisher"]].to_string(index=False))


if __name__ == "__main__":
    main()
