# -*- coding: utf-8 -*-
"""Attribute the outputs that no source document names.

For each unattributed official_*.csv, look for evidence in this order:
  1. the URL(s) inside the SAME function of the script that writes it
     (function-scoped, because a file-wide search picks up the neighbouring
     country's URL: Azerbaijan otherwise inherits Argentina's RENAPER link)
  2. the per-country notes copied into source_documentation/
  3. any remaining dated audit note in the shared folder

Prints what it found, and what is still unattributable, so the second group can
be re-collected rather than carried without provenance.
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

SHARED = Path("H:/My Drive/PROJECTS/_RESOURCES/official_data")
DOCS = Path(__file__).resolve().parent / "source_documentation"

URL = re.compile(r"https?://\S+")
OUT = re.compile(r"official_([a-z0-9_]+)\.csv")

MISSING = """argentina_buenos_aires azerbaijan belarus curacao czech_mv england france
france_angers france_nantes france_orleans france_paris france_saint_herblain
france_toulouse georgia hungary iceland indonesia iran kyrgyzstan morocco
new_caledonia_noumea newzealand_maori norway romania russia russia_moscow scotland
slovenia spain spain_basque_country_top100 sweden tunisia ukraine uzbekistan""".split()

# country folder name in source_documentation for outputs whose stem differs
DOC_KEY = {
    "england": "ENGLAND", "france": "FRANCE", "hungary": "HUNGARY", "iceland": "ICELAND",
    "indonesia": "INDONESIA", "norway": "NORWAY", "romania": "ROMANIA", "russia": "RUSSIA",
    "scotland": "SCOTLAND", "slovenia": "SLOVENIA", "spain": "SPAIN", "sweden": "SWEDEN",
    "ukraine": "UKRAINE", "morocco": "MOROCCO", "belarus": "BELARUS", "czech_mv": "CZECH_REPUBLIC",
    "georgia": "GEORGIA", "iran": "IRAN", "uzbekistan": "UZBEKISTAN", "curacao": "CURACAO",
}


def clean(url: str) -> str:
    return url.rstrip(").,;\"'")


def from_scripts() -> dict:
    found = {}
    for name in os.listdir(SHARED):
        if not name.endswith(".py"):
            continue
        text = io.open(SHARED / name, encoding="utf-8", errors="replace").read()
        for block in re.split(r"\ndef\s+", text):
            outs = set(OUT.findall(block))
            if not outs:
                continue
            urls = [clean(u) for u in URL.findall(block)]
            for key in outs:
                if urls and (key not in found or not found[key][1]):
                    found[key] = (name, urls[:3])
    return found


def from_declarations() -> dict:
    """Some scripts declare sources structurally, e.g.
           stem="official_france_paris", ... host="https://opendata.paris.fr"
    A function-scoped URL search misses these because the declaration sits in a
    module-level list, not inside the function that writes the file."""
    found = {}
    for name in os.listdir(SHARED):
        if not name.endswith(".py"):
            continue
        text = io.open(SHARED / name, encoding="utf-8", errors="replace").read()
        for match in re.finditer(r'stem\s*=\s*"official_([a-z0-9_]+)"(.{0,400}?)host\s*=\s*"([^"]+)"',
                                 text, re.S):
            found[match.group(1)] = (name, [clean(match.group(3))])
    return found


def from_docs() -> dict:
    found = {}
    if not DOCS.exists():
        return found
    for name in os.listdir(DOCS):
        country = name.split("__")[0]
        text = io.open(DOCS / name, encoding="utf-8", errors="replace").read()
        urls = [clean(u) for u in URL.findall(text)]
        if urls:
            found[country] = (name, urls[:3])
    return found


def main() -> None:
    scripts = from_scripts()
    declared = from_declarations()
    docs = from_docs()
    resolved, unresolved = [], []
    for key in MISSING:
        if key in declared:
            resolved.append((key, "declared in " + declared[key][0], declared[key][1][0]))
            continue
        if key in scripts and scripts[key][1]:
            resolved.append((key, "script " + scripts[key][0], scripts[key][1][0]))
            continue
        doc_key = DOC_KEY.get(key, key.upper())
        if doc_key in docs:
            resolved.append((key, "doc " + docs[doc_key][0], docs[doc_key][1][0]))
            continue
        unresolved.append(key)

    print(f"unattributed to start: {len(MISSING)}")
    print(f"  resolved            : {len(resolved)}")
    print(f"  still unattributable: {len(unresolved)}\n")
    for key, how, url in resolved:
        print(f"  {key:28s} {how[:42]:44s} {url[:70]}")
    if unresolved:
        print("\nSTILL NO SOURCE — re-collect or drop:")
        for key in unresolved:
            print("   official_" + key + ".csv")


if __name__ == "__main__":
    main()
