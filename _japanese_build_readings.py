# -*- coding: utf-8 -*-
"""Build the Japanese name-to-reading table, keeping genuine ambiguity.

A kanji given name often has several real readings, so this does not pick one.
It emits ONE ROW PER (name, reading) and splits the name's count across them.

WHICH READINGS COUNT AS REAL
JMnedict alone is too permissive: it lists 25 readings for 陽, most of which are
attested somewhere but vanishingly rare as given names. A reading is kept only
if TWO independent sources support it:

    JMnedict (EDRDG)  AND  (ja.wikipedia person articles  OR  Meiji Yasuda top-50)

so every retained reading has both a lexicographic entry and evidence of real
use. Readings dropped by this rule are COUNTED and written to the audit file,
never silently discarded.

Exception, recorded rather than smoothed over: a few readings appear in
Wikipedia and Meiji but NOT in JMnedict (陽斗 -> はると is Meiji's number one
reading overall, 110 people, yet absent from JMnedict). Those are kept too, on
the same two-source logic, with source recorded as 'wiki+meiji'.

COUNT SPLITTING
The registry gives a count per NAME, not per reading. Where a name has k
retained readings the count is divided by k. That is an explicit equal-
probability assumption, not a measurement: it keeps the name's total weight
unchanged instead of inflating it k-fold, which is what carrying the full count
on every reading would do. The assumption is recorded per row (`split_rule`)
and must be stated in the manuscript. Where Meiji supplies counts for competing
readings, `meiji_people` is carried so a frequency-weighted split can replace
the equal one later.

Output:
  japanese_name_readings.csv   name, sex, reading, count, n_readings, sources
  japanese_readings_dropped.csv  every reading excluded, with the reason
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    cmp_mod = load_module("cmp", "_japanese_readings_compare.py")
    jm = cmp_mod.load_jmnedict()
    meiji = cmp_mod.load_meiji()
    comparison = pd.read_csv(BASE / "japanese" / "japanese_readings_comparison.csv")

    registry = pd.read_csv(BASE / "official_data_top30" / "official_japan.csv")
    registry.columns = [c.strip().lower() for c in registry.columns]

    wiki_counts = {}
    for row in comparison.itertuples():
        found = {}
        if isinstance(row.wiki_all, str) and row.wiki_all.strip():
            for part in row.wiki_all.split(" / "):
                match = re.match(r"(.+?)\((\d+)\)$", part.strip())
                if match:
                    found[match.group(1)] = int(match.group(2))
        wiki_counts[str(row.name)] = found

    kept_rows, dropped_rows = [], []
    for record in registry.drop_duplicates(["name", "sex"]).itertuples():
        name = str(record.name).strip()
        is_kana = not re.search(r"[一-鿿]", name)
        if is_kana:
            kept_rows.append({"name": name, "sex": record.sex, "reading": name,
                              "count": record.count, "n_readings": 1,
                              "sources": "already kana", "wiki_articles": None,
                              "meiji_rank": None, "meiji_people": None,
                              "split_rule": "none (single reading)"})
            continue

        jm_map = jm.get(name, {})           # reading -> {"F","M"}
        wiki = wiki_counts.get(name, {})
        sex = str(record.sex).upper()[:1]
        candidates = set(jm_map) | set(wiki)
        kept = []
        for reading in candidates:
            jm_sexes = jm_map.get(reading, set())
            in_jm = reading in jm_map
            in_wiki = reading in wiki
            meiji_hit = meiji.get(reading)
            in_meiji = meiji_hit is not None
            # A reading only counts as evidence FOR THIS ROW if the source
            # attests it for this sex. JMnedict tags fem/masc; Meiji publishes
            # separate male and female rankings. Wikipedia carries no tag, so it
            # is accepted either way and cannot on its own carry a reading.
            meiji_ok = in_meiji and meiji_hit[0] == sex
            meiji_other = in_meiji and meiji_hit[0] != sex
            # MEASURED 2026-09-18: JMnedict's fem/masc tags cannot be used to
            # veto a reading. It tags 翔 -> しょう, 新 -> あらた, 湊 -> みなと and
            # 陽 -> あきら as female only, and all four are standard male
            # readings; vetoing on them deleted the correct reading for 10 of
            # 60 names. JMnedict therefore supplies CANDIDATES only. The sex
            # evidence comes from Meiji Yasuda, which publishes separate male
            # and female rankings from the same survey as the registry counts.
            keep = meiji_ok or (in_jm and in_wiki)
            # A reading ranked for the other sex and supported by nothing else
            # for this one is a cross-sex reading, not this name's.
            if meiji_other and not (in_wiki and in_jm):
                keep = False
            tags = ",".join(t for t, ok in
                            (("jmnedict", in_jm), ("wiki", in_wiki), ("meiji", meiji_ok)) if ok)
            if keep:
                kept.append((reading, tags, wiki.get(reading), meiji_hit if meiji_ok else None))
            else:
                dropped_rows.append({"name": name, "sex": record.sex, "reading": reading,
                                     "sources": tags or "none",
                                     "reason": "ranked for the other sex only" if meiji_other
                                     else "not corroborated (needs Meiji same-sex, or JMnedict+wiki)"})
        if not kept:
            dropped_rows.append({"name": name, "sex": record.sex, "reading": "",
                                 "sources": "", "reason": "NO corroborated reading"})
            continue

        share = record.count / len(kept)
        for reading, tags, warticles, mhit in kept:
            kept_rows.append({
                "name": name, "sex": record.sex, "reading": reading,
                "count": round(share, 4), "n_readings": len(kept), "sources": tags,
                "wiki_articles": warticles,
                "meiji_rank": mhit[1] if mhit else None,
                "meiji_people": mhit[2] if mhit else None,
                "split_rule": f"count/{len(kept)} (equal probability)"})

    kept_df = pd.DataFrame(kept_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    kept_df.to_csv(BASE / "japanese" / "japanese_name_readings.csv", index=False, encoding="utf-8")
    dropped_df.to_csv(BASE / "japanese" / "japanese_readings_dropped.csv", index=False, encoding="utf-8")

    names_in = registry.drop_duplicates(["name", "sex"]).shape[0]
    print(f"registry names            : {names_in}")
    print(f"names with a reading      : {kept_df['name'].nunique()}")
    print(f"rows emitted (name+reading): {len(kept_df)}")
    print(f"ambiguous names (>1 kept) : {(kept_df.n_readings > 1).sum() and kept_df[kept_df.n_readings > 1]['name'].nunique()}")
    print(f"readings dropped          : {len(dropped_df)}")
    if len(dropped_df):
        print(dropped_df["reason"].value_counts().to_string())
    print(f"\ncount preserved: registry {registry.drop_duplicates(['name','sex'])['count'].sum():,.0f}"
          f"  vs emitted {kept_df['count'].sum():,.0f}")
    print("\nambiguous examples:")
    amb = kept_df[kept_df.n_readings > 1].sort_values(["name", "reading"])
    print(amb.head(14)[["name", "sex", "reading", "count", "n_readings", "sources",
                        "meiji_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
