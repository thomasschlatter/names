# -*- coding: utf-8 -*-
"""Japanese name readings with sex taken from REAL PEOPLE, via Wikidata.

For each registry name, find ja.wikipedia articles whose title ends with that
given name, read the reading out of the lead gloss ("岩田 陽葵（いわた はるき、"),
then take each subject's sex from their Wikidata item (P21). The result is, per
reading, a count of how many real men and how many real women bear it.

WHY THIS REPLACES JMNEDICT'S GENDER TAGS
JMnedict tags 翔 -> しょう, 新 -> あらた, 湊 -> みなと and 陽 -> あきら as female
only; all four are standard male readings, and vetoing on those tags deleted the
correct reading for 10 of 60 names. Wikidata P21 on the actual article subjects
gets them right: 翔 -> しょう is 9 men and 0 women, 陽 -> あきら is 2 men, while
陽 -> あき is 1 woman.

WHAT THIS IS AND IS NOT
It is an observed sex distribution over notable people who bear the name. It is
not a birth-cohort distribution: Wikipedia's population is older, more male, and
selected for notability. So it is used to ASSIGN A READING TO A SEX, which it
does well, and not to estimate how common a reading is among babies, which it
would do badly. Frequency comes from Meiji Yasuda instead.

Output: japanese_wikidata_readings.csv
    name, reading, people_M, people_F, people_unknown, example_articles
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
HEADERS = {"User-Agent": "given-name-research/1.0 (academic; contact twangbang@gmail.com)"}
JA_API = "https://ja.wikipedia.org/w/api.php"
WD_API = "https://www.wikidata.org/w/api.php"
KANA = r"[ぁ-ゟ]"
SEX_ITEM = {"Q6581097": "M", "Q6581072": "F",
            "Q1097630": "I", "Q1052281": "F", "Q2449503": "M"}  # incl. trans items


def person_titles(name: str) -> list[str]:
    payload = requests.get(JA_API, params={
        "action": "query", "list": "search", "srsearch": f"intitle:{name}",
        "srlimit": 30, "format": "json", "formatversion": "2"},
        headers=HEADERS, timeout=45).json()
    return [hit["title"] for hit in payload.get("query", {}).get("search", [])
            if hit["title"].endswith(name)]


def readings_with_items(titles: list[str]) -> list[tuple]:
    out = []
    for start in range(0, len(titles), 20):
        chunk = titles[start:start + 20]
        payload = requests.get(JA_API, params={
            "action": "query", "prop": "extracts|pageprops", "exintro": "1",
            "explaintext": "1", "ppprop": "wikibase_item",
            "titles": "|".join(chunk), "format": "json", "formatversion": "2"},
            headers=HEADERS, timeout=45).json()
        for page in payload.get("query", {}).get("pages", []):
            match = re.search(r"（(" + KANA + r"+)[ 　]+(" + KANA + r"+)[、,]",
                              page.get("extract", ""))
            qid = (page.get("pageprops") or {}).get("wikibase_item")
            if match and qid:
                out.append((match.group(2), qid, page["title"]))
    return out


def sex_of(qids: list[str]) -> dict:
    found = {}
    for start in range(0, len(qids), 50):
        chunk = qids[start:start + 50]
        payload = requests.get(WD_API, params={
            "action": "wbgetentities", "ids": "|".join(chunk),
            "props": "claims", "format": "json"}, headers=HEADERS, timeout=45).json()
        for qid, entity in (payload.get("entities") or {}).items():
            claims = (entity.get("claims") or {}).get("P21") or []
            if not claims:
                continue
            value = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value", {})
            found[qid] = SEX_ITEM.get(value.get("id"), "?")
    return found


def main() -> None:
    registry = pd.read_csv(BASE / "official_data_top30" / "official_japan.csv")
    registry.columns = [c.strip().lower() for c in registry.columns]
    names = sorted({str(n).strip() for n in registry["name"]})

    rows = []
    for index, name in enumerate(names, 1):
        if not re.search(r"[一-鿿]", name):
            continue
        try:
            titles = person_titles(name)
            found = readings_with_items(titles) if titles else []
            sexes = sex_of([qid for _, qid, _ in found])
        except Exception as exc:
            print(f"   {name}: ERROR {type(exc).__name__}")
            continue
        tally: dict[str, dict] = {}
        for reading, qid, title in found:
            bucket = tally.setdefault(reading, {"M": 0, "F": 0, "?": 0, "eg": []})
            bucket[sexes.get(qid, "?") if sexes.get(qid) in ("M", "F") else "?"] += 1
            if len(bucket["eg"]) < 3:
                bucket["eg"].append(title)
        for reading, bucket in tally.items():
            rows.append({"name": name, "reading": reading,
                         "people_M": bucket["M"], "people_F": bucket["F"],
                         "people_unknown": bucket["?"],
                         "example_articles": " / ".join(bucket["eg"])})
        print(f"   {index:3d}/{len(names)} {name:6s} articles={len(titles):3d} "
              f"readings={len(tally)}")
        time.sleep(0.35)

    out = pd.DataFrame(rows)
    out.to_csv(BASE / "japanese" / "japanese_wikidata_readings.csv", index=False, encoding="utf-8")
    print(f"\n{len(out)} name+reading pairs from {out['name'].nunique()} names")
    print(f"people tagged M: {out.people_M.sum()}, F: {out.people_F.sum()}, "
          f"unknown: {out.people_unknown.sum()}")


if __name__ == "__main__":
    main()
