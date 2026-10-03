# -*- coding: utf-8 -*-
"""Collect Japanese given-name readings from three independent sources, compare.

The registry stores kanji; the analysis needs a pronunciation. Kanji given-name
readings are irregular, so no G2P can derive them: espeak -v ja returns the
English words "Chinese letter" for every one. The reading has to come from a
source, and this gathers every source available, then checks whether they agree.

  1. JMnedict (EDRDG, CC BY-SA)  - 743,662 proper-name entries, readings tagged
     fem / masc / given. Complete but UNRANKED: it lists every attested reading
     with no indication which is common.
  2. ja.wikipedia person articles - the lead glosses a name as
     "岩田 陽葵（いわた はるき、...", so the second kana token is the given-name
     reading. Counting articles gives a crude frequency, but the population is
     notable people, not babies.
  3. Meiji Yasuda 読み方ベスト50   - the top 50 readings per sex WITH counts and
     percentages, from the same survey the registry counts come from. Ranked,
     but it is a list of readings, not a kanji-to-reading mapping, so it can
     only adjudicate between candidates, not supply one.

Output: japanese_readings_comparison.csv
"""

from __future__ import annotations

import html
import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
SCRATCH = Path("C:/Users/twang/AppData/Local/Temp/claude/"
               "H--My-Drive-PROJECTS-NAME-PHONOLOGY-REGISTRY/"
               "0b86430e-87d2-46f4-b167-c6fdb6c94b9c/scratchpad")
HEADERS = {"User-Agent": "given-name-research/1.0 (academic; contact twangbang@gmail.com)"}
JA_API = "https://ja.wikipedia.org/w/api.php"
KANA = r"[\u3041-\u309F]"
TAGS = re.compile(r"<[^>]+>")


def kata_to_hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "\u30a1" <= c <= "\u30f6" else c for c in text)


def load_jmnedict() -> dict:
    path = SCRATCH / "JMnedict.xml"
    if not path.exists():
        return {}
    raw = io.open(path, encoding="utf-8").read()
    index: dict[str, list] = {}
    for entry in re.findall(r"<entry>(.*?)</entry>", raw, re.S):
        types = re.findall(r"<name_type>&(.*?);</name_type>", entry)
        if not ({"fem", "masc", "given"} & set(types)):
            continue
        readings = re.findall(r"<reb>(.*?)</reb>", entry)
        # Keep the gender tag with the reading. Pooling them assigns female
        # readings to male names: 陽 (male, n=291) otherwise picks up ひまり and
        # あかり, and 新 (male) picks up さら while losing あらた.
        sexes = set()
        if "fem" in types:
            sexes.add("F")
        if "masc" in types:
            sexes.add("M")
        if "given" in types and not sexes:
            sexes = {"F", "M"}
        for kanji in re.findall(r"<keb>(.*?)</keb>", entry):
            index.setdefault(kanji, {})
            for reading in readings:
                index[kanji].setdefault(reading, set()).update(sexes)
    return index


def load_meiji() -> dict:
    counts: dict[str, tuple] = {}
    for sex, filename in (("M", "meiji_read50_M.html"), ("F", "meiji_read50_F.html")):
        path = SCRATCH / filename
        if not path.exists():
            continue
        text = io.open(path, encoding="utf-8").read()
        rank = 0
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S):
            # The table has a leading spacer cell holding &nbsp;. Unescape before
            # filtering, or it survives as a non-empty cell and shifts every
            # column index by one (which silently keyed the rank as the reading).
            cells = [html.unescape(TAGS.sub("", c)).strip()
                     for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
            cells = [c for c in cells if c and c != " "]
            if len(cells) < 3 or cells[0] == "順位":
                continue
            position, reading, people = cells[0], cells[1], cells[2]
            digits = re.sub(r"[^\d]", "", position)
            if digits:
                rank = int(digits)
            n = re.sub(r"[^\d]", "", people)
            counts[kata_to_hira(reading)] = (sex, rank, int(n) if n else None)
    return counts


def wikipedia_readings(name: str) -> dict:
    search = requests.get(JA_API, params={
        "action": "query", "list": "search", "srsearch": f"intitle:{name}",
        "srlimit": 10, "format": "json", "formatversion": "2"},
        headers=HEADERS, timeout=45).json()
    titles = [x["title"] for x in search.get("query", {}).get("search", [])
              if x["title"].endswith(name)]
    if not titles:
        return {}
    extracts = requests.get(JA_API, params={
        "action": "query", "prop": "extracts", "exintro": "1", "explaintext": "1",
        "titles": "|".join(titles[:8]), "format": "json", "formatversion": "2"},
        headers=HEADERS, timeout=45).json()
    found: dict[str, int] = {}
    for page in extracts.get("query", {}).get("pages", []):
        text = page.get("extract", "")
        match = re.search(r"（(" + KANA + r"+)[ \u3000]+(" + KANA + r"+)[、,]", text)
        if match:
            found[match.group(2)] = found.get(match.group(2), 0) + 1
    return found


def main() -> None:
    jm = load_jmnedict()
    meiji = load_meiji()
    print(f"JMnedict given-name entries: {len(jm):,}")
    print(f"Meiji Yasuda ranked readings: {len(meiji)}")

    frame = pd.read_csv(BASE / "official_data_top30" / "official_japan.csv")
    frame.columns = [c.strip().lower() for c in frame.columns]
    rows = []
    for record in frame.drop_duplicates(["name", "sex"]).itertuples():
        name = str(record.name).strip()
        has_kanji = bool(re.search(r"[\u4e00-\u9fff]", name))
        wiki = {}
        if has_kanji:
            try:
                wiki = wikipedia_readings(name)
            except Exception:
                wiki = {}
            time.sleep(0.3)
        wiki_ranked = sorted(wiki.items(), key=lambda x: -x[1])
        wiki_top = wiki_ranked[0][0] if wiki_ranked else ("" if has_kanji else name)
        jm_list = jm.get(name, [])
        my_hit = meiji.get(wiki_top)
        rows.append({
            "name": name, "sex": record.sex, "count": record.count,
            "wiki_top": wiki_top,
            "wiki_articles": wiki_ranked[0][1] if wiki_ranked else 0,
            "wiki_all": " / ".join(f"{k}({v})" for k, v in wiki_ranked),
            "jmnedict_n": len(jm_list),
            "jmnedict_all": " / ".join(jm_list[:6]),
            "wiki_in_jmnedict": wiki_top in jm_list if (wiki_top and jm_list) else None,
            "meiji_rank": my_hit[1] if my_hit else None,
            "meiji_people": my_hit[2] if my_hit else None,
        })
        print(f"   {name:6s} wiki={wiki_top or '-':8s} "
              f"jm={len(jm_list):2d} meiji={my_hit[1] if my_hit else '-'}")

    out = pd.DataFrame(rows)
    out.to_csv(BASE / "japanese" / "japanese_readings_comparison.csv", index=False, encoding="utf-8")
    n = len(out)
    print(f"\nnames: {n}")
    print(f"  wikipedia supplied a reading : {(out.wiki_top != '').sum()}")
    print(f"  JMnedict has the name        : {(out.jmnedict_n > 0).sum()}")
    print(f"  wiki reading is in JMnedict  : {int(out.wiki_in_jmnedict.fillna(False).sum())}"
          f" of {int(out.wiki_in_jmnedict.notna().sum())} comparable")
    print(f"  wiki reading in Meiji top-50 : {int(out.meiji_rank.notna().sum())}")
    print(f"  JMnedict ambiguous (>1 reading): {(out.jmnedict_n > 1).sum()}")


if __name__ == "__main__":
    main()
