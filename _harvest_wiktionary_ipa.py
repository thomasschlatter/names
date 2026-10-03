# -*- coding: utf-8 -*-
"""Recover human IPA for registry names that WikiPron's snapshot does not cover.

WikiPron ships a frozen scrape of the English Wiktionary. Two things it misses:
entries added since, and entries that exist only on a language's own Wiktionary
edition (cs.wiktionary, pl.wiktionary, ...). This queries the live MediaWiki API
for names with no usable WikiPron entry.

Every transcription is attributed to a LANGUAGE SECTION of the page, because one
page carries several languages: en.wiktionary "Sophie" has /ˈsəʊfi/ under
English and /zoˈfiː/ under German. Taking the first IPA on the page would import
the wrong language's pronunciation.

Output: wiktionary_name_ipa.csv with source, name, wiki, language, ipa, url
so every recovered transcription can be traced back to the page it came from.

Usage:
    python _harvest_wiktionary_ipa.py --limit 200      sample, to measure yield
    python _harvest_wiktionary_ipa.py                  everything missing
"""

from __future__ import annotations

import argparse
import html
import io
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
SUBSET = BASE / "official_data_top30"
WP = Path("H:/My Drive/PROJECTS/_RESOURCES/LEXICONS/wikipron/wikipron-master/data/scrape/tsv")
UA = "given-name-research/1.0 (academic; contact twangbang@gmail.com)"
HEADERS = {"User-Agent": UA}

# espeak voice -> (WikiPron ISO, Wiktionary language-section name, wiki editions to try)
LANG_INFO = {
    "en": ("eng", "English", ["en"]),
    "de": ("deu", "German", ["en", "de"]),
    "es": ("spa", "Spanish", ["en", "es"]),
    "fr": ("fra", "French", ["en", "fr"]),
    "cs": ("ces", "Czech", ["en", "cs"]),
    "pl": ("pol", "Polish", ["en", "pl"]),
    "it": ("ita", "Italian", ["en", "it"]),
    "nl": ("nld", "Dutch", ["en", "nl"]),
    "pt": ("por", "Portuguese", ["en", "pt"]),
    "pt-br": ("por", "Portuguese", ["en", "pt"]),
    "sv": ("swe", "Swedish", ["en", "sv"]),
    "da": ("dan", "Danish", ["en", "da"]),
    "nb": ("nob", "Norwegian", ["en", "no"]),
    "fi": ("fin", "Finnish", ["en", "fi"]),
    "et": ("est", "Estonian", ["en", "et"]),
    "hu": ("hun", "Hungarian", ["en", "hu"]),
    "ro": ("ron", "Romanian", ["en", "ro"]),
    "hr": ("hbs", "Serbo-Croatian", ["en"]),
    "sl": ("slv", "Slovene", ["en", "sl"]),
    "sk": ("slk", "Slovak", ["en", "sk"]),
    "bg": ("bul", "Bulgarian", ["en", "bg"]),
    "ru": ("rus", "Russian", ["en", "ru"]),
    "uk": ("ukr", "Ukrainian", ["en", "uk"]),
    "lt": ("lit", "Lithuanian", ["en", "lt"]),
    "lv": ("lav", "Latvian", ["en", "lv"]),
    "is": ("isl", "Icelandic", ["en", "is"]),
    "fo": ("fao", "Faroese", ["en"]),
    "tr": ("tur", "Turkish", ["en", "tr"]),
    "ca": ("cat", "Catalan", ["en", "ca"]),
    "eu": ("eus", "Basque", ["en", "eu"]),
    "sq": ("sqi", "Albanian", ["en"]),
    "mk": ("mkd", "Macedonian", ["en", "mk"]),
    "id": ("ind", "Indonesian", ["en", "id"]),
    "mi": ("mri", "Maori", ["en"]),
    "az": ("aze", "Azerbaijani", ["en", "az"]),
    "hy": ("hye", "Armenian", ["en", "hy"]),
    "ka": ("kat", "Georgian", ["en", "ka"]),
    "kk": ("kaz", "Kazakh", ["en", "kk"]),
    "ky": ("kir", "Kyrgyz", ["en", "ky"]),
    "uz": ("uzb", "Uzbek", ["en", "uz"]),
    "be": ("bel", "Belarusian", ["en", "be"]),
    "fa": ("fas", "Persian", ["en", "fa"]),
    "kl": ("kal", "Greenlandic", ["en"]),
    "af": ("afr", "Afrikaans", ["en", "af"]),
}
IPA_SPAN = re.compile(r'<span class="IPA[ "][^>]*>(.*?)</span>', re.S)
TAGS = re.compile(r"<[^>]+>")


# Scripts with no upper/lower distinction. The capitalisation test below cannot
# work for these: applying it universally reported 0 usable entries for Georgian,
# Persian, Hebrew, Arabic, Japanese, Korean and Chinese when they in fact had 211
# between them, and sent every one of those names on a pointless lookup.
CASELESS_ISO = {"kat", "fas", "heb", "ara", "cmn", "jpn", "kor", "tha", "hye"}


def usable_wikipron(iso: str) -> set:
    """Names already covered by the WikiPron snapshot.

    For a cased script, only a capitalised entry counts: a lowercase-only entry
    is the common word, not the name (Australian "Willow", "Ivy", "Ruby").
    For a caseless script every entry counts, because case cannot disambiguate.
    """
    out = set()
    if not iso:
        return out
    caseless = iso in CASELESS_ISO
    for filename in os.listdir(WP):
        if not (filename.startswith(iso + "_") and filename.endswith(".tsv")):
            continue
        if "filtered" in filename:
            continue
        with io.open(WP / filename, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split("\t")
                if len(parts) == 2 and (caseless or parts[0][:1].isupper()):
                    out.add(parts[0].lower())
    return out


def page_ipa(title: str, wiki: str, section_name: str) -> list[str]:
    """IPA strings under the requested language section only."""
    response = requests.get(
        f"https://{wiki}.wiktionary.org/w/api.php",
        params={"action": "parse", "page": title, "prop": "text",
                "format": "json", "formatversion": "2", "maxlag": "5"},
        headers=HEADERS, timeout=45,
    )
    if response.status_code != 200:
        return []
    payload = response.json()
    if "error" in payload:
        return []
    text = payload.get("parse", {}).get("text", "")
    if not isinstance(text, str):
        return []
    # Split on level-2 headings, which separate languages on every edition.
    chunks = re.split(r"<h2[^>]*>", text)
    for chunk in chunks[1:]:
        heading = TAGS.sub("", chunk.split("</h2>")[0])
        if section_name.lower() not in heading.strip().lower():
            continue
        found = []
        for raw in IPA_SPAN.findall(chunk):
            value = html.unescape(TAGS.sub("", raw)).strip()
            if not value.startswith(("/", "[")) or len(value) <= 2:
                continue
            # Wiktionary lists partial respellings for shared stems, e.g. Susan
            # "/ˈsjuː-/". A fragment is not a transcription of the whole name.
            if value.rstrip("/]").endswith("-") or value.lstrip("/[").startswith("-"):
                continue
            found.append(value)
        return found
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many lookups (yield test)")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    text = io.open(BASE / "paper_registry.Rmd", encoding="utf-8").read()
    lang_map = dict(re.findall(
        r'"(official_[a-z0-9_]+)":\s*\("([^"]*)"',
        re.search(r"LANG_BY_SOURCE = \{(.*?)\n\}", text, re.S).group(1)))

    covered_cache: dict[str, set] = {}
    todo = []
    for path in sorted(SUBSET.glob("official_*.csv")):
        source = path.stem
        voice = lang_map.get(source)
        info = LANG_INFO.get(voice)
        if not info:
            continue
        iso, section, wikis = info
        if iso not in covered_cache:
            # Keep every language cached. Only capitalised entries are retained,
            # so each set is small, and sources are alphabetical which means
            # languages interleave: clearing per language re-reads the large
            # scrape files dozens of times.
            covered_cache[iso] = usable_wikipron(iso)
            print(f"  reference loaded: {iso} ({len(covered_cache[iso]):,} name-like entries)")
        covered = covered_cache[iso]
        for name in sorted({str(n).strip() for n in pd.read_csv(path)["name"]}):
            if name.lower() in covered:
                continue
            todo.append((source, name, section, wikis))

    # Work through the richest languages first: how much Wiktionary holds for a
    # language varies enormously (German 37,033 name-like entries against Uzbek
    # 25), so this ordering makes a partial run maximally useful.
    richness = {iso: len(names) for iso, names in covered_cache.items()}
    iso_of = {voice: info[0] for voice, info in LANG_INFO.items()}
    voice_of_source = {s: lang_map.get(s) for s, *_ in
                       [(x[0],) for x in todo]}
    todo.sort(key=lambda item: -richness.get(
        iso_of.get(voice_of_source.get(item[0]), ""), 0))

    print(f"names with no usable WikiPron entry: {len(todo):,}")
    if args.limit:
        step = max(1, len(todo) // args.limit)
        todo = todo[::step][:args.limit]
        print(f"sampling {len(todo)} of them, evenly spread across sources")

    # VARIANT RULE, applied here and to be stated in the manuscript:
    # every variant Wiktionary lists is kept, in page order, with variant_rank.
    # `preferred` marks the one variant to use when a single value is needed:
    # the first phonemic transcription (/slashes/) if any, otherwise the first
    # phonetic one ([brackets]). Nothing is averaged and nothing is discarded,
    # so a reader can see the alternatives that were not used.
    rows, hits = [], 0
    for index, (source, name, section, _wikis) in enumerate(todo, 1):
        try:
            found = page_ipa(name, "en", section)
        except Exception:
            found = []
        if found:
            hits += 1
            phonemic = [v for v in found if v.startswith("/")]
            preferred = phonemic[0] if phonemic else found[0]
            for rank, value in enumerate(found, 1):
                rows.append({"source": source.replace("official_", ""), "name": name,
                             "wiki": "en.wiktionary", "language": section,
                             "ipa": value, "variant_rank": rank,
                             "preferred": value == preferred,
                             "url": f"https://en.wiktionary.org/wiki/{name}"})
        time.sleep(args.sleep)
        if index % 50 == 0:
            print(f"  {index}/{len(todo)} looked up, {hits} with IPA "
                  f"({100 * hits / index:.1f}%)")
            pd.DataFrame(rows).to_csv(BASE / "audits" / "wiktionary_name_ipa.csv",
                                      index=False, encoding="utf-8")

    out = pd.DataFrame(rows)
    out.to_csv(BASE / "audits" / "wiktionary_name_ipa.csv", index=False, encoding="utf-8")
    print(f"\nlooked up {len(todo):,} names; {hits:,} gained a transcription "
          f"({100 * hits / max(len(todo), 1):.1f}%)")
    if len(out):
        print(f"{len(out):,} IPA strings written to wiktionary_name_ipa.csv")
        print(out.groupby("wiki").size().to_string())
        print("\nsample:")
        print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
