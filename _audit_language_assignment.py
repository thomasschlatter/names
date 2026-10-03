# -*- coding: utf-8 -*-
"""Audit every registry's language assignment against its actual names.

Two independent signals, neither of which invents a name's origin:

1. SCRIPT. The dominant Unicode script of the registry's names against the
   script the assigned voice expects. A Cyrillic registry read by a Latin voice
   is a defect regardless of which language is right.

2. AGREEMENT. The assigned transcriber scored against the WikiPron human
   reference for the assigned language, on the registry's own names, with
   SYMMETRIC normalisation (tone and stress marks, length, and case removed from
   both sides). Without that normalisation a good transcriber looks terrible:
   the Croatian check scored 0% raw and 92.3% normalised, purely because the
   reference marks pitch accent.

Low agreement is a FLAG, not a verdict. It can mean the wrong language, or it
can mean the reference is thin or narrow-transcribed. Sources with no reference
overlap are reported as unscored rather than silently passed.

Usage: python _audit_language_assignment.py
Writes language_assignment_audit.csv
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import unicodedata
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SUBSET = BASE / "official_data_top30"
WP = Path("H:/My Drive/PROJECTS/_RESOURCES/LEXICONS/wikipron/wikipron-master/data/scrape/tsv")
ESPEAK = Path(r"C:\Program Files\eSpeak NG\espeak-ng.exe")

# espeak voice -> WikiPron ISO-639-3. None means no reference is expected.
ISO = {
    "sq": "sqi", "es": "spa", "hy": "hye", "en": "eng", "de": "deu", "az": "aze",
    "be": "bel", "nl": "nld", "hr": "hbs", "pt-br": "por", "bg": "bul", "fr": "fra",
    "zh": "cmn", "cs": "ces", "da": "dan", "et": "est", "fo": "fao", "fi": "fin",
    "kl": "kal", "ka": "kat", "hu": "hun", "is": "isl", "id": "ind", "fa": "fas",
    "he": "heb", "it": "ita", "ja": "jpn", "kk": "kaz", "ky": "kir", "lv": "lav",
    "lt": "lit", "ar": "ara", "mi": "mri", "mk": "mkd", "nb": "nob", "pl": "pol",
    "pt": "por", "ro": "ron", "ru": "rus", "sk": "slk", "sl": "slv", "af": "afr",
    "ko": "kor", "ca": "cat", "eu": "eus", "sv": "swe", "tr": "tur", "uk": "ukr",
    "uz": "uzb", "pap": None, "poly": None,
}
# epitran code -> WikiPron ISO, for sources routed to epitran first.
EPI_ISO = {"fra-Latn": "fra", "kor-Hang": "kor", "tgl-Latn": "tgl", "zul-Latn": "zul"}

# Script each voice expects. Used only for the mismatch check.
SCRIPT = {
    "hy": "ARMENIAN", "ka": "GEORGIAN", "zh": "HAN", "ja": "HAN", "ko": "HANGUL",
    "he": "HEBREW", "ar": "ARABIC", "fa": "ARABIC", "ru": "CYRILLIC",
    "uk": "CYRILLIC", "be": "CYRILLIC", "bg": "CYRILLIC", "mk": "CYRILLIC",
    "kk": "CYRILLIC", "ky": "CYRILLIC", "sr": "CYRILLIC",
}
TONE_MARKS = {0x300, 0x301, 0x302, 0x303, 0x304, 0x306, 0x307, 0x308, 0x30B,
              0x30C, 0x30F, 0x311, 0x340, 0x341}
STRESS = "\u02c8\u02cc\u203f\u0361\u1d7b"


def rmd_lang_map() -> tuple[dict, set]:
    text = io.open(BASE / "paper_registry.Rmd", encoding="utf-8").read()
    block = re.search(r"LANG_BY_SOURCE = \{(.*?)\n\}", text, re.S).group(1)
    mapping = {}
    for source, voice, epi in re.findall(
        r'"(official_[a-z0-9_]+)":\s*\("([^"]*)",\s*(None|"[^"]+")\)', block
    ):
        mapping[source] = (voice, None if epi == "None" else epi.strip('"'))
    first = re.search(r"EPITRAN_FIRST_SOURCES = \{(.*?)\n\}", text, re.S).group(1)
    return mapping, set(re.findall(r'"(official_[a-z0-9_]+)"', first))


def dominant_script(names: list[str]) -> str:
    counts: dict[str, int] = {}
    for name in names:
        for ch in name:
            if not ch.isalpha():
                continue
            try:
                script = unicodedata.name(ch).split()[0]
            except ValueError:
                continue
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "?"
    return max(counts, key=counts.get)


_REF_CACHE: dict[str, dict] = {}


def reference(iso: str) -> dict:
    if iso in _REF_CACHE:
        return _REF_CACHE[iso]
    entries: dict[str, set] = {}
    for filename in os.listdir(WP):
        if not (filename.startswith(iso + "_") and filename.endswith(".tsv")):
            continue
        if "filtered" in filename:
            continue
        with io.open(WP / filename, encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    entries.setdefault(parts[0].lower(), set()).add(parts[1].replace(" ", ""))
    _REF_CACHE.clear()          # keep one language in memory at a time
    _REF_CACHE[iso] = entries
    return entries


def normalise(text: str) -> str:
    text = "".join(c for c in str(text) if c not in STRESS)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if ord(c) not in TONE_MARKS)
    return unicodedata.normalize("NFC", text).replace("\u02d0", "").lower()


def edit_ratio(a: str, b: str) -> float:
    if a == b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(b), 1)


def espeak_batch(names: list[str], voice: str) -> dict:
    if not ESPEAK.exists() or not voice:
        return {}
    out = {}
    for start in range(0, len(names), 300):
        chunk = names[start:start + 300]
        raw = subprocess.run(
            [str(ESPEAK), "--ipa=3", "-q", f"-v{voice}"],
            input="\n".join(chunk).encode("utf-8"), capture_output=True, timeout=120,
        ).stdout.decode("utf-8", errors="replace").splitlines()
        for name, ipa in zip(chunk, raw):
            out[name] = ipa.strip()
    return out


def main() -> None:
    lang_map, epitran_first = rmd_lang_map()
    rows = []
    for path in sorted(SUBSET.glob("official_*.csv")):
        source = path.stem
        voice, epi = lang_map.get(source, (None, None))
        frame = pd.read_csv(path)
        frame.columns = [c.strip().lower() for c in frame.columns]
        names = sorted({str(n).strip() for n in frame.get("name", [])})
        if not names:
            continue
        script = dominant_script(names)
        expected = SCRIPT.get(voice, "LATIN")
        record = {
            "source": source.replace("official_", ""),
            "voice": voice or "(unmapped)",
            "epitran": epi or "",
            "routed": "epitran" if source in epitran_first else "espeak",
            "names": len(names),
            "script": script,
            "script_expected": expected,
            "script_mismatch": script != expected and script != "?",
        }
        iso = EPI_ISO.get(epi) if source in epitran_first else ISO.get(voice)
        if iso:
            ref = reference(iso)
            overlap = [n for n in names if n.lower() in ref]
            record["ref_iso"] = iso
            record["ref_overlap"] = len(overlap)
            if overlap:
                if source in epitran_first and epi:
                    import epitran
                    transcriber = epitran.Epitran(epi)
                    hyp = {n: transcriber.transliterate(n) for n in overlap}
                else:
                    hyp = espeak_batch(overlap, voice)
                exact, dists = 0, []
                for name in overlap:
                    got = normalise(hyp.get(name, ""))
                    want = {normalise(r) for r in ref[name.lower()]}
                    if got in want:
                        exact += 1
                    dists.append(min(edit_ratio(got, w) for w in want))
                record["exact_pct"] = round(100 * exact / len(overlap), 1)
                record["edit_dist"] = round(sum(dists) / len(dists), 4)
        rows.append(record)
        print(f"  {record['source']:34s} {record['voice']:6s} "
              f"script={script:9s} ref={record.get('ref_overlap', 0):4d} "
              f"exact={record.get('exact_pct', float('nan'))}")

    audit = pd.DataFrame(rows)
    audit.to_csv(BASE / "audits" / "language_assignment_audit.csv", index=False, encoding="utf-8")

    print(f"\n{len(audit)} sources audited")
    mism = audit[audit.script_mismatch]
    print(f"\nSCRIPT MISMATCH ({len(mism)}):")
    for row in mism.itertuples():
        print(f"   {row.source:34s} voice={row.voice:6s} names are {row.script}, "
              f"voice expects {row.script_expected}")
    scored = audit[audit.get("exact_pct").notna()] if "exact_pct" in audit else audit.iloc[:0]
    print(f"\nscored against a human reference: {len(scored)}")
    if len(scored):
        worst = scored.sort_values("edit_dist", ascending=False).head(15)
        print("WORST AGREEMENT (candidates for a wrong voice):")
        print(worst[["source", "voice", "ref_overlap", "exact_pct", "edit_dist"]]
              .to_string(index=False))
    unscored = audit[~audit.index.isin(scored.index)]
    print(f"\nno reference overlap, cannot be scored: {len(unscored)}")
    print("   " + ", ".join(sorted(unscored.source))[:600])


if __name__ == "__main__":
    main()
