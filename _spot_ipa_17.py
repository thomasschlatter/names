# Spot-check IPA quality for the 17 registries used in the paper.
# For each source: show top-5 female and top-5 male names with IPA.
# Flag any obviously broken IPA (empty, parenthetical markers, non-IPA chars).

import re, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

SOURCES = [
    "official_australia", "official_austria", "official_azerbaijan",
    "official_england",   "official_faroe_islands", "official_italy",
    "official_latvia",    "official_netherlands",   "official_norway",
    "official_poland",    "official_slovenia",      "official_southkorea",
    "official_spain",     "official_sweden",        "official_taiwan",
    "official_usa",       "official_czech"
]

LANG_LABEL = {
    "official_australia":    "English (AU)",
    "official_austria":      "German",
    "official_azerbaijan":   "Azerbaijani",
    "official_england":      "English (EN)",
    "official_faroe_islands":"Faroese",
    "official_italy":        "Italian",
    "official_latvia":       "Latvian",
    "official_netherlands":  "Dutch",
    "official_norway":       "Norwegian",
    "official_poland":       "Polish",
    "official_slovenia":     "Slovenian",
    "official_southkorea":   "Korean",
    "official_spain":        "Spanish",
    "official_sweden":       "Swedish",
    "official_taiwan":       "Mandarin",
    "official_usa":          "English (US)",
    "official_czech":        "Czech",
}

# Try loading the full features file (has ipa column)
try:
    feat = pd.read_csv("official_ipa_features_main_conservative.csv")
    # add Taiwan and Czech from their probe files if needed
    extra_sources = [s for s in SOURCES if s not in feat["source"].unique()]
    print(f"Missing from main_conservative: {extra_sources}")
except Exception as e:
    print(f"Error loading features: {e}")
    feat = None

# Also load the spotcheck file for easy viewing
spot = pd.read_csv("official_ipa_spotcheck_sample.csv")
spot = spot[spot["source"].isin(SOURCES)]
spot_top = spot[spot["sample_type"] == "top_count_by_source_sex"].copy()

# Artifact detection
BAD_PATTERN = re.compile(r"[\(\)]|^\s*$|[a-zA-Z]{4,}")

def flag_ipa(ipa):
    if pd.isna(ipa) or str(ipa).strip() == "":
        return "EMPTY"
    s = str(ipa)
    if "(" in s or ")" in s:
        return "PARENTHETICAL"
    if re.search(r"[a-zA-Z]{4,}", s):
        return "LIKELY_ASCII_FALLBACK"
    return "OK"

spot_top["flag"] = spot_top["ipa"].apply(flag_ipa)

print("\n=== IPA spot-check: top names per registry ===\n")
for src in SOURCES:
    sub = spot_top[spot_top["source"] == src].copy()
    lang = LANG_LABEL.get(src, src)
    voice = sub["espeak_voice"].iloc[0] if len(sub) > 0 else "?"
    flags = sub["flag"].value_counts().to_dict()
    bad = {k: v for k, v in flags.items() if k != "OK"}
    print("-"*60)
    print(f"{lang} ({src})  voice={voice}  n_names={len(sub)}  issues={bad if bad else 'none'}")
    for sex in ["F", "M"]:
        rows = sub[sub["sex"] == sex].head(5)
        for _, r in rows.iterrows():
            flag = f" *** {r['flag']}" if r["flag"] != "OK" else ""
            print(f"  {sex}  {r['name']:<14} {str(r['ipa']):<30}{flag}")
    print()
