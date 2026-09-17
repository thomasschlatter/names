"""Build IPA/panphon features for official name data and model sex differences.

The pipeline is intentionally source-aware:
- read official_data/official_*.csv files with name, sex, year, count
- phonemize names with eSpeak NG where possible, with Epitran as a fallback
- convert IPA to panphon feature means
- fit a weighted logistic regression: male-vs-female ~ panphon features + source/year

Outputs:
- official_ipa_features.csv
- official_ipa_regression_coefficients.csv
- official_ipa_regression_summary.txt
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import panphon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

try:
    import epitran
except Exception:  # pragma: no cover - optional runtime fallback
    epitran = None

try:
    from pypinyin import Style as PinyinStyle
    from pypinyin import pinyin as pypinyin
except Exception:  # pragma: no cover - optional runtime fallback
    PinyinStyle = None
    pypinyin = None

try:
    from dragonmapper import transcriptions as zh_transcriptions
except Exception:  # pragma: no cover - optional runtime fallback
    zh_transcriptions = None

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
OFFICIAL = BASE / "official_data_top30"
ESPEAK = Path(r"C:\Program Files\eSpeak NG\espeak-ng.exe")

# Abjad scripts (Hebrew/Arabic) are written without short vowels; eSpeak on the
# unvocalized registry text returns consonant skeletons (e.g. Hebrew Adam -> ʔdm),
# which silently distorts every vowel/sonority feature. The g2p artifact detector
# cannot catch this (there is no language-switch or spell-out marker), so these
# sources are excluded outright rather than filtered.
EXCLUDE_SOURCES = {
    "official_israel",
    "official_israel_jewish_hebrew",
    "official_morocco",
    "official_tunisia",
}

ft = panphon.FeatureTable()
PANPHON_FEATURES = list(ft.names)
DROP_FEATURES = {"sg", "cg", "velaric", "hitone", "hireg"}
FEATURE_COLS = [f"pf_{name}" for name in PANPHON_FEATURES if name not in DROP_FEATURES]
MODEL_FEATURES = FEATURE_COLS + ["n_phones", "prop_vowel", "n_chars"]
G2P_FLAG_COLS = [
    "g2p_language_switch",
    "g2p_chinese_letter_artifact",
    "g2p_spelled_out_artifact",
    "g2p_any_artifact",
]
EPITRAN_FIRST_SOURCES = {
    "official_canada_quebec",
    "official_france_angers",
    "official_france_nantes",
    "official_france_orleans",
    "official_france_paris",
    "official_france_saint_herblain",
    "official_france_toulouse",
    "official_switzerland_french_region",
    "official_new_caledonia_noumea",
    "official_switzerland_geneva",
}

ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
LANG_MARKER_RE = re.compile(r"\(([a-z]{2,3}(?:-[a-z0-9]+)?)\)", re.I)
STRESS_RE = re.compile("[\u02c8\u02cc\u203f\u0361\u1d7b]")
CHINESE_WORD_HINT = "t\u0283a\u026ani"
LETTER_HINTS = ("let", "l\u0259t", "l\u025bt", "l\u026at")

LANG_BY_SOURCE = {
    "official_albania": ("sq", None),
    "official_argentina": ("es", None),
    "official_argentina_buenos_aires": ("es", None),
    "official_armenia": ("hy", None),
    "official_australia": ("en", None),
    "official_australia_new_south_wales": ("en", None),
    "official_australia_queensland": ("en", None),
    "official_australia_south_australia": ("en", None),
    "official_australia_victoria": ("en", None),
    "official_austria": ("de", None),
    "official_austria_vienna": ("de", None),
    "official_azerbaijan": ("az", None),
    "official_belarus": ("be", None),
    "official_belgium": ("nl", None),
    "official_bosnia_federation": ("hr", None),
    "official_brazil": ("pt-br", None),
    "official_bulgaria": ("bg", None),
    "official_canada": ("en", None),
    "official_canada_british_columbia": ("en", None),
    "official_canada_new_brunswick": ("en", None),
    "official_canada_nova_scotia": ("en", None),
    "official_canada_ontario": ("en", None),
    "official_canada_quebec": ("fr", "fra-Latn"),
    "official_china": ("zh", None),
    "official_colombia": ("es", None),
    "official_croatia": ("hr", None),
    "official_curacao": ("pap", None),
    "official_czech": ("cs", None),
    "official_czech_mv": ("cs", None),
    "official_denmark": ("da", None),
    "official_dominican_republic": ("es", None),
    "official_england": ("en", None),
    "official_estonia": ("et", None),
    "official_estonia_population_top100": ("et", None),
    "official_faroe_islands": ("fo", None),
    "official_finland": ("fi", None),
    "official_finland_finnish_children_2024": ("fi", None),
    "official_france": ("fr", None),
    "official_france_angers": ("fr", "fra-Latn"),
    "official_france_nantes": ("fr", "fra-Latn"),
    "official_france_orleans": ("fr", "fra-Latn"),
    "official_france_paris": ("fr", "fra-Latn"),
    "official_france_saint_herblain": ("fr", "fra-Latn"),
    "official_france_toulouse": ("fr", "fra-Latn"),
    "official_french_polynesia_pirae_polynesian_names": ("poly", None),
    "official_germany_aachen": ("de", None),
    "official_germany_berlin": ("de", None),
    "official_germany_cologne": ("de", None),
    "official_germany_essen": ("de", None),
    "official_germany_leipzig": ("de", None),
    "official_germany_munich": ("de", None),
    "official_germany_muenster": ("de", None),
    "official_greenland": ("kl", None),
    "official_greenland_greenlandic_names": ("kl", None),
    "official_georgia": ("ka", None),
    "official_hungary": ("hu", None),
    "official_iceland": ("is", None),
    "official_indonesia": ("id", None),
    "official_iran": ("fa", None),
    "official_ireland": ("en", None),
    "official_israel": ("he", None),
    "official_israel_jewish_hebrew": ("he", None),
    "official_italy": ("it", None),
    "official_japan": ("ja", None),
    "official_kazakhstan": ("kk", None),
    "official_kazakhstan_kazakh": ("kk", None),
    "official_kyrgyzstan": ("ky", None),
    "official_latvia": ("lv", None),
    "official_lithuania": ("lt", None),
    "official_morocco": ("ar", None),
    "official_mexico_city": ("es", None),
    "official_netherlands": ("nl", None),
    "official_new_caledonia_noumea": ("fr", "fra-Latn"),
    "official_newzealand": ("en", None),
    "official_newzealand_maori": ("mi", None),
    "official_north_macedonia": ("mk", None),
    "official_northern_ireland": ("en", None),
    "official_norway": ("nb", None),
    # eSpeak NG 1.52 on this machine has no Tagalog/Filipino voice; the official
    # top names are mostly English/Spanish-style spellings, so English is a
    # conservative fallback rather than dropping the source entirely.
    "official_philippines": ("en", None),
    "official_paraguay": ("es", None),
    "official_poland": ("pl", None),
    "official_portugal": ("pt", None),
    "official_romania": ("ro", None),
    "official_russia": ("ru", None),
    "official_russia_moscow": ("ru", None),
    "official_scotland": ("en", None),
    "official_slovakia": ("sk", None),
    "official_slovenia": ("sl", None),
    "official_southafrica": ("af", None),
    "official_southkorea": ("ko", "kor-Hang"),
    "official_spain": ("es", None),
    "official_spain_basque_country": ("es", None),
    "official_spain_basque_country_top100": ("eu", None),
    "official_spain_catalonia": ("ca", None),
    "official_spain_valencian_community": ("ca", None),
    "official_sweden": ("sv", None),
    "official_switzerland": ("de", None),
    "official_switzerland_french_region": ("fr", "fra-Latn"),
    "official_switzerland_geneva": ("fr", "fra-Latn"),
    "official_switzerland_italian_region": ("it", None),
    "official_switzerland_zurich": ("de", None),
    "official_switzerland_zurich_canton": ("de", None),
    "official_taiwan": ("zh", None),
    "official_tunisia": ("ar", None),
    "official_turkey": ("tr", None),
    "official_turkey_isimanlam_top500": ("tr", None),
    "official_uruguay": ("es", None),
    "official_uruguay_montevideo": ("es", None),
    "official_ukraine": ("uk", None),
    "official_uzbekistan": ("uz", None),
    "official_us_california": ("en", None),
    "official_us_newyork_city": ("en", None),
    "official_us_newyork_state": ("en", None),
    "official_usa": ("en", None),
}


def normalize_ipa_text(value: str) -> str:
    return ZERO_WIDTH_RE.sub("", str(value or ""))


def clean_ipa(value: str) -> str:
    value = normalize_ipa_text(value)
    return STRESS_RE.sub("", value).strip()


def simplify_for_artifact_detection(value: str) -> str:
    value = clean_ipa(value).lower()
    value = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in value if not unicodedata.category(ch).startswith("M"))


def g2p_artifact_flags(
    ipa: str,
    n_phones: float | int | None,
    n_chars: float | int | None,
) -> dict[str, bool]:
    normalized = clean_ipa(ipa)
    simple = simplify_for_artifact_detection(ipa)
    language_switch = bool(LANG_MARKER_RE.search(normalized))
    chinese_letter = CHINESE_WORD_HINT in simple and any(hint in simple for hint in LETTER_HINTS)
    try:
        phones_per_char = float(n_phones) / max(int(n_chars), 1)
    except Exception:
        phones_per_char = float("nan")
    spelled_out = bool(language_switch and np.isfinite(phones_per_char) and phones_per_char >= 1.8)
    any_artifact = bool(language_switch or chinese_letter or spelled_out)
    return {
        "g2p_language_switch": language_switch,
        "g2p_chinese_letter_artifact": chinese_letter,
        "g2p_spelled_out_artifact": spelled_out,
        "g2p_any_artifact": any_artifact,
    }


def add_g2p_quality_columns(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    if not {"ipa", "n_phones"}.issubset(features.columns):
        return features
    if "n_chars" not in features.columns:
        features["n_chars"] = features["name"].astype(str).str.len()
    if all(col in features.columns for col in G2P_FLAG_COLS):
        for col in G2P_FLAG_COLS:
            features[col] = features[col].astype(bool)
        return features
    flags = [
        g2p_artifact_flags(row.ipa, row.n_phones, row.n_chars)
        for row in features[["ipa", "n_phones", "n_chars"]].itertuples(index=False)
    ]
    flag_df = pd.DataFrame(flags, index=features.index)
    for col in G2P_FLAG_COLS:
        features[col] = flag_df[col].fillna(False).astype(bool)
    return features


def espeak_batch(names: list[str], lang: str, chunk: int = 300) -> dict[str, str]:
    if not ESPEAK.exists():
        return {}
    out: dict[str, str] = {}
    for i in range(0, len(names), chunk):
        batch = names[i : i + chunk]
        raw = subprocess.run(
            [str(ESPEAK), "--ipa=3", "-q", f"-v{lang}"],
            input="\n".join(batch).encode("utf-8"),
            capture_output=True,
            timeout=90,
        ).stdout
        lines = raw.decode("utf-8", errors="replace").splitlines()
        for name, ipa in zip(batch, lines):
            ipa = clean_ipa(ipa)
            if ipa:
                out[name] = ipa
    return out


def chinese_pinyin_ipa_batch(names: list[str]) -> dict[str, str]:
    """Convert Hanzi names through pinyin before IPA.

    eSpeak's Mandarin voice handles many Hanzi names but can switch languages
    for some common name characters, e.g. Taiwanese names ending in 雄. Pinyin
    conversion avoids those spelling artifacts while still allowing eSpeak as a
    fallback if the optional libraries are unavailable.
    """

    if pypinyin is None or PinyinStyle is None or zh_transcriptions is None:
        return {}
    out = {}
    for name in names:
        try:
            syllables = pypinyin(
                name,
                style=PinyinStyle.TONE3,
                heteronym=False,
                errors="default",
            )
            pinyin_text = " ".join(item[0] for item in syllables if item and str(item[0]).strip())
            if not pinyin_text:
                continue
            ipa = clean_ipa(zh_transcriptions.pinyin_to_ipa(pinyin_text))
            if ipa:
                out[name] = ipa
        except Exception:
            continue
    return out


def croatian_ipa_batch(names: list[str]) -> dict[str, str]:
    """Conservative segmental Croatian G2P for transparent name spellings."""

    digraphs = {
        "dž": "dʒ",
        "lj": "ʎ",
        "nj": "ɲ",
    }
    letters = {
        "a": "a",
        "b": "b",
        "c": "ts",
        "č": "tʃ",
        "ć": "tɕ",
        "d": "d",
        "đ": "dʑ",
        "e": "e",
        "f": "f",
        "g": "ɡ",
        "h": "x",
        "i": "i",
        "j": "j",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "o": "o",
        "p": "p",
        "r": "r",
        "s": "s",
        "š": "ʃ",
        "t": "t",
        "u": "u",
        "v": "v",
        "z": "z",
        "ž": "ʒ",
    }
    out: dict[str, str] = {}
    for name in names:
        text = unicodedata.normalize("NFC", name).lower()
        ipa_parts: list[str] = []
        i = 0
        while i < len(text):
            if text[i].isspace() or text[i] in "-'’.":
                i += 1
                continue
            pair = text[i : i + 2]
            if pair in digraphs:
                ipa_parts.append(digraphs[pair])
                i += 2
                continue
            sound = letters.get(text[i])
            if sound is None:
                ipa_parts = []
                break
            ipa_parts.append(sound)
            i += 1
        if ipa_parts:
            out[name] = "".join(ipa_parts)
    return out


def polynesian_ipa_batch(names: list[str]) -> dict[str, str]:
    """Transparent segmental G2P for filtered Polynesian-pattern names."""

    letters = {
        "a": "a",
        "e": "e",
        "i": "i",
        "o": "o",
        "u": "u",
        "f": "f",
        "h": "h",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "p": "p",
        "r": "ɾ",
        "t": "t",
        "v": "ʋ",
        "w": "w",
    }
    vowels = set("aeiou")
    out: dict[str, str] = {}
    for name in names:
        text = unicodedata.normalize(
            "NFD",
            name.lower().replace("’", "'").replace("`", "'").replace("ʻ", "'"),
        )
        ipa_parts: list[str] = []
        last_vowel_idx: int | None = None
        failed = False
        for char in text:
            if char == "\u0304" and last_vowel_idx is not None:
                ipa_parts[last_vowel_idx] = f"{ipa_parts[last_vowel_idx]}ː"
                continue
            if unicodedata.category(char).startswith("M"):
                continue
            if char.isspace() or char in "-.":
                last_vowel_idx = None
                continue
            if char == "'":
                ipa_parts.append("ʔ")
                last_vowel_idx = None
                continue
            sound = letters.get(char)
            if sound is None:
                failed = True
                break
            ipa_parts.append(sound)
            last_vowel_idx = len(ipa_parts) - 1 if char in vowels else None
        if ipa_parts and not failed:
            out[name] = "".join(ipa_parts)
    return out


def phonemizer_label(source: str) -> str:
    espeak_code, _ = LANG_BY_SOURCE.get(source, ("", None))
    if source == "official_croatia":
        return "custom-hr-orthographic"
    if espeak_code == "poly":
        return "custom-polynesian-orthographic"
    if source in EPITRAN_FIRST_SOURCES:
        _, epi_code = LANG_BY_SOURCE.get(source, ("", None))
        return f"epitran:{epi_code}" if epi_code else espeak_code
    if espeak_code == "zh" and pypinyin is not None and zh_transcriptions is not None:
        return "pypinyin+dragonmapper"
    return espeak_code


_EPITRAN_CACHE = {}


def epitran_batch(names: list[str], code: str | None) -> dict[str, str]:
    if not code or epitran is None:
        return {}
    if code not in _EPITRAN_CACHE:
        try:
            _EPITRAN_CACHE[code] = epitran.Epitran(code)
        except Exception:
            return {}
    ep = _EPITRAN_CACHE[code]
    out = {}
    for name in names:
        try:
            ipa = clean_ipa(ep.transliterate(name))
            if ipa:
                out[name] = ipa
        except Exception:
            continue
    return out


def ipa_to_features(ipa: str) -> dict[str, float] | None:
    segs = ft.word_to_vector_list(ipa, numeric=True)
    if not segs:
        return None
    arr = (np.array(segs, dtype=float) + 1.0) / 2.0
    row = {}
    for idx, feature in enumerate(PANPHON_FEATURES):
        if feature in DROP_FEATURES:
            continue
        row[f"pf_{feature}"] = float(arr[:, idx].mean())
    syl_idx = PANPHON_FEATURES.index("syl")
    row["n_phones"] = float(len(segs))
    row["prop_vowel"] = float(arr[:, syl_idx].mean())
    return row


def load_source(path: Path, max_names_per_sex_source: int | None) -> pd.DataFrame | None:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(col).strip().lower() for col in df.columns]
    required = {"name", "sex", "count"}
    if not required.issubset(df.columns):
        return None
    if "year" not in df.columns:
        df["year"] = np.nan
    df = df[["name", "sex", "year", "count"]].copy()
    df["source"] = path.stem
    df["name"] = df["name"].astype(str).str.strip()
    df["sex"] = df["sex"].astype(str).str.upper().str.strip().str[0]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df[(df["name"] != "") & df["sex"].isin(["F", "M"]) & (df["count"] > 0)]
    if df.empty:
        return None
    df = df.groupby(["source", "name", "sex", "year"], dropna=False, as_index=False)["count"].sum()
    if max_names_per_sex_source:
        df = (
            df.sort_values("count", ascending=False)
            .groupby(["source", "sex"], group_keys=False)
            .head(max_names_per_sex_source)
        )
    return df


def weighted_rate(mask: pd.Series, weights: pd.Series) -> float:
    weights = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
    if weights.sum() <= 0:
        return float("nan")
    return float(np.average(mask.astype(float), weights=weights))


def write_g2p_audit(df: pd.DataFrame, args: argparse.Namespace) -> set[str]:
    if df.empty or "g2p_any_artifact" not in df.columns:
        return set()
    rows = []
    for source, group in df.groupby("source", sort=True):
        drop_source = weighted_rate(group["g2p_any_artifact"], group["count"]) >= args.source_artifact_threshold
        kept = group.iloc[0:0] if drop_source else group[~group["g2p_any_artifact"]]
        rows.append(
            {
                "source": source,
                "espeak_voice": phonemizer_label(source),
                "rows_before_filter": int(len(group)),
                "unique_names_before_filter": int(group["name"].nunique()),
                "weighted_count_before_filter": float(group["count"].sum()),
                "artifact_rate": float(group["g2p_any_artifact"].mean()),
                "artifact_rate_weighted": weighted_rate(group["g2p_any_artifact"], group["count"]),
                "language_switch_rate_weighted": weighted_rate(group["g2p_language_switch"], group["count"]),
                "chinese_letter_artifact_rate_weighted": weighted_rate(
                    group["g2p_chinese_letter_artifact"], group["count"]
                ),
                "spelled_out_artifact_rate_weighted": weighted_rate(
                    group["g2p_spelled_out_artifact"], group["count"]
                ),
                "drop_source": bool(drop_source),
                "rows_after_filter": int(len(kept)),
                "unique_names_after_filter": int(kept["name"].nunique()),
                "weighted_count_after_filter": float(kept["count"].sum()),
            }
        )
    audit = pd.DataFrame(rows).sort_values("artifact_rate_weighted", ascending=False)
    audit.to_csv(args.g2p_audit_out, index=False)
    dropped = set(audit.loc[audit["drop_source"], "source"])
    if args.filter_g2p_artifacts:
        print(
            f"G2P filter: dropping {len(dropped)} sources at "
            f">= {args.source_artifact_threshold:.0%} weighted artifacts; "
            f"row-filtering lower-artifact sources."
        )
    return dropped


def build_feature_table(args: argparse.Namespace) -> pd.DataFrame:
    files = sorted(OFFICIAL.glob("official_*.csv"))
    files = [p for p in files if p.stem not in EXCLUDE_SOURCES]
    if args.sources:
        wanted = set(args.sources)
        files = [path for path in files if path.stem in wanted or path.name in wanted]

    frames = []
    for path in files:
        if path.stem not in LANG_BY_SOURCE:
            print(f"SKIP {path.stem}: no language mapping")
            continue
        df = load_source(path, args.max_names_per_sex_source)
        if df is None or df.empty:
            print(f"SKIP {path.stem}: no usable rows")
            continue
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    existing = None
    cache_path = args.cache_in if args.cache_in else args.features_out
    if args.reuse_cache and cache_path.exists():
        existing = pd.read_csv(cache_path, low_memory=False)
        cache_cols = ["source", "name", "ipa"] + MODEL_FEATURES + G2P_FLAG_COLS
        existing = existing[[col for col in cache_cols if col in existing.columns]].drop_duplicates(["source", "name"])
        existing = add_g2p_quality_columns(existing)

    feature_frames = []
    for source, group in data.groupby("source"):
        source_rows = group[["source", "name"]].drop_duplicates()
        if existing is not None:
            cached = source_rows.merge(existing, on=["source", "name"], how="inner")
            cached_names = set(cached["name"])
            if not cached.empty:
                feature_frames.append(cached)
        else:
            cached_names = set()

        todo = sorted(set(source_rows["name"]) - cached_names)
        if not todo:
            continue

        espeak_code, epi_code = LANG_BY_SOURCE[source]
        if source == "official_croatia":
            ipa_map = croatian_ipa_batch(todo)
        elif espeak_code == "poly":
            ipa_map = polynesian_ipa_batch(todo)
        elif source in EPITRAN_FIRST_SOURCES:
            ipa_map = epitran_batch(todo, epi_code)
            missing_for_espeak = [name for name in todo if name not in ipa_map]
            ipa_map.update(espeak_batch(missing_for_espeak, espeak_code) if espeak_code else {})
        elif espeak_code == "zh":
            ipa_map = chinese_pinyin_ipa_batch(todo)
            missing_for_espeak = [name for name in todo if name not in ipa_map]
            ipa_map.update(espeak_batch(missing_for_espeak, espeak_code) if espeak_code else {})
        else:
            ipa_map = espeak_batch(todo, espeak_code) if espeak_code else {}
        missing = [name for name in todo if name not in ipa_map]
        ipa_map.update(epitran_batch(missing, epi_code))

        rows = []
        for name in todo:
            ipa = ipa_map.get(name, "")
            feats = ipa_to_features(ipa) if ipa else None
            if feats:
                n_chars = len(name)
                flags = g2p_artifact_flags(ipa, feats["n_phones"], n_chars)
                rows.append(
                    {
                        "source": source,
                        "name": name,
                        "ipa": ipa,
                        "n_chars": n_chars,
                        **feats,
                        **flags,
                    }
                )
        print(f"{source}: {len(rows)}/{len(todo)} new names phonemized")
        if rows:
            feature_frames.append(pd.DataFrame(rows))

    features = pd.concat(feature_frames, ignore_index=True).drop_duplicates(["source", "name"])
    features = add_g2p_quality_columns(features)
    out = data.merge(features, on=["source", "name"], how="inner")
    out = add_g2p_quality_columns(out)
    out["sex_male"] = (out["sex"] == "M").astype(int)
    out["year_centered"] = out.groupby("source")["year"].transform(lambda s: s - s.mean())
    out["year_centered"] = out["year_centered"].fillna(0)
    dropped_sources = write_g2p_audit(out, args)
    if args.filter_g2p_artifacts:
        before_rows = len(out)
        before_count = out["count"].sum()
        out = out[(~out["source"].isin(dropped_sources)) & (~out["g2p_any_artifact"])].copy()
        print(
            f"G2P filter kept {len(out):,}/{before_rows:,} rows "
            f"({out['count'].sum():,.0f}/{before_count:,.0f} weighted count)."
        )
    out.to_csv(args.features_out, index=False)
    return out


def run_regression(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    model_df = df.dropna(subset=MODEL_FEATURES + ["sex_male", "count"]).copy()
    if args.min_source_rows:
        keep = model_df["source"].value_counts()
        keep = set(keep[keep >= args.min_source_rows].index)
        model_df = model_df[model_df["source"].isin(keep)]

    numeric_cols = MODEL_FEATURES + ["year_centered"]
    X_num = model_df[numeric_cols].astype(float).copy()
    means = X_num.mean()
    sds = X_num.std(ddof=0).replace(0, 1)
    X_num = (X_num - means) / sds
    x_parts = [X_num.reset_index(drop=True)]
    if args.source_fixed_effects:
        dummies = pd.get_dummies(model_df["source"], prefix="src", drop_first=True, dtype=float).reset_index(drop=True)
        x_parts.append(dummies)
    X = pd.concat(x_parts, axis=1)
    y = model_df["sex_male"].astype(int)
    weights = model_df["count"].astype(float).clip(lower=1)

    fit = LogisticRegression(
        C=args.regularization_c,
        max_iter=args.max_iter,
        solver="lbfgs",
    )
    fit.fit(X, y, sample_weight=weights)
    proba = fit.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, proba, sample_weight=weights)
    except Exception:
        auc = float("nan")
    coef = pd.DataFrame(
        {
            "term": X.columns,
            "coef": fit.coef_[0],
            "odds_ratio_per_sd": np.exp(fit.coef_[0]),
        }
    )
    coef = coef[coef["term"].isin(numeric_cols)].copy()
    coef["abs_coef"] = coef["coef"].abs()
    coef = coef.sort_values("abs_coef", ascending=False)
    coef.to_csv(args.coef_out, index=False)
    summary = [
        f"Rows used: {len(model_df):,}",
        f"Weighted births/name counts: {weights.sum():,.0f}",
        f"Sources used: {model_df['source'].nunique()}",
        f"Feature terms: {len(MODEL_FEATURES)}",
        f"Source fixed effects: {args.source_fixed_effects}",
        f"Logistic regularization C: {args.regularization_c}",
        f"G2P artifact row/source filtering: {args.filter_g2p_artifacts}",
        f"G2P source artifact threshold: {args.source_artifact_threshold:.3f}",
        f"G2P audit file: {args.g2p_audit_out.name}",
        f"Weighted log loss: {log_loss(y, proba, sample_weight=weights):.6f}",
        f"Weighted accuracy: {accuracy_score(y, pred, sample_weight=weights):.6f}",
        f"Weighted ROC AUC: {auc:.6f}",
        "",
        "Coefficients are from standardized numeric predictors.",
        "Positive coefficients predict male-coded names; negative coefficients predict female-coded names.",
    ]
    args.summary_out.write_text("\n".join(summary), encoding="utf-8")
    return coef


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", help="Optional source stems/files to include.")
    parser.add_argument("--max-names-per-sex-source", type=int, default=None)
    parser.add_argument("--min-source-rows", type=int, default=20)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--no-source-fixed-effects", dest="source_fixed_effects", action="store_false")
    parser.add_argument("--no-reuse-cache", dest="reuse_cache", action="store_false")
    parser.add_argument(
        "--cache-in",
        type=Path,
        default=None,
        help="Optional existing feature cache to reuse when writing to a different features-out file.",
    )
    parser.add_argument(
        "--filter-g2p-artifacts",
        action="store_true",
        help="Remove artifact rows and drop sources above the source artifact threshold.",
    )
    parser.add_argument(
        "--source-artifact-threshold",
        type=float,
        default=0.25,
        help="Drop a source when its count-weighted G2P artifact rate is at or above this value.",
    )
    parser.add_argument("--features-out", type=Path, default=BASE / "official_ipa_features.csv")
    parser.add_argument("--coef-out", type=Path, default=BASE / "official_ipa_regression_coefficients.csv")
    parser.add_argument("--summary-out", type=Path, default=BASE / "official_ipa_regression_summary.txt")
    parser.add_argument("--g2p-audit-out", type=Path, default=BASE / "official_ipa_g2p_audit.csv")
    parser.set_defaults(source_fixed_effects=True, reuse_cache=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_feature_table(args)
    print(f"Feature rows: {len(df):,}; sources: {df['source'].nunique()}")
    coef = run_regression(df, args)
    print("Top panphon/model terms:")
    print(coef.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
