# -*- coding: utf-8 -*-
"""Rebuild the Analysis 3 panel: phonological gender gap against gender equality.

`equality_final_v4.csv` (May 2026) has no producer script anywhere in the
project's history, and its numbers CANNOT be reproduced: its `mean_F` matches no
feature this pipeline computes (closest are pf_cont at 0.023 and pf_son at
0.034 mean absolute difference), and its nF/nM are full-list sizes rather than
any subset used here. 50 of its 101 countries have no registry at all.

So this does not try to reproduce it. It builds a NEW panel from data this
project can regenerate, and states every choice:

  GAP (d)      Cohen's d on SONORITY (pf_son) between female and male names,
               computed from the same feature table the regression uses, so
               Analyses 2 and 3 rest on one transcription pass. Names are
               weighted by their registry count, and the pooled standard
               deviation is the weighted one.
  COUNTRY      subnational sources are pooled into their country, because for
               several countries (Germany above all) the city registries ARE
               the national evidence. The registries behind each country are
               listed in the output.
  EQUALITY     Global Gender Gap Index from equality_data/ggi_panel.csv, most
               recent year available per country.

Countries in the old panel with no registry are NOT carried over. They are
listed separately so the decision about them is explicit rather than inherited.

Output: audits/equality_panel_rebuilt.csv
        audits/equality_panel_dropped.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
FEATURES = BASE / "official_ipa_features_top30.csv"
GGI = BASE / "equality_data" / "ggi_panel.csv"
OLD = BASE / "equality_final_v4.csv"
OUT = BASE / "audits"

# source_id -> country. Only entries whose country cannot be read off the
# source_id prefix need listing.
COUNTRY_FIX = {
    "england": "United Kingdom", "scotland": "United Kingdom",
    "northern_ireland": "United Kingdom", "wales": "United Kingdom",
    "usa": "United States", "us": "United States",
    "newzealand": "New Zealand", "southkorea": "South Korea",
    "southafrica": "South Africa", "czech": "Czechia", "czech_mv": "Czechia",
    "bosnia_federation": "Bosnia and Herzegovina",
    "north_macedonia": "North Macedonia",
    "dominican_republic": "Dominican Republic",
    "faroe_islands": "Faroe Islands", "new_caledonia": "New Caledonia",
    "french_polynesia": "French Polynesia", "curacao": "Curacao",
    "israel_jewish_hebrew": "Israel", "mexico_city": "Mexico",
    "spain_basque_country": "Spain", "spain_catalonia": "Spain",
    "spain_valencian_community": "Spain", "uruguay_montevideo": "Uruguay",
    "argentina_buenos_aires": "Argentina", "russia_moscow": "Russia",
    "kazakhstan_kazakh": "Kazakhstan", "turkey_isimanlam": "Turkey",
    "estonia_population": "Estonia", "finland_finnish_children": "Finland",
    "greenland_greenlandic_names": "Greenland",
    "newzealand_maori": "New Zealand",
}
MULTIWORD = ["new_caledonia", "french_polynesia", "north_macedonia",
             "dominican_republic", "faroe_islands", "bosnia_federation",
             "czech_mv", "estonia_population", "finland_finnish_children",
             "greenland_greenlandic_names", "israel_jewish_hebrew",
             "newzealand_maori", "spain_basque_country", "spain_catalonia",
             "spain_valencian_community", "uruguay_montevideo",
             "argentina_buenos_aires", "russia_moscow", "kazakhstan_kazakh",
             "turkey_isimanlam", "mexico_city", "northern_ireland"]


# The GGI panel uses its own country naming ("UK", "USA", formal UN long forms).
# Matching on the plain name silently loses 8 countries, including the two
# largest English-speaking ones.
GGI_ALIAS = {
    "United Kingdom": "UK", "United States": "USA",
    "Czechia": "Czech Republic", "South Korea": "Korea (the Republic of)",
    "Russia": "Russian Federation (the)", "Iran": "Iran (Islamic Republic of)",
    "Philippines": "Philippines (the)",
    "Dominican Republic": "Dominican Republic (the)",
}
# Territories the Global Gender Gap Index does not cover at all. Recorded so a
# missing GGI is not mistaken for a join failure.
NOT_IN_GGI = {"Taiwan", "Curacao", "Faroe Islands", "French Polynesia",
              "Greenland", "New Caledonia"}


def country_of(source_id: str) -> str:
    for key in sorted(MULTIWORD, key=len, reverse=True):
        if source_id.startswith(key):
            return COUNTRY_FIX.get(key, key.replace("_", " ").title())
    if source_id in COUNTRY_FIX:
        return COUNTRY_FIX[source_id]
    head = source_id.split("_")[0]
    return COUNTRY_FIX.get(head, head.replace("_", " ").title())


def weighted_d(values_f, w_f, values_m, w_m) -> tuple:
    """Count-weighted Cohen's d, female minus male."""
    def wmean(v, w):
        return float(np.average(v, weights=w))

    def wvar(v, w, mean):
        return float(np.average((v - mean) ** 2, weights=w))

    mf, mm = wmean(values_f, w_f), wmean(values_m, w_m)
    nf, nm = len(values_f), len(values_m)
    vf, vm = wvar(values_f, w_f, mf), wvar(values_m, w_m, mm)
    pooled = ((nf - 1) * vf + (nm - 1) * vm) / max(nf + nm - 2, 1)
    sd = np.sqrt(pooled)
    return (mf - mm) / sd if sd > 0 else np.nan, mf, mm, nf, nm


def main() -> None:
    OUT.mkdir(exist_ok=True)
    feat = pd.read_csv(FEATURES, low_memory=False)
    feat["country"] = feat["source"].str.replace("official_", "", regex=False).map(country_of)

    rows = []
    for country, group in feat.groupby("country"):
        female = group[group["sex"] == "F"]
        male = group[group["sex"] == "M"]
        vf = pd.to_numeric(female["pf_son"], errors="coerce")
        vm = pd.to_numeric(male["pf_son"], errors="coerce")
        wf = pd.to_numeric(female["count"], errors="coerce").fillna(1).clip(lower=1)
        wm = pd.to_numeric(male["count"], errors="coerce").fillna(1).clip(lower=1)
        kf, km = vf.notna(), vm.notna()
        if kf.sum() < 5 or km.sum() < 5:
            continue
        d, mf, mm, nf, nm = weighted_d(vf[kf], wf[kf], vm[km], wm[km])
        rows.append({"country": country, "d_sonority": d,
                     "mean_F": mf, "mean_M": mm, "nF": nf, "nM": nm,
                     "registries": group["source"].nunique(),
                     "registry_list": ";".join(sorted(group["source"].unique()))[:200]})
    panel = pd.DataFrame(rows)

    ggi = pd.read_csv(GGI)
    ggi = ggi.dropna(subset=["gggi_ggi"])
    latest = (ggi.sort_values("year").groupby("country", as_index=False)
                 .last()[["country", "year", "gggi_ggi"]]
                 .rename(columns={"year": "ggi_year", "gggi_ggi": "GGI"}))
    panel["ggi_name"] = panel["country"].map(lambda c: GGI_ALIAS.get(c, c))
    panel = panel.merge(latest.rename(columns={"country": "ggi_name"}),
                        on="ggi_name", how="left")
    panel["ggi_status"] = np.where(
        panel["GGI"].notna(), "matched",
        np.where(panel["country"].isin(NOT_IN_GGI),
                 "not covered by the Global Gender Gap Index", "NO MATCH"))

    old = pd.read_csv(OLD)
    old_countries = set(old["country"].astype(str))
    dropped = sorted(old_countries - set(panel["country"]))
    pd.DataFrame({"country_in_old_panel_without_registry": dropped}).to_csv(
        OUT / "equality_panel_dropped.csv", index=False, encoding="utf-8")
    panel.sort_values("country").to_csv(OUT / "equality_panel_rebuilt.csv",
                                        index=False, encoding="utf-8")

    unmatched = panel[panel["ggi_status"] == "NO MATCH"]
    if len(unmatched):
        print(f"still unmatched to GGI ({len(unmatched)}): "
              f"{', '.join(sorted(unmatched['country']))}")
    with_ggi = panel[panel["GGI"].notna()]
    print(f"countries with registry data : {len(panel)}")
    print(f"  of which GGI matched       : {len(with_ggi)}")
    print(f"old-panel countries with no registry, not carried over: {len(dropped)}")
    print(f"   {', '.join(dropped[:18])}{' ...' if len(dropped) > 18 else ''}")
    if len(with_ggi) > 3:
        share = (with_ggi["d_sonority"] > 0).mean()
        r = with_ggi["d_sonority"].corr(with_ggi["GGI"])
        rs = with_ggi["d_sonority"].corr(with_ggi["GGI"], method="spearman")
        print(f"\nd > 0 in {share:.1%} of countries "
              f"({int((with_ggi['d_sonority'] > 0).sum())}/{len(with_ggi)})")
        print(f"correlation of d with GGI: Pearson {r:+.3f}, Spearman {rs:+.3f}")
        print("\nold panel for comparison: "
              f"{(old['d'] > 0).mean():.1%} positive, "
              f"Pearson {old['d'].corr(old['GGI']):+.3f}")


if __name__ == "__main__":
    main()
