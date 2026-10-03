# -*- coding: utf-8 -*-
"""Execute the Rmd's python chunks directly, with the cache off, to test the strict parser."""
import io, re, sys
from pathlib import Path
src=io.open("paper_registry.Rmd",encoding="utf-8").read()
chunks=re.findall(r"```\{python (pipeline-constants|pipeline-g2p|pipeline-fit)[^}]*\}\n(.*?)\n```", src, re.S)
class R: PROJECT_DIR=str(Path(".").resolve())
g={"__name__":"pipeline","r":R()}
for name,code in chunks:
    exec(compile(code,f"<{name}>","exec"),g)
from types import SimpleNamespace
BASE=g["BASE"]
CFG=SimpleNamespace(sources=None,max_names_per_sex_source=None,min_source_rows=20,
    regularization_c=1.0,max_iter=2000,source_fixed_effects=True,
    reuse_cache=False, cache_in=None, filter_g2p_artifacts=True,
    source_artifact_threshold=0.25,
    features_out=BASE/"_check_features.csv", coef_out=BASE/"_check_coef.csv",
    summary_out=BASE/"_check_summary.txt", g2p_audit_out=BASE/"_check_g2p_audit.csv",
    unparsed_out=BASE/"official_ipa_unparsed_top30.csv")
try:
    g["build_feature_table"](CFG)
    print("\nNO UNPARSED SYMBOLS: every transcription parsed completely.")
except g["UnparsedIPA_Halt"] as e:
    print("\nHALTED AS DESIGNED:\n", e)
