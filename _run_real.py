# -*- coding: utf-8 -*-
"""Run the manuscript's pipeline chunks to the real output paths."""
import io, re
from pathlib import Path
from types import SimpleNamespace
src=io.open("paper_registry.Rmd",encoding="utf-8").read()
chunks=re.findall(r"```\{python (pipeline-constants|pipeline-g2p|pipeline-fit)[^}]*\}\n(.*?)\n```",src,re.S)
class R: PROJECT_DIR=str(Path(".").resolve())
g={"__name__":"pipeline","r":R()}
for name,code in chunks: exec(compile(code,f"<{name}>","exec"),g)
BASE=g["BASE"]
CFG=SimpleNamespace(sources=None,max_names_per_sex_source=None,min_source_rows=20,
    regularization_c=1.0,max_iter=2000,source_fixed_effects=True,
    reuse_cache=True,cache_in=BASE/"official_ipa_features_top30.csv",
    filter_g2p_artifacts=True,source_artifact_threshold=0.25,
    features_out=BASE/"official_ipa_features_top30.csv",
    coef_out=BASE/"official_ipa_regression_coefficients_top30.csv",
    summary_out=BASE/"official_ipa_regression_summary_top30.txt",
    g2p_audit_out=BASE/"official_ipa_g2p_audit_top30.csv",
    unparsed_out=BASE/"official_ipa_unparsed_top30.csv",
    on_unparsed="exclude")
df=g["build_feature_table"](CFG)
print(f"\nFeature rows: {len(df):,}; sources: {df['source'].nunique()}")
g["run_regression"](df,CFG)
print("\n--- NEW SUMMARY ---")
print(io.open(CFG.summary_out,encoding="utf-8").read())
