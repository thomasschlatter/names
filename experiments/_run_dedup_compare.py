# -*- coding: utf-8 -*-
"""Run the manuscript's regression on the de-duplicated subset and compare."""
import io, re
from pathlib import Path
from types import SimpleNamespace
src=io.open("paper_registry.Rmd",encoding="utf-8").read()
chunks=re.findall(r"```\{python (pipeline-constants|pipeline-g2p|pipeline-fit)[^}]*\}\n(.*?)\n```",src,re.S)
class R: PROJECT_DIR=str(Path(".").resolve())
g={"__name__":"pipeline","r":R()}
for name,code in chunks: exec(compile(code,f"<{name}>","exec"),g)
BASE=g["BASE"]
g["OFFICIAL"]=BASE/"official_data_top30_dedup"
CFG=SimpleNamespace(sources=None,max_names_per_sex_source=None,min_source_rows=20,
    regularization_c=1.0,max_iter=2000,source_fixed_effects=True,
    reuse_cache=True, cache_in=BASE/"official_ipa_features_top30.csv",
    filter_g2p_artifacts=True, source_artifact_threshold=0.25,
    features_out=BASE/"_dedup_features.csv", coef_out=BASE/"_dedup_coef.csv",
    summary_out=BASE/"_dedup_summary.txt", g2p_audit_out=BASE/"_dedup_g2p_audit.csv",
    unparsed_out=BASE/"_dedup_unparsed.csv", on_unparsed="exclude")
df=g["build_feature_table"](CFG)
g["run_regression"](df,CFG)
print("\n--- DEDUP RESULT ---")
print(io.open(BASE/"_dedup_summary.txt",encoding="utf-8").read())
