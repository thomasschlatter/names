# -*- coding: utf-8 -*-
"""How much of the registry set can be validated against WikiPron human IPA?"""
import os, io, glob, re, sys
import pandas as pd
WP="H:/My Drive/PROJECTS/_RESOURCES/LEXICONS/wikipron/wikipron-master/data/scrape/tsv"

# espeak voice -> WikiPron ISO-639-3
ISO={"sq":"sqi","es":"spa","hy":"hye","en":"eng","de":"deu","az":"aze","be":"bel",
 "nl":"nld","hr":"hbs","pt-br":"por","bg":"bul","fr":"fra","zh":"cmn","cs":"ces",
 "da":"dan","et":"est","fo":"fao","fi":"fin","kl":"kal","ka":"kat","hu":"hun",
 "is":"isl","id":"ind","fa":"fas","he":"heb","it":"ita","ja":"jpn","kk":"kaz",
 "ky":"kir","lv":"lav","lt":"lit","ar":"ara","mi":"mri","mk":"mkd","nb":"nob",
 "pl":"pol","pt":"por","ro":"ron","ru":"rus","sk":"slk","sl":"slv","af":"afr",
 "ko":"kor","ca":"cat","eu":"eus","sv":"swe","tr":"tur","uk":"ukr","uz":"uzb",
 "pap":None,"poly":None}

src=io.open("paper_registry.Rmd",encoding="utf-8").read()
m=re.search(r"LANG_BY_SOURCE = \{(.*?)\n\}", src, re.S)
LANG={}
for k,v in re.findall(r'"(official_[a-z0-9_]+)":\s*\("([^"]+)"', m.group(1)):
    LANG[k]=v

files=os.listdir(WP)
def wp_file(iso):
    if not iso: return None
    c=[f for f in files if f.startswith(iso+"_") and f.endswith("_broad.tsv") and "filtered" not in f]
    if not c: c=[f for f in files if f.startswith(iso+"_") and f.endswith(".tsv")]
    return c[0] if c else None

cache={}
def load(fn):
    if fn not in cache:
        s=set()
        with io.open(os.path.join(WP,fn),encoding="utf-8") as fh:
            for line in fh:
                p=line.split("\t")
                if len(p)==2: s.add(p[0])
        cache[fn]=s
    return cache[fn]

rows=[]
for f in sorted(glob.glob("official_data_top30/official_*.csv")):
    stem=os.path.basename(f)[:-4]
    voice=LANG.get(stem)
    if voice is None:
        rows.append((stem,"-","no language mapping",0,0)); continue
    iso=ISO.get(voice); fn=wp_file(iso)
    d=pd.read_csv(f); d.columns=[c.strip().lower() for c in d.columns]
    names=set(d["name"].astype(str).str.strip()) if "name" in d.columns else set()
    if not fn:
        rows.append((stem,voice,"NO WikiPron reference",len(names),0)); continue
    ref=load(fn)
    hit=sum(1 for n in names if n in ref)
    rows.append((stem,voice,fn,len(names),hit))

out=pd.DataFrame(rows,columns=["source","voice","reference","names","with_ref"])
out["pct"]=(100*out.with_ref/out.names.clip(lower=1)).round(1)
out.to_csv("audits/_validation_coverage.csv",index=False,encoding="utf-8")
print(f"sources: {len(out)}")
print(f"  no WikiPron reference at all : {(out.reference=='NO WikiPron reference').sum()}")
print(f"  no language mapping          : {(out.reference=='no language mapping').sum()}")
ok=out[~out.reference.isin(['NO WikiPron reference','no language mapping'])]
print(f"  reference available          : {len(ok)}")
print(f"\nvalidatable names: {ok.with_ref.sum()} of {ok.names.sum()} ({100*ok.with_ref.sum()/ok.names.sum():.1f}%)")
print("\nper-source coverage, worst 18:")
print(ok.sort_values('pct')[['source','voice','names','with_ref','pct']].head(18).to_string(index=False))
print("\nbest 10:")
print(ok.sort_values('pct',ascending=False)[['source','voice','names','with_ref','pct']].head(10).to_string(index=False))
