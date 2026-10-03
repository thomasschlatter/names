# -*- coding: utf-8 -*-
import os, io, glob, re
import pandas as pd
WP="H:/My Drive/PROJECTS/_RESOURCES/LEXICONS/wikipron/wikipron-master/data/scrape/tsv"
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
LANG=dict(re.findall(r'"(official_[a-z0-9_]+)":\s*\("([^"]+)"', m.group(1)))
files=os.listdir(WP)
cache={}
def refset(iso):
    if iso in cache: return cache[iso]
    fns=[f for f in files if f.startswith(iso+"_") and f.endswith(".tsv") and "filtered" not in f]
    s=set()
    for fn in fns:
        with io.open(os.path.join(WP,fn),encoding="utf-8") as fh:
            for line in fh:
                p=line.split("\t")
                if len(p)==2: s.add(p[0].lower())
    cache[iso]=(s,len(fns))
    return cache[iso]
rows=[]
for f in sorted(glob.glob("official_data_top30/official_*.csv")):
    stem=os.path.basename(f)[:-4]; voice=LANG.get(stem)
    iso=ISO.get(voice) if voice else None
    d=pd.read_csv(f); d.columns=[c.strip().lower() for c in d.columns]
    names=[str(n).strip() for n in d["name"]] if "name" in d.columns else []
    if not iso:
        rows.append((stem,voice or "-","none",len(names),0)); continue
    s,nf=refset(iso)
    hit=sum(1 for n in names if n.lower() in s)
    rows.append((stem,voice,f"{iso}({nf} files,{len(s):,})",len(names),hit))
out=pd.DataFrame(rows,columns=["source","voice","reference","names","with_ref"])
out["pct"]=(100*out.with_ref/out.names.clip(lower=1)).round(1)
out.to_csv("audits/_validation_coverage.csv",index=False,encoding="utf-8")
tot=out.names.sum(); hit=out.with_ref.sum()
print(f"sources {len(out)}; names {tot}; with human reference {hit} ({100*hit/tot:.1f}%)")
band=lambda lo,hi: out[(out.pct>=lo)&(out.pct<hi)]
for lo,hi,lab in [(0,1,"0% (no reference for any name)"),(1,25,"1-25%"),(25,50,"25-50%"),(50,80,"50-80%"),(80,101,"80-100%")]:
    b=band(lo,hi); print(f"  {lab:32s} {len(b):3d} sources, {b.names.sum():5d} names")
print("\nsources with NO validatable name:")
print(", ".join(sorted(out[out.pct==0].source.str.replace('official_',''))))
