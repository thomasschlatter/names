# -*- coding: utf-8 -*-
"""Can every collected data file be traced to a source?

Policy being enforced: a list we cannot attribute to a source is not usable.
For each data file, look for either
  (a) a producer script that builds it, or
  (b) a mention in one of the dated source-audit documents, or
  (c) a provenance manifest row naming the URL it came from.
Files with none of those are listed so they can be backed up and re-collected.

Usage: python _audit_sources.py
Writes source_attribution_audit.csv
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
SHARED = Path("H:/My Drive/PROJECTS/_RESOURCES/official_data")

DOC_SUFFIXES = (".md", ".txt")
SCRIPT_SUFFIXES = (".py", ".Rmd", ".R")


def read_text(path: Path) -> str:
    try:
        return io.open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""


def gather(folder: Path, suffixes) -> dict:
    out = {}
    if not folder.exists():
        return out
    for name in os.listdir(folder):
        if name.endswith(suffixes):
            out[name] = read_text(folder / name)
    return out


def main() -> None:
    docs = {}
    scripts = {}
    for folder in (SHARED, PROJECT):
        docs.update({f"{folder.name}/{k}": v for k, v in gather(folder, DOC_SUFFIXES).items()})
        scripts.update({f"{folder.name}/{k}": v for k, v in gather(folder, SCRIPT_SUFFIXES).items()})

    # Manifests that record a URL per collected file.
    manifests = {}
    for folder in (SHARED, PROJECT):
        for name in os.listdir(folder) if folder.exists() else []:
            if name.endswith("manifest.csv") or name.endswith("_manifest.csv"):
                try:
                    manifests[f"{folder.name}/{name}"] = pd.read_csv(folder / name)
                except Exception:
                    pass

    targets = []
    for folder in (SHARED, PROJECT):
        if not folder.exists():
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".csv"):
                continue
            if name.endswith("manifest.csv"):
                continue
            targets.append((folder, name))

    rows = []
    for folder, name in targets:
        stem = name[:-4]
        path = folder / name
        try:
            size = path.stat().st_size
        except Exception:
            size = None

        producing = [s for s, text in scripts.items()
                     if re.search(re.escape(name), text) or re.search(re.escape(stem) + r"\b", text)]
        # A script that only READS the file is not a producer; look for a write.
        writers = [s for s in producing
                   if re.search(r"(to_csv|write_csv|write\.csv)[^\n]{0,120}" + re.escape(stem),
                                scripts[s]) or
                      re.search(re.escape(stem) + r"[^\n]{0,120}(to_csv|write_csv|write\.csv)",
                                scripts[s])]
        documented = [d for d, text in docs.items() if name in text or stem in text]
        in_manifest = [m for m, frame in manifests.items()
                       if frame.astype(str).apply(
                           lambda col: col.str.contains(stem, regex=False, na=False)).any().any()]

        if writers:
            verdict = "producer script"
        elif in_manifest:
            verdict = "manifest records source URL"
        elif documented:
            verdict = "named in a source-audit document"
        else:
            verdict = "NO SOURCE FOUND"

        rows.append({"folder": folder.name, "file": name, "bytes": size,
                     "verdict": verdict,
                     "producer": "; ".join(writers[:2]),
                     "documented_in": "; ".join(documented[:2]),
                     "manifest": "; ".join(in_manifest[:1]),
                     "read_by": "; ".join(s for s in producing if s not in writers)[:120]})

    audit = pd.DataFrame(rows)
    audit.to_csv(PROJECT / "audits" / "source_attribution_audit.csv", index=False, encoding="utf-8")
    print(f"data files checked: {len(audit)}")
    print(audit["verdict"].value_counts().to_string())
    unsourced = audit[audit.verdict == "NO SOURCE FOUND"]
    print(f"\nNO SOURCE FOUND ({len(unsourced)}):")
    for row in unsourced.itertuples():
        kb = f"{row.bytes/1024:,.0f} KB" if row.bytes else "?"
        print(f"   {row.folder}/{row.file}  ({kb})"
              + (f"  read by: {row.read_by}" if row.read_by else "  [not read by anything]"))


if __name__ == "__main__":
    main()
