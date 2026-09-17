"""Compute count-weighted Cohen's d for pf_delrel and pf_distr across all 17 registries,
then append to official_ipa_d_main_conservative_plus_taiwan_czech.csv."""
import csv, math, sys
import pandas as pd
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

NEW_FEATS = ['pf_delrel', 'pf_distr']

def weighted_d_df(df, feat, source):
    """Count-weighted Cohen's d (F - M) for one feature in a features dataframe."""
    rows = {}
    for sex in ['f', 'm']:
        g = df[df['sex'].str.lower() == sex]
        if len(g) == 0:
            return None
        w = g['count'].values.astype(float)
        v = g[feat].values.astype(float)
        mask = ~__import__('numpy').isnan(v)
        w, v = w[mask], v[mask]
        if w.sum() < 10:
            return None
        mu = float((v * w).sum() / w.sum())
        var = float(((v - mu)**2 * w).sum() / w.sum())
        rows[sex] = (mu, var, float(w.sum()))
    if len(rows) < 2:
        return None
    mf, vf, wf = rows['f']
    mm, vm, wm = rows['m']
    sd = math.sqrt((vf + vm) / 2)
    if sd == 0:
        return None
    return {'source': source, 'feature': feat,
            'mean_f': mf, 'mean_m': mm,
            'd': (mf - mm) / sd,
            'nF': wf, 'nM': wm}

new_rows = []

# --- 15 main registries (plus Taiwan if present) ---
# Try the combined plus_taiwan file first; fall back to main_conservative
for fname in ['official_ipa_features_main_conservative_plus_taiwan.csv',
              'official_ipa_features_main_conservative.csv']:
    try:
        feat_main = pd.read_csv(fname, low_memory=False)
        print(f'Loaded {fname}: {len(feat_main)} rows, sources: {sorted(feat_main["source"].unique())}')
        break
    except FileNotFoundError:
        continue

# Filter out artifact rows
if 'g2p_any_artifact' in feat_main.columns:
    feat_main = feat_main[feat_main['g2p_any_artifact'].astype(str) != 'True']

for feat in NEW_FEATS:
    if feat not in feat_main.columns:
        print(f'WARNING: {feat} not in {fname}')
        continue
    for source, grp in feat_main.groupby('source'):
        r = weighted_d_df(grp, feat, source)
        if r:
            new_rows.append(r)
            print(f'  {source:35s} {feat}  d={r["d"]:+.3f}')

# --- Czech (separate pipeline) ---
try:
    feat_cz = pd.read_csv('official_ipa_features_czech_probe.csv')
    print(f'\nLoaded Czech features: {len(feat_cz)} rows')
    for feat in NEW_FEATS:
        if feat not in feat_cz.columns:
            print(f'WARNING: {feat} not in Czech features')
            continue
        r = weighted_d_df(feat_cz, feat, 'official_czech')
        if r:
            new_rows.append(r)
            print(f'  official_czech                      {feat}  d={r["d"]:+.3f}')
except FileNotFoundError:
    print('Czech features file not found.')

print(f'\nComputed {len(new_rows)} new d values.')

# --- Append to d table ---
d_tbl = pd.read_csv('official_ipa_d_main_conservative_plus_taiwan_czech.csv')
print(f'Original d table: {len(d_tbl)} rows, features: {sorted(d_tbl["feature"].unique())}')

# Remove any existing rows for these features (shouldn't exist but be safe)
d_tbl = d_tbl[~d_tbl['feature'].isin(NEW_FEATS)]
new_df = pd.DataFrame(new_rows)
d_extended = pd.concat([d_tbl, new_df], ignore_index=True)
d_extended.to_csv('official_ipa_d_main_conservative_plus_taiwan_czech.csv', index=False)
print(f'Extended d table: {len(d_extended)} rows, features: {sorted(d_extended["feature"].unique())}')

# Summary: direction consistency
print('\n=== Direction summary (F > M = positive d) ===')
for feat in NEW_FEATS:
    sub = d_extended[d_extended['feature'] == feat]
    n_pos = (sub['d'] > 0).sum()
    n_neg = (sub['d'] < 0).sum()
    mean_d = sub['d'].mean()
    print(f'{feat}: mean_d={mean_d:+.3f}, F>M={n_pos}/{len(sub)}, M>F={n_neg}/{len(sub)}')
    print(sub[['source','d']].sort_values('d').to_string(index=False))
    print()
