import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')

d_tbl = pd.read_csv('official_ipa_d_main_conservative_plus_taiwan_czech.csv')
feat_main = pd.read_csv('official_ipa_features_main_conservative.csv')

matched = ['official_australia','official_england','official_usa',
           'official_netherlands','official_czech']
feats_check = ['pf_delrel','pf_ant','pf_distr','n_phones']

print('=== d values for excluded features (language-matched registries) ===')
for feat in feats_check:
    sub = d_tbl[(d_tbl['feature']==feat) & (d_tbl['source'].isin(matched))]
    if len(sub):
        vals = sub.set_index('source')['d']
        print(feat + ':')
        for src in matched:
            if src in vals.index:
                print('  {:30s}  d={:+.3f}'.format(src, vals[src]))
    print()

print('=== pf_delrel: all 17 registries ===')
sub = d_tbl[d_tbl['feature']=='pf_delrel'][['source','d']].sort_values('d')
print(sub.to_string(index=False))
print()

print('=== pf_ant: all 17 registries ===')
sub = d_tbl[d_tbl['feature']=='pf_ant'][['source','d']].sort_values('d')
print(sub.to_string(index=False))
print()

# Feature correlations in registry d (wide format)
pivot = d_tbl.pivot_table(index='source', columns='feature', values='d')
current_13 = ['prop_vowel','pf_son','pf_cont','pf_voi','pf_lo','pf_lat','pf_nas',
              'pf_back','pf_hi','pf_lab','pf_strid','pf_round','pf_cor']
candidates  = ['pf_delrel','pf_ant','pf_distr']
available = [f for f in current_13 + candidates if f in pivot.columns]
corr = pivot[available].corr()
print('=== Correlations of candidate features with current 13 (across 17 registry d values) ===')
for cand in candidates:
    if cand in corr.columns:
        print('\n' + cand + ':')
        vals = corr[cand][current_13].sort_values()
        for feat, val in vals.items():
            print('  {:15s}  r={:+.3f}'.format(feat, val))
