import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

feat = pd.read_csv('official_ipa_features_main_conservative.csv')
de = feat[feat['source']=='official_austria'].copy()

female_de = ['Julia','Lisa','Maria','Anna','Sabrina','Laura','Sandra','Katharina','Monika','Christina']
male_de   = ['Thomas','Michael','Stefan','Andreas','Christian','Klaus','Hans','Peter','Franz','Karl']

print('=== FEMALE names: long vowel marking ===')
for name in female_de:
    rows = de[de['name'].str.lower() == name.lower()].drop_duplicates('name')
    if not rows.empty:
        r = rows.iloc[0]
        print('{:<15} {:<28} pf_long={:.3f}'.format(name, str(r['ipa']), r['pf_long']))

print()
print('=== MALE names: long vowel marking ===')
for name in male_de:
    rows = de[de['name'].str.lower() == name.lower()].drop_duplicates('name')
    if not rows.empty:
        r = rows.iloc[0]
        print('{:<15} {:<28} pf_long={:.3f}'.format(name, str(r['ipa']), r['pf_long']))

print()
# Summary: what proportion of vowels is long in female vs male names
# by checking whether ipa contains ː
de['n_long'] = de['ipa'].str.count('ː')
de['n_phones'] = de['ipa'].str.len()  # rough proxy
print('Mean pf_long: F={:.3f}  M={:.3f}  d={:.3f}'.format(
    de[de['sex']=='F']['pf_long'].mean(),
    de[de['sex']=='M']['pf_long'].mean(),
    (de[de['sex']=='F']['pf_long'].mean() - de[de['sex']=='M']['pf_long'].mean()) /
    de['pf_long'].std()
))
