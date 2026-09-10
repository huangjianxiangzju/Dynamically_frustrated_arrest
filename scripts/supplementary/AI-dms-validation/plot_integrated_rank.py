"""
Supplementary Figure 14: Integrated CB + VESM + MD Percentile Rank
------------------------------------------------------
Panel A: Scatter pct_cb vs pct_vesm, dot size = md_n, colored by evidence_count.
Panel B: Top 30 residues by combined_score, stacked bar of percentile contributions.

Style: Helvetica, no panel titles, dpi=600, legends below panels.
"""

import os
from pathlib import Path
from PIL import Image
import argparse
ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description="Reproduce Supplementary Figure 14 and integrated residue ranks.")
parser.add_argument("--data", type=Path, default=ROOT / "data/AI-validation-[CB,VESM]")
parser.add_argument("--output", type=Path, default=ROOT / "figures/submission/Supplementary_figures")
parser.add_argument("--table-output", type=Path, default=None)
args = parser.parse_args()
DATA = args.data.resolve()
OUT = args.output.resolve()
TABLE_OUT = args.table_output.resolve() if args.table_output else DATA
OUT.mkdir(exist_ok=True, parents=True)
TABLE_OUT.mkdir(exist_ok=True, parents=True)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import rankdata
from adjustText import adjust_text

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# CONFIG
# ==========================================
STEPWISE_PAIRS = ["6_vs_8", "8_vs_10", "10_vs_12", "12_vs_14", "14_vs_16", "16_vs_18"]
VESM_FILE      = "SpCas9_VESM3B_full_position_summary.csv"
MD_FILE        = "full_superset.csv"

DOMAIN_MAP = [
    (1,    59,  'RuvC-I'),
    (60,   94,  'BH'),
    (95,   176, 'REC1-A'),
    (177,  305, 'REC2'),
    (306,  495, 'REC1-B'),
    (496,  717, 'REC3'),
    (718,  764, 'RuvC-II'),
    (765,  780, 'L1'),
    (781,  905, 'HNH'),
    (906,  918, 'L2'),
    (919,  1100,'RuvC-III'),
    (1101, 1368,'PI'),
]

def get_domain(pos):
    for lo, hi, name in DOMAIN_MAP:
        if lo <= pos <= hi:
            return name
    return 'Unknown'

# ==========================================
# 1. CB
# ==========================================
dfs_all = []   # all 1368 positions for proper percentile ranking
for pair in STEPWISE_PAIRS:
    path = f"CB_results_{pair}_proteinmpnn/position_summary.csv"
    if not (DATA / path).exists():
        raise FileNotFoundError(DATA / path)
    tmp = pd.read_csv(DATA / path)
    tmp['driving_force'] = -tmp['CB_bias_zscore']
    dfs_all.append(tmp[['position', 'wt', 'driving_force']])

# All positions — used for CB percentile rank so non-MD dots are not pinned to 0
cb_all = (pd.concat(dfs_all)
          .groupby(['position', 'wt'], as_index=False)['driving_force'].mean()
          .rename(columns={'driving_force': 'cb_force'}))

# Use the verified Figure 7 one-row-per-residue annotation method.
md_meta = pd.read_csv(DATA / MD_FILE)
categories = ['Switch', 'GCCM', 'SB_hub', 'Hydro_hub', 'BC']
assert md_meta['Residue'].is_unique and len(md_meta) == 311
assert md_meta[categories].eq('Y').sum().tolist() == [84, 54, 46, 90, 89]
md_meta['Hub_Overlap_Count'] = md_meta[categories].eq('Y').sum(axis=1)
assert (md_meta['Hub_Overlap_Count'] >= 2).sum() == 52
md_meta['MD_Roles'] = md_meta.apply(
    lambda r: ' | '.join(c for c in categories if r[c] == 'Y'), axis=1)
cb_md = md_meta.rename(columns={'Residue': 'position'})[
    ['position', 'MD_Roles', 'Hub_Overlap_Count']]
cb_agg = pd.merge(cb_all, cb_md, on='position', how='left', validate='one_to_one')
assert cb_agg['position'].is_unique and len(cb_agg) == 1368

# ==========================================
# 2. VESM
# ==========================================
vesm = pd.read_csv(DATA / VESM_FILE)
vesm['vesm_constraint'] = -vesm['mean_LLR']
vesm_sub = vesm[['position', 'vesm_constraint']].copy()
print(f"VESM: {len(vesm_sub)} positions")

# ==========================================
# 3. MD
# ==========================================
md = pd.read_csv(DATA / MD_FILE)
md = md.rename(columns={'Residue': 'position', 'n': 'md_n'})
md_sub = md[['position', 'md_n']].copy()
assert md_sub['position'].is_unique
expected = md_meta.set_index('Residue')['Hub_Overlap_Count'].sort_index()
assert np.array_equal(md_sub.set_index('position')['md_n'].sort_index().to_numpy(), expected.to_numpy())
print(f"MD superset: {len(md_sub)} residues")

# ==========================================
# 4. MERGE
# ==========================================
merged = pd.merge(cb_agg, vesm_sub, on='position', how='outer')
merged = pd.merge(merged, md_sub,   on='position', how='outer')
merged['md_n'] = merged['md_n'].fillna(0).astype(int)
merged['wt']   = merged['wt'].fillna('?')
assert merged['position'].is_unique and len(merged) == 1368
assert merged[['cb_force', 'vesm_constraint']].notna().all().all()
merged['domain'] = merged['position'].apply(get_domain)
print(f"Merged: {len(merged)} total positions")

# ==========================================
# 5. PERCENTILE RANKS
# ==========================================
def pct_rank(series):
    vals = series.copy()
    mask = vals.notna()
    ranks = np.zeros(len(vals))
    ranks[mask] = rankdata(vals[mask], method='average') / mask.sum() * 100
    return ranks

merged['pct_cb']   = pct_rank(merged['cb_force'])
merged['pct_vesm'] = pct_rank(merged['vesm_constraint'])
merged['pct_md']   = pct_rank(merged['md_n'])

merged['combined_score'] = merged[['pct_cb', 'pct_vesm', 'pct_md']].mean(axis=1)
merged['evidence_count'] = (
    (merged['pct_cb']   > 50).astype(int) +
    (merged['pct_vesm'] > 50).astype(int) +
    (merged['pct_md']   > 50).astype(int)
)

# ==========================================
# 6. SAVE TABLE
# ==========================================
out_cols = ['position', 'wt', 'domain', 'cb_force', 'vesm_constraint', 'md_n',
            'pct_cb', 'pct_vesm', 'pct_md', 'combined_score', 'evidence_count', 'MD_Roles']
merged_out = (merged[out_cols]
              .sort_values('combined_score', ascending=False)
              .round({'cb_force': 4, 'vesm_constraint': 3,
                      'pct_cb': 1, 'pct_vesm': 1, 'pct_md': 1, 'combined_score': 2}))
merged_out.to_csv(TABLE_OUT / "integrated_rank_table.csv", index=False)
merged[out_cols].sort_values('combined_score', ascending=False).to_csv(
    TABLE_OUT / "integrated_rank_table_full_precision.csv", index=False)
merged[['position', 'wt', 'domain', 'evidence_count']].query('evidence_count == 3').sort_values('position').to_csv(
    TABLE_OUT / "three_method_consensus_residues.csv", index=False)
assert merged['evidence_count'].value_counts().sort_index().tolist() == [267, 606, 412, 83]
print("\nSaved: integrated_rank_table.csv")

print("\n=== Evidence breakdown ===")
for n in [3, 2, 1, 0]:
    sub = merged[merged['evidence_count'] == n]
    print(f"  {n} methods above median: {len(sub)} residues")

print("\n=== Top 20 by combined score ===")
print(merged_out.head(20)[['position', 'wt', 'domain', 'pct_cb', 'pct_vesm', 'pct_md',
                            'combined_score', 'evidence_count', 'MD_Roles']].to_string(index=False))

# ==========================================
# 7. FIGURE
# ==========================================
ev_colors = {0: '#CCCCCC', 1: '#4E79A7', 2: '#F28E2B', 3: '#E15759'}
ev_labels = {
    0: f"No method above median (n={( merged['evidence_count']==0).sum()})",
    1: f"One method (n={(  merged['evidence_count']==1).sum()})",
    2: f"Two methods (n={( merged['evidence_count']==2).sum()})",
    3: f"Three methods (n={(merged['evidence_count']==3).sum()})",
}
md_size_map = {0: 8, 1: 18, 2: 35, 3: 55, 4: 75, 5: 95}

np.random.seed(0)
fig = plt.figure(figsize=(18, 8))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.32)
ax_scatter = fig.add_subplot(gs[0])
ax_bar     = fig.add_subplot(gs[1])

# --- Panel A: pct_cb vs pct_vesm ---
for ev in [0, 1, 2, 3]:
    sub = merged[merged['evidence_count'] == ev]
    ax_scatter.scatter(
        sub['pct_cb'], sub['pct_vesm'],
        s=sub['md_n'].clip(0, 5).map(md_size_map),
        c=ev_colors[ev],
        alpha=0.45 if ev == 0 else 0.85,
        lw=0.5, edgecolors='none' if ev < 2 else 'black',
        zorder=ev + 1, label=ev_labels[ev]
    )

# Label top 15
top15  = merged_out.head(15)
texts  = []
for _, row in top15.iterrows():
    t = ax_scatter.text(row['pct_cb'], row['pct_vesm'],
                        f"{row['wt']}{int(row['position'])}",
                        fontsize=8, fontweight='bold', color='#5A0000')
    texts.append(t)
adjust_text(texts, ax=ax_scatter, iter_lim=500,
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.5),
            expand_points=(2.5, 3.5))

ax_scatter.axvline(50, color='#777777', ls='--', lw=0.8, alpha=0.6)
ax_scatter.axhline(50, color='#777777', ls='--', lw=0.8, alpha=0.6)

# Quadrant text — top-left corners, on top of all scatter dots
#ax_scatter.text(51, 99, 'CB + VESM', ha='left', va='top',
#                fontsize=13, color='#8B0000', style='italic', zorder=10)
#ax_scatter.text(1, 99, 'VESM only', ha='left', va='top',
#                fontsize=13, color='#1A4D8F', style='italic', zorder=10)
#ax_scatter.text(51,  1, 'CB only',   ha='left', va='bottom',
#                fontsize=13, color='#5A3E00', style='italic', zorder=10)

ax_scatter.set_xlabel('CB percentile rank', fontsize=20)
ax_scatter.set_ylabel('VESM substitution-intolerance percentile rank', fontsize=20)
ax_scatter.set_xlim(0, 101)
ax_scatter.set_ylim(0, 101)
ax_scatter.tick_params(labelsize=18)
ax_scatter.spines['top'].set_visible(False)
ax_scatter.spines['right'].set_visible(False)

# Legend below panel A
ax_scatter.legend(title='Methods above median', title_fontsize=11,
                  loc='upper center', bbox_to_anchor=(0.5, -0.14),
                  fontsize=11, frameon=False, ncol=2)

# --- Panel B: stacked bar top 30 ---
top30   = merged_out.head(30).copy().reset_index(drop=True)
labels30 = [f"{row['wt']}{int(row['position'])}" for _, row in top30.iterrows()]
x = np.arange(len(top30))
w = 0.6

ax_bar.bar(x, top30['pct_cb'],   w, label='CB percentile',   color='#F28E2B', alpha=0.85)
ax_bar.bar(x, top30['pct_vesm'], w, bottom=top30['pct_cb'],
           label='VESM percentile', color='#4E79A7', alpha=0.85)
ax_bar.bar(x, top30['pct_md'],   w, bottom=top30['pct_cb'] + top30['pct_vesm'],
           label='MD percentile',   color='#E15759', alpha=0.85)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels30, rotation=75, ha='right', fontsize=7.5)
ax_bar.set_ylabel('Summed percentile scores', fontsize=18)
ax_bar.tick_params(axis='y', labelsize=16)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

# Legend below panel B
ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              fontsize=11, frameon=False, ncol=3)

fig.text(.035, .91, 'A', fontsize=22)
fig.text(.565, .91, 'B', fontsize=22)
# Render vector artwork directly at sufficient pixel density; no raster upsampling.
fig.savefig(OUT / 'Supplementary_Figure_14.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUT / 'Supplementary_Figure_14.pdf',
            bbox_inches='tight', facecolor='white')
with Image.open(OUT / 'Supplementary_Figure_14.png') as im:
    im.convert('RGB').save(OUT / 'Supplementary_Figure_14.tif',
                          dpi=(600, 600), compression='tiff_lzw')
fig.savefig(OUT / 'preview.png', dpi=100, bbox_inches='tight', facecolor='white')
print("\nSaved: Supplementary_Figure_14.png/.pdf/.tif")
plt.close()
