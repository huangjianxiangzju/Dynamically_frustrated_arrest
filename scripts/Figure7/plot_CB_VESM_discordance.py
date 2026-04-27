"""
Figure 9: CB vs VESM Discordance Analysis
------------------------------------------
Panel A: CB driving force vs VESM evolutionary constraint scatter,
         colored by MD hub category, quadrant crosshairs at medians.
Panel B: Quadrant composition bar (total / MD-annotated / super-hub).

Style: Helvetica, no panel titles, dpi=600, legend below panel A.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from adjustText import adjust_text
from scipy.stats import pearsonr, spearmanr

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# CONFIG
# ==========================================
STEPWISE_PAIRS = ["6_vs_8", "8_vs_10", "10_vs_12", "12_vs_14", "14_vs_16", "16_vs_18"]
VESM_FILE = "SpCas9_VESM3B_full_position_summary.csv"

C_BG     = '#BBBBBB'
C_SINGLE = '#4E79A7'
C_MULTI  = '#E15759'

# ==========================================
# 1. CB: mean stepwise driving force
# ==========================================
dfs_all = []   # all positions (for background scatter)
dfs_md  = []   # MD-annotated positions only (for quadrant/enrichment)
for pair in STEPWISE_PAIRS:
    path = f"CB_results_{pair}_proteinmpnn/position_summary.csv"
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping.")
        continue
    tmp = pd.read_csv(path)
    tmp['driving_force'] = -tmp['CB_bias_zscore']
    dfs_all.append(tmp[['position', 'wt', 'driving_force']])
    dfs_md.append(tmp[['position', 'wt', 'is_MD_switch', 'MD_Roles',
                        'Hub_Overlap_Count', 'driving_force']])

# All positions aggregated (for scatter background)
cb_all = (pd.concat(dfs_all)
          .groupby(['position', 'wt'], as_index=False)['driving_force'].mean()
          .rename(columns={'driving_force': 'mean_CB_force'}))
print(f"CB all positions: {len(cb_all)}")

# MD-annotated positions aggregated (for quadrant bars and labels)
cb_agg = (pd.concat(dfs_md)
          .groupby(['position', 'wt', 'is_MD_switch', 'MD_Roles', 'Hub_Overlap_Count'],
                   as_index=False)['driving_force'].mean()
          .rename(columns={'driving_force': 'mean_CB_force'}))
print(f"CB MD positions: {len(cb_agg)}")

# ==========================================
# 2. VESM
# ==========================================
vesm = pd.read_csv(VESM_FILE)
vesm['constraint'] = -vesm['mean_LLR']
vesm_sub = vesm[['position', 'constraint']].copy()
print(f"VESM: {len(vesm_sub)} positions")

# ==========================================
# 3. MERGE & QUADRANT
# ==========================================
# Full scatter merge: all positions with both CB and VESM scores
merged_full = pd.merge(cb_all, vesm_sub, on='position', how='inner')
# Also merge MD metadata in for coloring
merged_full = pd.merge(merged_full, cb_agg[['position','is_MD_switch','MD_Roles','Hub_Overlap_Count']],
                       on='position', how='left')
merged_full['Hub_Overlap_Count'] = merged_full['Hub_Overlap_Count'].fillna(0).astype(int)
print(f"Merged full (scatter): {len(merged_full)} positions")

# Quadrant analysis on the same full set
cb_med  = merged_full['mean_CB_force'].median()
ves_med = merged_full['constraint'].median()
merged_quad = merged_full  # alias for downstream code

cb_med  = merged_quad['mean_CB_force'].median()
ves_med = merged_quad['constraint'].median()

def assign_quadrant(row):
    hi_cb  = row['mean_CB_force'] > cb_med
    hi_ves = row['constraint'] > ves_med
    if   hi_cb and hi_ves:      return 'Q1_core_driver'
    elif not hi_cb and hi_ves:  return 'Q2_functional_invariant'
    elif hi_cb and not hi_ves:  return 'Q4_plastic_driver'
    else:                       return 'Q3_background'

merged_quad['quadrant'] = merged_quad.apply(assign_quadrant, axis=1)
merged_quad.to_csv("cb_vesm_quadrant_table.csv", index=False)
print("Saved: cb_vesm_quadrant_table.csv")

print("\nQuadrant counts:")
print(merged_quad['quadrant'].value_counts())

for q in ['Q1_core_driver', 'Q2_functional_invariant', 'Q4_plastic_driver', 'Q3_background']:
    sub    = merged_quad[(merged_quad['quadrant'] == q) & (merged_quad['Hub_Overlap_Count'] >= 1)]
    n_sup  = merged_quad[(merged_quad['quadrant'] == q) & (merged_quad['Hub_Overlap_Count'] >= 2)].shape[0]
    print(f"{q}: {merged_quad[merged_quad['quadrant']==q].shape[0]} total, "
          f"{len(sub)} MD-annotated, {n_sup} super-hubs")

# ==========================================
# 4. CORRELATION
# ==========================================
r_p, p_p = pearsonr(merged_quad['mean_CB_force'], merged_quad['constraint'])
r_s, p_s = spearmanr(merged_quad['mean_CB_force'], merged_quad['constraint'])
print(f"\nPearson  r = {r_p:.3f}, p = {p_p:.2e}")
print(f"Spearman r = {r_s:.3f}, p = {p_s:.2e}")

# ==========================================
# 5. FIGURE
# ==========================================
fig = plt.figure(figsize=(18, 7))
gs  = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.28)
ax_main = fig.add_subplot(gs[0])
ax_quad = fig.add_subplot(gs[1])

# --- Panel A: scatter ---
bg     = merged_quad[merged_quad['Hub_Overlap_Count'] == 0]
single = merged_quad[merged_quad['Hub_Overlap_Count'] == 1]
multi  = merged_quad[merged_quad['Hub_Overlap_Count'] >= 2]

ax_main.scatter(bg['mean_CB_force'],     bg['constraint'],
                s=8,  c=C_BG,     alpha=0.35, lw=0, zorder=1,
                label=f'Non-hub residues (n={len(bg)})')
ax_main.scatter(single['mean_CB_force'], single['constraint'],
                s=22, c=C_SINGLE, alpha=0.70, lw=0, zorder=2,
                label=f'Single-category MD hub (n={len(single)})')
ax_main.scatter(multi['mean_CB_force'],  multi['constraint'],
                s=70, c=C_MULTI,  alpha=0.95, edgecolors='black', lw=0.7,
                zorder=4, label=f'Multi-category MD hub / super-hub (n={len(multi)})')

# Median crosshairs
ax_main.axvline(cb_med,  color='#444444', ls='--', lw=1.0, alpha=0.6)
ax_main.axhline(ves_med, color='#444444', ls='--', lw=1.0, alpha=0.6)

# Quadrant labels
x_min, x_max = merged_quad['mean_CB_force'].min(), merged_quad['mean_CB_force'].max()
y_min, y_max = merged_quad['constraint'].min(),    merged_quad['constraint'].max()
px = (x_max - x_min) * 0.02
py = (y_max - y_min) * 0.02

ax_main.text(x_max - px, y_max - py, 'Q1: Core drivers\n(high CB, high VESM)',
             ha='right', va='top', fontsize=13, color='#8B0000',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0E0', alpha=0.7))
ax_main.text(x_min + px, y_max - py, 'Q2: Functional invariants\n(low CB, high VESM)',
             ha='left',  va='top', fontsize=13, color='#1A4D8F',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0EEFF', alpha=0.7))
ax_main.text(x_max - px, y_min + py, 'Q4: Plastic drivers\n(high CB, low VESM)',
             ha='right', va='bottom', fontsize=13, color='#5A3E00',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5D0', alpha=0.7))
ax_main.text(x_min + px, y_min + py, 'Q3: Background\n(low CB, low VESM)',
             ha='left',  va='bottom', fontsize=13, color='#555555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', alpha=0.7))

# Label super-hubs
texts = []
for _, row in multi.iterrows():
    t = ax_main.text(row['mean_CB_force'], row['constraint'],
                     f"{row['wt']}{int(row['position'])}",
                     fontsize=9, fontweight='bold', color='#8B0000')
    texts.append(t)
adjust_text(texts, ax=ax_main,
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.6),
            expand_points=(1.8, 2.5))

# Spearman annotation — bottom-right, away from Q2 quadrant label
ax_main.text(0.97, 0.75,
             f"Spearman r = {r_s:.3f}\np = {p_s:.2e}",
             transform=ax_main.transAxes,
             fontsize=14, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

ax_main.set_xlabel('Mean CB Driving Force (stepwise)', fontsize=20)
ax_main.set_ylabel('Evolutionary Constraint (\u2212mean LLR)', fontsize=20)
ax_main.tick_params(labelsize=18)
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)

# Legend below panel A
ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fontsize=13, frameon=False, ncol=3)

# --- Panel B: quadrant composition bar ---
q_keys   = ['Q1_core_driver', 'Q2_functional_invariant', 'Q4_plastic_driver', 'Q3_background']
q_labels = ['Q1\nCore drivers', 'Q2\nFunctional\ninvariants',
            'Q4\nPlastic drivers', 'Q3\nBackground']
q_colors = [C_MULTI, C_SINGLE, '#F28E2B', C_BG]

for i, (key, label, color) in enumerate(zip(q_keys, q_labels, q_colors)):
    sub     = merged_quad[merged_quad['quadrant'] == key]
    n_total = len(sub)
    n_md    = (sub['Hub_Overlap_Count'] >= 1).sum()
    n_super = (sub['Hub_Overlap_Count'] >= 2).sum()
    # Three bars: total (transparent), MD (solid color), super-hub (hatched)
    ax_quad.bar(i, n_total, color=color, alpha=0.25, edgecolor='black', lw=0.8)
    ax_quad.bar(i, n_md,    color=color, alpha=0.75, edgecolor='black', lw=0.8)
    ax_quad.bar(i, n_super, color=color, alpha=1.00, edgecolor='black', lw=1.2,
                hatch='///')
    ax_quad.text(i, n_total + 4, str(n_total),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

# Legend for bar shading — use actual visual encoding
from matplotlib.patches import Patch
bar_legend = [
    Patch(facecolor='#888888', alpha=0.25, edgecolor='black',
          label='All residues'),
    Patch(facecolor='#888888', alpha=0.75, edgecolor='black',
          label='MD-annotated (≥1 hub)'),
    Patch(facecolor='#888888', alpha=1.00, edgecolor='black', hatch='///',
          label='Super-hub (≥2 hubs)'),
]
ax_quad.legend(handles=bar_legend, loc='upper center', bbox_to_anchor=(0.3, -0.15),
               fontsize=11, frameon=False, ncol=1)

ax_quad.set_xticks(range(4))
ax_quad.set_xticklabels(q_labels, fontsize=12, rotation=30, ha='right')
ax_quad.set_ylabel('Number of residues', fontsize=18)
ax_quad.tick_params(axis='y', labelsize=16)
ax_quad.spines['top'].set_visible(False)
ax_quad.spines['right'].set_visible(False)

plt.savefig('CB_VESM_Discordance.png', dpi=600, bbox_inches='tight')
plt.savefig('CB_VESM_Discordance.pdf', bbox_inches='tight')
print("\nSaved: CB_VESM_Discordance.png/.pdf")
plt.close()

# ==========================================
# 6. TOP RESIDUES PER QUADRANT
# ==========================================
print("\n=== Top 10 per quadrant (MD-annotated, sorted by CB force) ===")
for key, label in zip(q_keys, q_labels):
    sub = merged_quad[(merged_quad['quadrant'] == key) & (merged_quad['Hub_Overlap_Count'] >= 1)]
    sub = sub.sort_values('mean_CB_force', ascending=False).head(10)
    print(f"\n--- {label.replace(chr(10), ' ')} ---")
    print(sub[['position', 'wt', 'mean_CB_force', 'constraint',
               'Hub_Overlap_Count', 'MD_Roles']].to_string(index=False))
