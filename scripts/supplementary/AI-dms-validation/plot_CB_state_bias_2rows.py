"""
Figure S10: CB Directionality Analysis
----------------------------------------
Panel A (top two rows): Flipper residue heatmap split into two halves for legibility.
Panel B: Pre-transition-biased MD-role enrichment across transitions.
Panel C: Post-transition-biased MD-role enrichment across transitions.

Style: Helvetica, no panel titles, dpi=600, legend below panel B.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import fisher_exact

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# CONFIG
# ==========================================
STEPWISE_PAIRS    = ["6_vs_8", "8_vs_10", "10_vs_12", "12_vs_14", "14_vs_16", "16_vs_18"]
PAIR_LABELS       = ["6-nt->8-nt", "8-nt->10-nt", "10-nt->12-nt", "12-nt->14-nt", "14-nt->16-nt", "16-nt->18-nt"]
MD_ROLES          = ["Allosteric_Switch", "GCCM_Hub", "SaltBridge_Hub",
                     "Hydrophobic_Hub", "Centrality_Hub"]
ZSCORE_THRESHOLD  = 1.0

ROLE_COLORS = {
    'Allosteric_Switch': '#E15759',
    'GCCM_Hub':          '#4E79A7',
    'SaltBridge_Hub':    '#F28E2B',
    'Hydrophobic_Hub':   '#76B7B2',
    'Centrality_Hub':    '#59A14F',
}

# ==========================================
# 1. LOAD ALL STEPWISE DATA
# ==========================================
records = []
for pair in STEPWISE_PAIRS:
    path = f"CB_results_{pair}_proteinmpnn/position_summary.csv"
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping.")
        continue
    df = pd.read_csv(path)
    df['transition'] = pair
    df['direction']  = 'neutral'
    df.loc[df['CB_bias_zscore'] >=  ZSCORE_THRESHOLD, 'direction'] = 'state2_biased'
    df.loc[df['CB_bias_zscore'] <= -ZSCORE_THRESHOLD, 'direction'] = 'state1_biased'
    records.append(df)

all_df = pd.concat(records, ignore_index=True)
print(f"Loaded {len(all_df)} position-transition records.")
print(f"Direction counts:\n{all_df['direction'].value_counts()}\n")

all_df.to_csv("cb_state_bias_table.csv", index=False)
print("Saved: cb_state_bias_table.csv")

# ==========================================
# 2. ENRICHMENT
# ==========================================
enrich_records = []
for pair in STEPWISE_PAIRS:
    sub   = all_df[all_df['transition'] == pair]
    total = len(sub)
    for direction in ['state1_biased', 'state2_biased']:
        dir_sub = sub[sub['direction'] == direction]
        for role in MD_ROLES:
            in_dir_in_role    = dir_sub['MD_Roles'].fillna('').str.contains(role).sum()
            in_dir_not_role   = len(dir_sub) - in_dir_in_role
            not_dir_in_role   = sub[sub['direction'] != direction]['MD_Roles'].fillna('').str.contains(role).sum()
            not_dir_not_role  = total - in_dir_in_role - in_dir_not_role - not_dir_in_role
            _, p = fisher_exact([[in_dir_in_role, in_dir_not_role],
                                  [not_dir_in_role, not_dir_not_role]],
                                 alternative='greater')
            frac = in_dir_in_role / len(dir_sub) if len(dir_sub) > 0 else 0
            enrich_records.append({
                'transition': pair, 'direction': direction, 'MD_role': role,
                'n_role_in_dir': in_dir_in_role, 'frac_in_dir': frac, 'fisher_p': p
            })

enrich_df = pd.DataFrame(enrich_records)
enrich_df.to_csv("cb_directionality_enrichment.csv", index=False)
print("Saved: cb_directionality_enrichment.csv")

# ==========================================
# 3. FLIPPER RESIDUES
# ==========================================
biased_only = all_df[all_df['direction'] != 'neutral'][
    ['position', 'wt', 'transition', 'direction', 'MD_Roles', 'Hub_Overlap_Count', 'CB_bias_zscore']]
pivot = biased_only.pivot_table(
    index=['position', 'wt', 'MD_Roles', 'Hub_Overlap_Count'],
    columns='transition', values='direction', aggfunc='first')
pivot = pivot.reindex(columns=STEPWISE_PAIRS)

def is_flipper(row):
    vals = row.dropna().values
    return ('state1_biased' in vals) and ('state2_biased' in vals)

flipper_mask = pivot.apply(is_flipper, axis=1)
flippers     = pivot[flipper_mask].reset_index()
print(f"\nFlipper residues: {len(flippers)}")
flippers.to_csv("cb_direction_flippers.csv", index=False)
print("Saved: cb_direction_flippers.csv")

# ==========================================
# 4. FIGURE
# ==========================================
# Layout: 4 rows x 2 cols
#   row 0: heatmap top half  (colspan 2)
#   row 1: heatmap bot half  (colspan 2)
#   row 2: panel B (left) + panel C (right)
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(
    3, 2,
    height_ratios=[1, 1, 1.1],
    hspace=0.55,
    wspace=0.32,
)
ax_a1 = fig.add_subplot(gs[0, :])   # heatmap row 1
ax_a2 = fig.add_subplot(gs[1, :])   # heatmap row 2
ax_b1 = fig.add_subplot(gs[2, 0])   # Panel B
ax_b2 = fig.add_subplot(gs[2, 1])   # Panel C

# ------------------------------------------
# Panel B / C: enrichment bar charts
# ------------------------------------------
x     = np.arange(len(STEPWISE_PAIRS))
width = 0.15

for ax, direction, ylabel_suffix in [
    (ax_b1, 'state1_biased', 'Pre-transition residues'),
    (ax_b2, 'state2_biased', 'Post-transition residues'),
]:
    sub = enrich_df[enrich_df['direction'] == direction]
    for i, role in enumerate(MD_ROLES):
        role_data = sub[sub['MD_role'] == role]
        fracs = [
            role_data[role_data['transition'] == p]['frac_in_dir'].values[0]
            if len(role_data[role_data['transition'] == p]) > 0 else 0
            for p in STEPWISE_PAIRS
        ]
        ax.bar(x + i * width, fracs, width,
               label=role.replace('_', ' '),
               color=ROLE_COLORS[role], alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(PAIR_LABELS, fontsize=14, rotation=20, ha='right')
    ax.set_ylabel(f'Fraction in allosteric hub category\n({ylabel_suffix})', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend below panel B only
ax_b1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28),
             fontsize=11, frameon=False, ncol=3)

# ------------------------------------------
# Panel A: split flipper heatmap
# ------------------------------------------
if len(flippers) > 0:
    direction_to_num = {'state1_biased': -1, 'state2_biased': 1}
    heatmap_data = (flippers[STEPWISE_PAIRS]
                    .map(lambda x: direction_to_num.get(x, x) if isinstance(x, str) else x)
                    .fillna(0)
                    .values)

    # Sort by Hub_Overlap_Count desc, then position asc
    sort_idx = flippers.sort_values(
        ['Hub_Overlap_Count', 'position'], ascending=[False, True]).index
    heatmap_data   = heatmap_data[sort_idx]
    flippers_sorted = flippers.loc[sort_idx].reset_index(drop=True)

    # Build x-axis labels
    labels = []
    for _, row in flippers_sorted.iterrows():
        md = row['MD_Roles']
        tag = ''
        if pd.notna(md) and md not in ('None', ''):
            abbr = (md.replace('Allosteric_Switch', 'AS')
                      .replace('GCCM_Hub', 'GH')
                      .replace('SaltBridge_Hub', 'SB')
                      .replace('Hydrophobic_Hub', 'HH')
                      .replace('Centrality_Hub', 'CH'))
            tag = f' [{abbr[:10]}]'
        labels.append(f"{row['wt']}{int(row['position'])}{tag}")

    n_total  = len(flippers_sorted)
    n_half   = (n_total + 1) // 2          # ceiling split: first half gets extra if odd
    idx_top  = np.arange(0, n_half)
    idx_bot  = np.arange(n_half, n_total)

    im = None
    for ax, idx in [(ax_a1, idx_top), (ax_a2, idx_bot)]:
        if len(idx) == 0:
            ax.axis('off')
            continue
        chunk      = heatmap_data[idx]       # shape (n_chunk, n_transitions)
        chunk_lbls = [labels[i] for i in idx]

        im = ax.imshow(chunk.T, aspect='auto', cmap='RdBu_r',
                       vmin=-1, vmax=1, interpolation='nearest')
        ax.set_yticks(range(len(STEPWISE_PAIRS)))
        ax.set_yticklabels(PAIR_LABELS, fontsize=13)
        ax.set_xticks(range(len(chunk_lbls)))
        ax.set_xticklabels(chunk_lbls, rotation=90, ha='center', fontsize=12)
        ax.tick_params(axis='x', length=2)
        ax.set_ylabel('Transition', fontsize=14)

    ax_a2.set_xlabel('Residue (sorted by hub overlap count)', fontsize=14)

    # Single shared colorbar anchored to the right of ax_a2
    if im is not None:
        cbar = fig.colorbar(im, ax=[ax_a1, ax_a2],
                            orientation='vertical',
                            fraction=0.012, pad=0.01, shrink=0.85)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['Pre-transition', 'Neutral', 'Post-transition'], fontsize=11)
        cbar.ax.tick_params(labelsize=11)

else:
    for ax in (ax_a1, ax_a2):
        ax.text(0.5, 0.5, 'No flipper residues found at current threshold',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')

plt.savefig('FigureS10_CB_Directionality.png', dpi=600, bbox_inches='tight')
plt.savefig('FigureS10_CB_Directionality.pdf', bbox_inches='tight')
print("\nSaved: FigureS10_CB_Directionality.png/.pdf")
plt.close()

