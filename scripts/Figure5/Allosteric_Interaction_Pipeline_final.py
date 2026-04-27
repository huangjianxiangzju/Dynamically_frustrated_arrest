import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import warnings
import re

try:
    import kaleido
except ImportError:
    print("Notice: kaleido is not installed. PNG export for Plotly might be limited.")

warnings.filterwarnings('ignore')

# --- FONT SETTINGS ---
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# 0. ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(description="SpCas9 Allosteric Interaction Network Pipeline")
parser.add_argument("mode", choices=['saltbridge', 'hbond', 'hydro'],
                    help="Interaction type: saltbridge, hbond, or hydro")
parser.add_argument("--cutoff", type=float, default=None,
                    help="Override default occupancy threshold")

args = parser.parse_args()
MODE = args.mode

if MODE == 'saltbridge':
    FILE_NAME, PREFIX, TITLE_NAME = 'competitive_salt_bridges.csv', 'SB', 'Salt Bridges'
    OCCUPANCY_THRESHOLD = args.cutoff if args.cutoff is not None else 0.50
    MIN_FLOW_THRESHOLD = 3
elif MODE == 'hbond':
    FILE_NAME, PREFIX, TITLE_NAME = 'competitive_hbonds.csv', 'Hbond', 'Hydrogen Bonds'
    OCCUPANCY_THRESHOLD = args.cutoff if args.cutoff is not None else 0.30
    MIN_FLOW_THRESHOLD = 3
elif MODE == 'hydro':
    FILE_NAME, PREFIX, TITLE_NAME = 'competitive_hydrophobic.csv', 'Hydro', 'Hydrophobic Contacts'
    OCCUPANCY_THRESHOLD = args.cutoff if args.cutoff is not None else 0.40
    MIN_FLOW_THRESHOLD = 5

print(f"==================================================")
print(f"  {TITLE_NAME} Allosteric Analysis Pipeline")
print(f"  File: {FILE_NAME}")
print(f"  Occupancy threshold: {OCCUPANCY_THRESHOLD}")
print(f"  Sankey flow filter: > {MIN_FLOW_THRESHOLD} pairs")
print(f"==================================================")

raw_states = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']
display_states = [s.replace('nt', '-nt') for s in raw_states]
state_mapping = dict(zip(raw_states, display_states))

# ==========================================
# 1. DATA LOADING
# ==========================================
master_dict = {}
for rs in raw_states:
    file_path = f"{rs}/{FILE_NAME}"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            int_type = row['Interaction_Type']
            pos = str(row['Pos_Residue'])
            target = str(row['Target_Residue'])
            pair = sorted([pos, target])
            sb_id = f"{pair[0]}_{pair[1]}"

            if sb_id not in master_dict:
                master_dict[sb_id] = {s: 0.0 for s in display_states}
                master_dict[sb_id]['Type'] = int_type

            master_dict[sb_id][state_mapping[rs]] = max(master_dict[sb_id][state_mapping[rs]], row['Occupancy'])
    else:
        print(f"Warning: {file_path} not found")

df_master = pd.DataFrame.from_dict(master_dict, orient='index')
if df_master.empty:
    print(f"Error: No data loaded. Check {FILE_NAME} files.")
    sys.exit(1)

df_bin = (df_master[display_states] >= OCCUPANCY_THRESHOLD).astype(int)
df_bin = df_bin[df_bin.sum(axis=1) > 0]

# Separate constitutive interactions
constitutive_mask = (df_bin.sum(axis=1) == len(display_states))
constitutive_indices = df_bin[constitutive_mask].index
df_constitutive = df_master.loc[constitutive_indices]

if not df_constitutive.empty:
    df_constitutive = df_constitutive.sort_values(by=display_states[0], ascending=False)
    out_const = f"{PREFIX}_Constitutive_Interactions.csv"
    df_constitutive.to_csv(out_const)
    print(f"  Separated {len(df_constitutive)} constitutive interactions -> {out_const}")

df_bin = df_bin[~constitutive_mask]
print(f"  Dynamic {TITLE_NAME} for network analysis: {len(df_bin)}")

if len(df_bin) == 0:
    print("  No dynamic data. Try lowering --cutoff.")
    sys.exit(1)

# ==========================================
# A. Sankey Diagram
# ==========================================
print("Generating A: Sankey diagram...")
first_app = df_bin.apply(lambda x: x.idxmax(), axis=1)
last_app = df_bin.apply(lambda x: x[::-1].idxmax(), axis=1)
sankey_df = pd.DataFrame({'First': first_app, 'Last': last_app})
counts = sankey_df.groupby(['First', 'Last']).size().reset_index(name='Count')
counts = counts[counts['Count'] > MIN_FLOW_THRESHOLD]

labels = [f"Start: {s}" for s in display_states] + [f"End: {s}" for s in display_states]
source_indices = [display_states.index(f) for f in counts['First']]
target_indices = [display_states.index(l) + len(display_states) for l in counts['Last']]
node_x = [0.01] * 7 + [0.99] * 7
node_y = [i / 6.0 for i in range(7)] * 2

fig_sankey = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
              label=labels, x=node_x, y=node_y, color="royalblue"),
    link=dict(source=source_indices, target=target_indices,
              value=counts['Count'], color="rgba(173, 216, 230, 0.5)")
)])

fig_sankey.update_layout(font=dict(family="Helvetica, Arial", size=14), width=1000, height=800)
fig_sankey.write_html(f"{PREFIX}_A_Sankey_Diagram.html")
try:
    fig_sankey.write_image(f"{PREFIX}_A_Sankey_Diagram.png", scale=2)
except:
    pass

# ==========================================
# B. Net Flux
# ==========================================
print("Generating B: Net flux plot...")
transitions = []
for i in range(1, len(display_states)):
    p, c = display_states[i-1], display_states[i]
    step = f"{p} \u2192 {c}"
    app = ((df_bin[c] == 1) & (df_bin[p] == 0)).sum()
    dis = ((df_bin[c] == 0) & (df_bin[p] == 1)).sum()
    transitions.append({'Step': step, 'Appearing': app, 'Disappearing': dis, 'Net': app - dis})

df_trans = pd.DataFrame(transitions)
fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(df_trans['Step'], df_trans['Appearing'], color='forestgreen', alpha=0.7, label='Appearing (+)')
ax.bar(df_trans['Step'], -df_trans['Disappearing'], color='crimson', alpha=0.7, label='Disappearing (\u2212)')
ax.plot(df_trans['Step'], df_trans['Net'], marker='o', color='black', linewidth=2.5, label='Net flux')
ax.axhline(0, color='black', lw=1, linestyle='--')
ax.set_ylabel('Number of residues', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
plt.xticks(rotation=45, ha='right')
ax.legend(fontsize=14, loc='upper right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PREFIX}_B_Net_Flux.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{PREFIX}_B_Net_Flux.pdf", bbox_inches='tight')
plt.close()

# ==========================================
# C. Overlap Matrix
# ==========================================
print("Generating C: Overlap heatmap...")
fig, ax = plt.subplots(figsize=(8, 6))
overlap = df_bin.T.dot(df_bin)
sns.heatmap(overlap, annot=True, fmt="d", cmap="YlGnBu",
            cbar_kws={'label': f'Shared {TITLE_NAME}'},
            annot_kws={'size': 14}, ax=ax)
ax.tick_params(axis='both', labelsize=16)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(f'Shared {TITLE_NAME}', fontsize=16)
plt.tight_layout()
plt.savefig(f"{PREFIX}_C_Overlap_Heatmap.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{PREFIX}_C_Overlap_Heatmap.pdf", bbox_inches='tight')
plt.close()

# ==========================================
# D. Gap Distribution
# ==========================================
print("Generating D: Gap distribution...")
gaps = []
for _, row in df_bin.iterrows():
    seq_stripped = "".join(row[display_states].astype(str)).strip('0')
    if '0' in seq_stripped:
        gaps.extend([len(b) for b in seq_stripped.split('1') if len(b) > 0])

fig, ax = plt.subplots(figsize=(7, 5))
if gaps:
    sns.countplot(x=gaps, palette="viridis", ax=ax)
    ax.set_xlabel('Gap length (states)', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
else:
    ax.text(0.5, 0.5, "No internal gaps detected.", ha='center', va='center', fontsize=16)
ax.tick_params(axis='both', labelsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PREFIX}_D_Gap_Distribution.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{PREFIX}_D_Gap_Distribution.pdf", bbox_inches='tight')
plt.close()

# ==========================================
# E. K-Means Clustering
# ==========================================
print("Generating E: K-Means clustering...")
n_clusters = min(4, len(df_bin))
if n_clusters > 1:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(df_bin[display_states])
    df_bin['Cluster'] = kmeans.labels_
    df_sorted = df_bin.sort_values(by=['Cluster', '6-nt', '18-nt'])
else:
    df_bin['Cluster'] = 0
    df_sorted = df_bin

df_sorted.to_csv(f"{PREFIX}_E_Clustered_Data.csv")
unique_clusters = sorted(df_sorted['Cluster'].unique())

for cluster_id in unique_clusters:
    cluster_df = df_sorted[df_sorted['Cluster'] == cluster_id]
    n_rows = len(cluster_df)

    fig_height = min(100, max(6, n_rows * 0.15))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    show_yticks = True if n_rows <= 150 else False
    sns.heatmap(cluster_df[display_states], cmap=["#eeeeee", "#313695"],
                cbar=False, yticklabels=show_yticks, ax=ax)

    if show_yticks:
        ax.tick_params(axis='y', labelsize=7, rotation=0)
    ax.tick_params(axis='x', labelsize=18)

    plt.tight_layout()
    plt.savefig(f"{PREFIX}_E_KMeans_Cluster_{cluster_id}.png", dpi=600, bbox_inches='tight')
    if n_rows < 1500:
        plt.savefig(f"{PREFIX}_E_KMeans_Cluster_{cluster_id}.pdf", bbox_inches='tight')
    plt.close()

# ==========================================
# F. Partner-Switching Dynamics
# ==========================================
print("Generating F: Partner-switching dynamics...")

domain_defs = {
    'RuvC': [(1, 59), (718, 764), (919, 1100)],
    'BH': [(60, 94)],
    'REC1': [(95, 176), (306, 495)],
    'REC2': [(177, 305)],
    'REC3': [(496, 717)],
    'L1': [(765, 780)],
    'HNH': [(781, 905)],
    'L2': [(906, 918)],
    'PI': [(1101, 1368)],
}

def get_hub_domain(res_str):
    res_str = str(res_str)
    if res_str.startswith('sgRNA'): return 'sgRNA'
    if res_str.startswith('TS'): return 'TS'
    if res_str.startswith('NTS'): return 'NTS'

    match = re.search(r'(\d+)', res_str)
    if not match: return 'Other'
    resid = int(match.group(1))
    for dom, segments in domain_defs.items():
        for start, end in segments:
            if start <= resid <= end:
                return dom
    return 'Other'

df_valid = df_master.loc[df_bin.index].copy()
df_valid['Pos_Residue'] = [idx.split('_')[0] for idx in df_valid.index]
df_valid['Target_Residue'] = [idx.split('_')[1] for idx in df_valid.index]

switchers = []
export_records = []

for anchor_col, partner_col in [('Pos_Residue', 'Target_Residue'), ('Target_Residue', 'Pos_Residue')]:
    for anchor_res, group in df_valid.groupby(anchor_col):
        if len(group) > 1:
            dominant_partners = group[display_states].idxmax(axis=0)
            if len(dominant_partners.unique()) > 1:
                dyn_score = group[display_states].var(axis=1).sum()
                if not any(anchor_res == s[0] for s in switchers):
                    switchers.append((anchor_res, dyn_score, group, partner_col))

                    anchor_domain = get_hub_domain(anchor_res)
                    all_partners = group[partner_col].tolist()
                    partner_domains = [get_hub_domain(p) for p in all_partners]
                    partner_info = ", ".join([f"{p}({d})" for p, d in zip(all_partners, partner_domains)])
                    dom_path = " -> ".join(dominant_partners.values)

                    export_records.append({
                        'Hub_Residue': anchor_res,
                        'Hub_Domain': anchor_domain,
                        'Dynamic_Score': round(dyn_score, 3),
                        'Partners_Count': len(all_partners),
                        'Partners_Detail': partner_info,
                        'Dominant_Partner_Path (6nt -> 18nt)': dom_path
                    })

switchers.sort(key=lambda x: x[1], reverse=True)

if export_records:
    df_export = pd.DataFrame(export_records).sort_values(by='Dynamic_Score', ascending=False)
    df_export.insert(0, 'Rank', range(1, len(df_export) + 1))
    csv_filename = f"{PREFIX}_All_Dynamic_Hubs_Ranked.csv"
    df_export.to_csv(csv_filename, index=False)
    print(f"  Exported {len(df_export)} dynamic hubs -> {csv_filename}")

if len(switchers) > 0:
    colors = sns.color_palette("Set1", 10)
    top_switchers = switchers[:6]
    n_cols = min(3, len(top_switchers))
    n_rows_plot = int(np.ceil(len(top_switchers) / 3))

    fig, axes = plt.subplots(n_rows_plot, n_cols, figsize=(5 * n_cols, 4 * n_rows_plot),
                             sharex=True, sharey=True)
    if len(top_switchers) > 1:
        axes = np.array(axes).flatten()
    else:
        axes = [axes]

    for i, (anchor_res, score, group, partner_col) in enumerate(top_switchers):
        ax = axes[i]
        targets = group[partner_col].tolist()
        occupancies = group[display_states].values

        for j, target in enumerate(targets):
            ax.plot(display_states, occupancies[j], marker='o', markersize=8,
                    linewidth=2.5, color=colors[j % len(colors)], label=target, alpha=0.85)

        ax.text(0.5, 1.05, f"Hub: {anchor_res}", transform=ax.transAxes,
                ha='center', va='bottom', fontsize=14)
        ax.text(0.5, 0.98, f"(Dyn score: {score:.2f})", transform=ax.transAxes,
                ha='center', va='top', fontsize=11, color='gray')
        ax.set_ylim(-0.05, 1.35)
        ax.legend(loc='upper center', ncol=2, fontsize=9, framealpha=0.9, columnspacing=1.0)
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i % n_cols == 0:
            ax.set_ylabel('Occupancy', fontsize=18)

    for j in range(len(top_switchers), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{PREFIX}_F_Partner_Switching_Top6.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{PREFIX}_F_Partner_Switching_Top6.pdf", bbox_inches='tight')
    plt.close()

print(f"\n  Pipeline complete.")
