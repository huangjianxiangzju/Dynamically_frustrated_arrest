"""
Figure 7C: Betweenness Centrality Evolution
All 8 key residues with thick/thin line approach.
Thick lines = dominant peaks (migration story)
Thin lines = supporting context
"""

import sys, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd

# --- FONT SETTINGS ---
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR     = '.'
PKL_NAME     = 'network_G_nierzwicki.pkl'
STATE_LABELS = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']
DISPLAY_STATES = [s.replace('nt', '-nt') for s in STATE_LABELS]
FIG_DIR      = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Primary residues (thick lines) — the migration story
# Y450 (early) -> L1/L2/R789 (late)
PRIMARY = {
    'Y450 (TS sensor)':        {'resid': 450,  'color': '#FF8C00', 'marker': '^',
                                 'lw': 2.8, 'ms': 10, 'alpha': 1.0, 'zorder': 10},
    'L1 (res 768)':            {'resid': 768,  'color': '#FFD700', 'marker': 'v',
                                 'lw': 2.8, 'ms': 10, 'alpha': 1.0, 'zorder': 9},
    'L2 (res 916)':            {'resid': 916,  'color': '#CCAA00', 'marker': '<',
                                 'lw': 2.8, 'ms': 10, 'alpha': 1.0, 'zorder': 9},
    'R789 (REC-HNH hinge)':    {'resid': 789,  'color': '#55A868', 'marker': 's',
                                 'lw': 2.8, 'ms': 10, 'alpha': 1.0, 'zorder': 8},
}

# Secondary residues (thin lines) — supporting context
SECONDARY = {
    r'K692 (REC3 $\alpha$37)':  {'resid': 692,  'color': '#937860', 'marker': '*',
                                  'lw': 1.3, 'ms': 8, 'alpha': 0.55, 'zorder': 4},
    'K1200 (NTS sensor)':       {'resid': 1200, 'color': '#D62728', 'marker': 'D',
                                  'lw': 1.3, 'ms': 8, 'alpha': 0.55, 'zorder': 4},
}

ALL_RESIDUES = {**PRIMARY, **SECONDARY}

# ==========================================
# 2. LOAD NETWORK
# ==========================================
def load_graph(label):
    path = None
    for p in [os.path.join(DATA_DIR, label, PKL_NAME),
              os.path.join(DATA_DIR, f'{label}_{PKL_NAME}')]:
        if os.path.isfile(p):
            path = p
            break

    if path is None:
        print(f'  [WARN] pkl file not found for {label}')
        return None, None

    with open(path, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, tuple):
        G = obj[0]
        residue_ids = obj[1] if len(obj) == 2 else obj[2]
    elif isinstance(obj, nx.Graph):
        G = obj
        residue_ids = list(range(1, G.number_of_nodes() + 1))
    else:
        return None, None

    return G, residue_ids

# ==========================================
# 3. COMPUTE CENTRALITY
# ==========================================
print("Computing betweenness centrality across all states...")

evolution_data = {name: [] for name in ALL_RESIDUES}

for label in STATE_LABELS:
    print(f"  Processing {label}...")
    G, res_ids = load_graph(label)

    if G is None:
        for name in ALL_RESIDUES:
            evolution_data[name].append(0)
        continue

    bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
    bc_resid = {res_ids[node]: val for node, val in bc.items()}

    for name, info in ALL_RESIDUES.items():
        evolution_data[name].append(bc_resid.get(info['resid'], 0.0))

# ==========================================
# 4. EXPORT CSV
# ==========================================
csv_rows = []
for name, info in ALL_RESIDUES.items():
    for i, state in enumerate(DISPLAY_STATES):
        csv_rows.append({
            'Residue': name,
            'Residue_ID': info['resid'],
            'State': state,
            'Betweenness_Centrality': evolution_data[name][i]
        })

df = pd.DataFrame(csv_rows)
df.to_csv("Betweenness_Centrality_Evolution.csv", index=False)
print("Saved: Betweenness_Centrality_Evolution.csv")

# ==========================================
# 5. PLOT
# ==========================================
print("Generating centrality evolution plot...")

fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(STATE_LABELS))

# Plot secondary (thin) lines first so primary lines draw on top
for name, info in SECONDARY.items():
    data = evolution_data[name]
    ax.plot(x, data, marker=info['marker'], color=info['color'],
            linewidth=info['lw'], markersize=info['ms'],
            alpha=info['alpha'], label=name, zorder=info['zorder'])

# Plot primary (thick) lines on top
for name, info in PRIMARY.items():
    data = evolution_data[name]
    ax.plot(x, data, marker=info['marker'], color=info['color'],
            linewidth=info['lw'], markersize=info['ms'],
            alpha=info['alpha'], label=name, zorder=info['zorder'])

ax.set_xticks(x)
ax.set_xticklabels(DISPLAY_STATES, fontsize=18)
ax.set_xlabel('R-loop state', fontsize=20)
ax.set_ylabel('Betweenness centrality', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# Legend: primary residues first, then secondary
handles, labels = ax.get_legend_handles_labels()
# Reorder: primary first (last 4 plotted), then secondary (first 4 plotted)
n_sec = len(SECONDARY)
n_pri = len(PRIMARY)
reordered_handles = handles[n_sec:] + handles[:n_sec]
reordered_labels = labels[n_sec:] + labels[:n_sec]

ax.legend(reordered_handles, reordered_labels,
          fontsize=11, loc='upper left', framealpha=0.9, ncol=2)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07C_centrality_evolution.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{FIG_DIR}/07C_centrality_evolution.pdf", bbox_inches='tight')
print(f"Saved: {FIG_DIR}/07C_centrality_evolution.png and .pdf")
plt.close()

# ==========================================
# 6. PRINT SUMMARY
# ==========================================
print("\n--- Betweenness Centrality Summary ---")
print("\n  PRIMARY (thick lines):")
for name, info in PRIMARY.items():
    vals = evolution_data[name]
    peak_state = DISPLAY_STATES[np.argmax(vals)]
    peak_val = max(vals)
    print(f"    {name}: peak at {peak_state} ({peak_val:.6f})")

print("\n  SECONDARY (thin lines):")
for name, info in SECONDARY.items():
    vals = evolution_data[name]
    peak_state = DISPLAY_STATES[np.argmax(vals)]
    peak_val = max(vals)
    print(f"    {name}: peak at {peak_state} ({peak_val:.6f})")
