"""
Plot optimal path impedance for 3 groups across 7 R-loop states.
Reads from pre-computed Allosteric_Pathways_Summary.txt files
using the 00_parse_pathways.py parser.

Directory structure expected:
    6nt/Allosteric_Pathways_Summary.txt
    8nt/Allosteric_Pathways_Summary.txt
    ...
    18nt/Allosteric_Pathways_Summary.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from importlib import import_module

# --- FONT SETTINGS ---
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# 1. LOAD DATA USING EXISTING PARSER
# ==========================================
# Import the parser module (must be in the same directory or PYTHONPATH)
try:
    parser = import_module('00_parse_pathways')
except ModuleNotFoundError:
    # Fallback: define parser inline
    import os

    STATE_LABELS = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']
    DATA_DIR = ''
    FILENAME = 'Allosteric_Pathways_Summary.txt'

    def parse_pathway_file(filepath):
        results = {}
        current_pair = None
        with open(filepath, 'r') as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('Pair:'):
                    tokens = line.split()
                    src = int(tokens[2])
                    snk = int(tokens[5])
                    current_pair = (src, snk)
                    results[current_pair] = {'impedances': [], 'paths': []}
                elif line.startswith('Path') and current_pair is not None:
                    parts = line.split('|')
                    impedance = float(parts[1].split(':')[1].strip())
                    route_str = parts[2].split(':')[1].strip()
                    route = [int(x.strip()) for x in route_str.split('->')]
                    results[current_pair]['impedances'].append(impedance)
                    results[current_pair]['paths'].append(route)
        return results

    def load_all_states():
        all_results = []
        for label in STATE_LABELS:
            fpath = os.path.join(DATA_DIR, label, FILENAME)
            if not os.path.isfile(fpath):
                print(f"[WARNING] File not found: {fpath}")
                all_results.append({})
            else:
                all_results.append(parse_pathway_file(fpath))
                print(f"[OK] Loaded {fpath}")
        return all_results

    class parser:
        STATE_LABELS = STATE_LABELS
        load_all_states = staticmethod(load_all_states)

print("Loading pathway data...")
all_results = parser.load_all_states()

states_raw = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']
display_states = [s.replace('nt', '-nt') for s in states_raw]

# ==========================================
# 2. DEFINE 3 GROUPS
# ==========================================
groups = {
    r'REC3 $\alpha$37 $\rightarrow$ HNH': {
        'pairs': [(692, 840), (694, 840), (695, 840), (698, 840)],
        'color': '#2166AC',
        'marker': 'o',
    },
    r'HNH hinge': {
        'pairs': [(789, 841), (789, 858), (794, 841), (794, 858)],
        'color': '#00AA00',
        'marker': 's',
    },
    r'Y450 $\rightarrow$ catalytic sites': {
        'pairs': [(450, 840), (450, 10)],
        'color': '#FF8C00',
        'marker': '^',
    },
    r'K1200 $\rightarrow$ catalytic sites': {
        'pairs': [(1200, 840), (1200, 10)],
        'color': '#D62728',
        'marker': 'D',
    },
}

# ==========================================
# 3. EXTRACT OPTIMAL IMPEDANCE PER PAIR PER STATE
# ==========================================
print("\nExtracting optimal impedance per group...")

results = {}

for group_name, group_info in groups.items():
    group_means = []
    group_sems = []
    group_all = []

    for state_idx, state_data in enumerate(all_results):
        pair_optimal = []

        for source, sink in group_info['pairs']:
            pair_key = (source, sink)

            if pair_key in state_data and state_data[pair_key]['impedances']:
                # Optimal = minimum impedance (shortest path)
                optimal = min(state_data[pair_key]['impedances'])
                pair_optimal.append(optimal)
            else:
                print(f"  Warning: pair ({source}, {sink}) not found in {states_raw[state_idx]}")

        if pair_optimal:
            group_means.append(np.mean(pair_optimal))
            group_sems.append(np.std(pair_optimal, ddof=1) / np.sqrt(len(pair_optimal))
                              if len(pair_optimal) > 1 else 0)
        else:
            group_means.append(np.nan)
            group_sems.append(0)

        group_all.append(pair_optimal)

    results[group_name] = {
        'means': group_means,
        'sems': group_sems,
        'all_pairs': group_all,
    }

# ==========================================
# 4. EXPORT CSV
# ==========================================
csv_rows = []
for group_name, group_info in groups.items():
    for state_idx, state_label in enumerate(display_states):
        pair_vals = results[group_name]['all_pairs'][state_idx]
        for pair_idx, (source, sink) in enumerate(group_info['pairs']):
            imp = pair_vals[pair_idx] if pair_idx < len(pair_vals) else np.nan
            csv_rows.append({
                'Group': group_name,
                'Source': source,
                'Sink': sink,
                'State': state_label,
                'Optimal_Impedance': imp
            })

df_out = pd.DataFrame(csv_rows)
df_out.to_csv("Path_Impedance_All_Groups.csv", index=False)
print("\nSaved: Path_Impedance_All_Groups.csv")

# ==========================================
# 5. PLOT
# ==========================================
print("Generating impedance plot...")

fig, ax = plt.subplots(figsize=(9, 6))

for group_name, group_info in groups.items():
    means = results[group_name]['means']
    sems = results[group_name]['sems']

    ax.errorbar(display_states, means, yerr=sems,
                fmt=f'-{group_info["marker"]}',
                color=group_info['color'],
                ecolor=group_info['color'],
                elinewidth=1.2, capsize=4, capthick=1.2,
                markersize=9, linewidth=2.2, alpha=0.9,
                label=group_name, zorder=5)

ax.set_xlabel('R-loop state', fontsize=20)
ax.set_ylabel(r'Optimal path impedance ($-\ln|C_{ij}|$)', fontsize=18)
ax.tick_params(axis='both', labelsize=18)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# Note: lower impedance = better communication
ax.legend(fontsize=12, loc='best', framealpha=0.9)

plt.tight_layout()
plt.savefig("Path_Impedance_Groups.png", dpi=600, bbox_inches='tight')
plt.savefig("Path_Impedance_Groups.pdf", bbox_inches='tight')
print("Saved: Path_Impedance_Groups.png and .pdf")
plt.close()

# ==========================================
# 6. PRINT SUMMARY
# ==========================================
print("\n--- Optimal Path Impedance Summary ---")
for group_name in groups:
    print(f"\n{group_name}:")
    for i, s in enumerate(display_states):
        m = results[group_name]['means'][i]
        sem = results[group_name]['sems'][i]
        print(f"  {s}: {m:.4f} +/- {sem:.4f}")
