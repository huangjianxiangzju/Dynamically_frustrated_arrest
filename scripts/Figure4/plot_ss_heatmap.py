import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# --- FONT SETTINGS ---
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# --- 1. LOAD DATA ---
df = pd.read_csv("Global_SS_RawData.csv")

states_raw = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']
states_display = ['6-nt', '8-nt', '10-nt', '12-nt', '14-nt', '16-nt', '18-nt']

NUM_RESIDUES = 1368

# Build matrix from CSV columns
global_matrix = np.zeros((NUM_RESIDUES, len(states_raw)))
for c_idx, state in enumerate(states_raw):
    col_name = state + '_Ordered_Pct'
    global_matrix[:, c_idx] = df[col_name].values[:NUM_RESIDUES]

# --- 2. DOMAIN DEFINITIONS ---
domains = [
    (1,    59,   "RuvC-I"),
    (60,   94,   "BH"),
    (95,   176,  "REC1-A"),
    (177,  305,  "REC2"),
    (306,  495,  "REC1-B"),
    (496,  717,  "REC3"),
    (718,  764,  "RuvC-II"),
    (765,  780,  "L1"),
    (781,  905,  "HNH"),
    (906,  918,  "L2"),
    (919,  1100, "RuvC-III"),
    (1101, 1368, "PI"),
]

# --- 3. PLOT ---
fig, ax = plt.subplots(figsize=(8, 18))

# Color scheme: deep navy to white to deep crimson
from matplotlib.colors import LinearSegmentedColormap
colors_cmap = ['#0D1B2A', '#1B2838', '#1B4965', '#5FA8D3',
               '#FFFFFF',
               '#F4845F', '#E63946', '#9D0208', '#370617']
cmap = LinearSegmentedColormap.from_list('navy_crimson', colors_cmap, N=256)

im = ax.imshow(global_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100,
               interpolation='nearest')

# --- 4. X-AXIS (states at bottom) ---
ax.set_xticks(np.arange(len(states_raw)))
ax.set_xticklabels(states_display, fontsize=18)
ax.xaxis.tick_bottom()
ax.tick_params(axis='x', length=5, width=1.2, pad=8)

# --- 5. Y-AXIS (domain labels) ---
y_ticks = []
y_labels = []

for i, (start, end, name) in enumerate(domains):
    plot_start = start - 1
    plot_end = end - 1

    # Domain separator lines
    if i < len(domains) - 1:
        ax.axhline(y=plot_end + 0.5, color='black', linewidth=1.0, linestyle='-')

    midpoint = plot_start + (plot_end - plot_start) / 2
    y_ticks.append(midpoint)
    y_labels.append(name)

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=18)
ax.tick_params(axis='y', length=4, width=1.0, pad=6)

# No ylabel (reader already knows the domains)

# --- 6. COLORBAR ---
cbar = fig.colorbar(im, ax=ax, pad=0.03, aspect=40, shrink=0.75)
cbar.set_label('Ordered state occupancy (%)', fontsize=20, rotation=270, labelpad=28)
cbar.ax.tick_params(labelsize=16, length=4, width=1.0)

# No title

# --- 7. SAVE ---
plt.tight_layout()
plt.savefig("Global_SS_Heatmap.png", dpi=600, bbox_inches='tight')
plt.savefig("Global_SS_Heatmap.pdf", bbox_inches='tight')
print("Saved: Global_SS_Heatmap.png and .pdf")
plt.close()
