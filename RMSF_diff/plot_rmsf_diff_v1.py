import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
import sys

# --- FONT SETTINGS ---
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# --- 1. ARGUMENT PARSING ---
if len(sys.argv) < 3:
    print("Usage: python plot_rmsf_diff.py <State1> <State2>")
    print("Example: python plot_rmsf_diff.py 8nt 6nt")
    print("Note: This calculates (State1 - State2)")
    sys.exit(1)

state1_name = sys.argv[1]  # e.g., "8nt"
state2_name = sys.argv[2]  # e.g., "6nt"

# Display names with hyphen
def display_name(s):
    return s.replace("nt", "-nt")

state1_disp = display_name(state1_name)
state2_disp = display_name(state2_name)

# Number of replicas
N_REPLICAS = 5
replicas = [f'replica{i}' for i in range(1, N_REPLICAS + 1)]
filename = 'rmsf.xvg'

# --- 2. DOMAIN DEFINITIONS ---
def rgb(r, g, b):
    return (r/255.0, g/255.0, b/255.0)

domains = [
    ("RuvC",        1,    59,   rgb(51, 122, 204)),    # RuvC-I
    ("Bridge Helix",60,   94,   rgb(255, 51, 204)),    # BH
    ("REC1",        95,   176,  rgb(230, 230, 229)),   # REC1-A
    ("REC2",        177,  305,  rgb(218, 215, 235)),   # REC2
    ("REC1",        306,  495,  rgb(230, 230, 229)),   # REC1-B
    ("REC3",        496,  717,  rgb(252, 209, 166)),   # REC3
    ("RuvC",        718,  764,  rgb(51, 122, 204)),    # RuvC-II
    ("L1",          765,  780,  rgb(255, 255, 122)),   # L1 - yellow like HNH
    ("HNH",         781,  905,  rgb(255, 255, 122)),   # HNH
    ("L2",          906,  918,  rgb(255, 255, 122)),   # L2 - yellow like HNH
    ("RuvC",        919,  1100, rgb(51, 122, 204)),    # RuvC-III
    ("PAM Int.",    1101, 1368, rgb(255, 153, 153)),   # PI
]

# --- 3. DATA LOADING FUNCTION ---
def load_state_data(state_folder):
    data_stack = []
    residues = []
    
    print(f"Loading data for state: {state_folder}...")
    
    for rep in replicas:
        file_path = os.path.join(state_folder, rep, filename)
        
        if not os.path.exists(file_path):
            print(f"  Error: {file_path} not found. Filling with NaNs.")
            data_stack.append(None)
            continue
            
        try:
            raw = np.loadtxt(file_path, comments=['@', '#'])
            
            if len(residues) == 0:
                residues = raw[:, 0]
            
            if len(raw[:, 1]) != len(residues):
                print(f"  Warning: Length mismatch in {rep}. Skipping.")
                continue
                
            data_stack.append(raw[:, 1])
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")

    return np.array(data_stack), residues

# --- 4. LOAD AND CALCULATE ---
stack1, resid1 = load_state_data(state1_name)
stack2, resid2 = load_state_data(state2_name)

if len(stack1) == 0 or len(stack2) == 0:
    print("Error: Could not load data. Exiting.")
    sys.exit(1)

if not np.array_equal(resid1, resid2):
    print("Error: Residue indices do not match between the two states!")
    sys.exit(1)

residues = resid1

mean1 = np.mean(stack1, axis=0)
sem1  = np.std(stack1, axis=0) / np.sqrt(N_REPLICAS)

mean2 = np.mean(stack2, axis=0)
sem2  = np.std(stack2, axis=0) / np.sqrt(N_REPLICAS)

diff_mean = mean1 - mean2
diff_error = np.sqrt(sem1**2 + sem2**2)

# --- 5. PLOTTING ---
fig, ax = plt.subplots(figsize=(14, 6))

# A. Zero line
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.8, zorder=5)

# B. Difference line
ax.plot(residues, diff_mean, color='black', linewidth=1.5,
        label=f'\u0394RMSF ({state1_disp} \u2212 {state2_disp})', zorder=10)

# C. Error envelope
ax.fill_between(residues, diff_mean - diff_error, diff_mean + diff_error,
                color='black', alpha=0.3, label='Propagated error (SEM)', zorder=9)

# D. Domain coloring
added_labels = {}
for name, start, end, color in domains:
    ax.axvspan(start, end, facecolor=color, alpha=0.6, edgecolor=None, zorder=1)
    
    if name not in added_labels:
        added_labels[name] = mpatches.Patch(color=color, label=name)

# --- 6. FORMATTING ---
ax.set_xlim(1, 1368)
y_max = np.max(np.abs(diff_mean) + diff_error) * 1.1
ax.set_ylim(-y_max, y_max)

ax.set_xlabel('Residue Index', fontsize=20)
ax.set_ylabel(f'\u0394RMSF (nm)\n[{state1_disp} \u2212 {state2_disp}]', fontsize=20)
ax.set_title(f'RMSF Difference: {state1_disp} vs {state2_disp}', fontsize=20)

ax.tick_params(axis='both', labelsize=18)

# Legend
legend_patches = [added_labels[k] for k in added_labels]
legend_patches.append(mpatches.Patch(color='black', label=f'\u0394RMSF \u00B1 SEM'))

ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=7, frameon=False, fontsize=12)

plt.tight_layout()
out_name = f'rmsf_diff_{state1_name}_minus_{state2_name}.png'
plt.savefig(out_name, dpi=600, bbox_inches='tight')
plt.savefig(out_name.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Plot saved as {out_name} and .pdf")
plt.close()
