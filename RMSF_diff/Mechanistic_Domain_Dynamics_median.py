import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. Configuration & Domain Definitions
# ==========================================
INPUT_CSV = "rmsf_summary_all_states.csv"

# Used for reading the CSV columns
states = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']

# Used for plotting on the X-axis
display_states = ['6-nt', '8-nt', '10-nt', '12-nt', '14-nt', '16-nt', '18-nt']

# Final domain composition provided
domain_defs = [
    ("RuvC-I", 1, 59, "#4C82C4"),
    ("BH", 60, 94, "#DA4997"),
    ("REC1-A", 95, 176, "#E1E1E1"),
    ("REC2", 177, 305, "#D0C5E3"),
    ("REC1-B", 306, 495, "#E1E1E1"),
    ("REC3", 496, 717, "#F4B98E"),
    ("RuvC-II", 718, 764, "#4C82C4"),
    ("L1", 765, 780, "#E6D84E"),
    ("HNH", 781, 905, "#FAEA55"),
    ("L2", 906, 918, "#E6D84E"),
    ("RuvC-III", 919, 1100, "#4C82C4"),
    ("PI", 1101, 1368, "#F28C8D")
]

# Create a dictionary for quick color lookup
color_map = {d[0]: d[3] for d in domain_defs}

# Define the panels with the simplified titles
groups = {
    'Rigid Scaffold': 
        ['RuvC-I', 'RuvC-II', 'BH', 'REC1-A', 'REC1-B', 'PI'],
        
    'Sensors and Transducers': 
        ['REC3', 'REC2', 'L1', 'L2', 'RuvC-III'],
        
    'Catalytic Payload': 
        ['HNH']
}

# ==========================================
# 2. Data Loading & Processing (Median)
# ==========================================
print(f"Loading data from {INPUT_CSV}...")
df_raw = pd.read_csv(INPUT_CSV)

# Initialize an empty DataFrame to store the domain medians
df_domains = pd.DataFrame(index=[d[0] for d in domain_defs], columns=states)

# Calculate median RMSF for each domain at each state
for state in states:
    mean_col = f"{state}_Mean" 
    
    for name, start, end, _ in domain_defs:
        # Filter raw data for the specific residue range
        domain_data = df_raw[(df_raw['Residue'] >= start) & (df_raw['Residue'] <= end)]
        
        # Calculate the MEDIAN RMSF for this domain in this state
        median_rmsf = domain_data[mean_col].median()
        df_domains.loc[name, state] = median_rmsf

# ==========================================
# 3. Plotting Setup
# ==========================================
# Set global font fallback to ensure Helvetica is prioritized
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']

sns.set_theme(style="whitegrid")
# Kept your increased figure width
fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

# Use distinct markers since some domains share the same color
markers = ['o', 's', '^', 'D', 'v', 'P']

for idx, (group_title, group_domains) in enumerate(groups.items()):
    ax = axes[idx]
    
    for c_idx, domain in enumerate(group_domains):
        # Convert nm to Angstroms by multiplying by 10
        y_values = df_domains.loc[domain].values * 10
        
        d_color = color_map[domain]
        d_marker = markers[c_idx % len(markers)]
        
        ax.plot(display_states, y_values, marker=d_marker, markersize=8, linewidth=2.5, 
                label=domain, color=d_color, alpha=0.9, markeredgecolor='black', markeredgewidth=0.5)
    
    # --- Add the 12nt vertical transition line (Annotation removed) ---
    #ax.axvline(x=kink_index, color='grey', linestyle='--', linewidth=2, alpha=0.7, zorder=0)

    # Panel formatting with updated titles and fonts
    ax.set_title(group_title, fontsize=18, fontweight='bold', pad=15)
    
    # Set exactly to 0-2.5 Angstroms (Equivalent to 0-0.25 nm)
    ax.set_ylim(0, 2.5)
    
    # Updated X-axis label with Helvetica, size 18
    ax.set_xlabel('R-loop Length', fontsize=18, fontweight='bold', fontname='Helvetica')
    
    if idx == 0:
        # Updated Y-axis label to reflect Angstroms
        ax.set_ylabel('Median RMSF (?)', fontsize=16)
    
    # Legend settings: inside the plot, no circling/frame
    legend = ax.legend(title='Domains', fontsize=16, loc='upper left', frameon=False)
    plt.setp(legend.get_title(), fontweight='bold')
    
    # Clean up borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5)

# Adjust layout
plt.tight_layout()
output_file = "Fig_Mechanistic_Domain_Dynamics_Clean.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"? Plotting complete! Image saved as {output_file}")