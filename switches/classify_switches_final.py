import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- FONT SETTINGS ---
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_CSV = "All_Residues_SS_Shift.csv"
SHIFT_THRESHOLD = 40.0

early_states = ['6nt_Ordered_Pct', '8nt_Ordered_Pct']
late_states = ['16nt_Ordered_Pct', '18nt_Ordered_Pct']

# ==========================================
# 2. LOAD AND CLASSIFY DATA
# ==========================================
print(f"Classifying Allosteric Switches (> {SHIFT_THRESHOLD}% shift)...\n")

try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_CSV}.")
    exit()

switches = df[df['Max_Shift_Pct'] >= SHIFT_THRESHOLD].copy()

if switches.empty:
    print("No residues found above the threshold!")
    exit()

switches['Early_Avg'] = switches[early_states].mean(axis=1)
switches['Late_Avg'] = switches[late_states].mean(axis=1)
switches['Delta_Structure'] = switches['Late_Avg'] - switches['Early_Avg']

conditions = [
    (switches['Delta_Structure'] >= 20.0),
    (switches['Delta_Structure'] <= -20.0)
]
choices = ['Folder (Disorder \u2192 Order)', 'Melter (Order \u2192 Disorder)']
switches['Trend'] = np.select(conditions, choices, default='Transient / Complex')

# ==========================================
# 3. PRINT GROUPED SUMMARY TABLE
# ==========================================
print(f"--- DETAILED SWITCH RESIDUE SUMMARY ---")
switches_sorted = switches.sort_values(by=['Domain', 'Trend', 'Residue_Number'])

current_domain = ""
for index, row in switches_sorted.iterrows():
    if row['Domain'] != current_domain:
        current_domain = row['Domain']
        print(f"\n>> Domain: {current_domain}")
    
    res = int(row['Residue_Number'])
    trend = row['Trend']
    max_shift = row['Max_Shift_Pct']
    early_val = row['Early_Avg']
    late_val = row['Late_Avg']
    
    print(f"   Res {res:<4} | {trend:<28} | Max Shift: {max_shift:>4.1f}% | (6nt: {early_val:>4.1f}% -> 18nt: {late_val:>4.1f}%)")

switches_sorted.to_csv("Classified_Switches.csv", index=False)
print(f"\n \u2713 Saved classified data to Classified_Switches.csv")

# ==========================================
# 4. SS SHIFT DISTRIBUTION (LOG Y-AXIS)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.arange(0, df['Max_Shift_Pct'].max() + 2, 2)
ax.hist(df['Max_Shift_Pct'], bins=bins, color='#888888', edgecolor='black', linewidth=0.5)

# Threshold lines
ax.axvline(x=15, color='#1976D2', linestyle='--', linewidth=1.5, label='Flexible (15%)')
ax.axvline(x=40, color='#D32F2F', linestyle='--', linewidth=1.5, label='Allosteric Switch (40%)')

# Shaded regions
ax.axvspan(0, 15, alpha=0.08, color='#888888', label='Stable Core (<15%)')
ax.axvspan(15, 40, alpha=0.08, color='#1976D2', label='Flexible (15\u201340%)')
ax.axvspan(40, df['Max_Shift_Pct'].max() + 2, alpha=0.08, color='#D32F2F', label='Allosteric Switches (>40%)')

ax.set_yscale('log')
ax.set_xlabel('Maximum secondary structure shift (%)', fontsize=20)
ax.set_ylabel('Number of residues', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("ss_shift_distribution.png", dpi=600, bbox_inches='tight')
plt.savefig("ss_shift_distribution.pdf", bbox_inches='tight')
print(" \u2713 Saved ss_shift_distribution.png and .pdf")
plt.close()

# ==========================================
# 5. DOMAIN SWITCH TRENDS (STACKED BAR)
# ==========================================
grouped = switches.groupby(['Domain', 'Trend']).size().unstack(fill_value=0)

domain_order = ["RuvC-I", "BH", "REC1-A", "REC2", "REC1-B", "REC3",
                "RuvC-II", "L1", "HNH", "L2", "RuvC-III", "PI"]
plot_order = [d for d in domain_order if d in grouped.index]
grouped = grouped.reindex(plot_order)

for t in ['Folder (Disorder \u2192 Order)', 'Melter (Order \u2192 Disorder)', 'Transient / Complex']:
    if t not in grouped.columns:
        grouped[t] = 0

colors = {
    'Folder (Disorder \u2192 Order)': '#009688',
    'Melter (Order \u2192 Disorder)': '#FF7043',
    'Transient / Complex': '#7E57C2'
}

fig, ax = plt.subplots(figsize=(10, 6))
grouped[['Folder (Disorder \u2192 Order)', 'Transient / Complex', 'Melter (Order \u2192 Disorder)']].plot(
    kind='bar', stacked=True,
    color=[colors['Folder (Disorder \u2192 Order)'],
           colors['Transient / Complex'],
           colors['Melter (Order \u2192 Disorder)']],
    ax=ax, edgecolor='black', width=0.7
)

ax.set_ylabel('Number of switch residues', fontsize=20)
ax.set_xlabel('', fontsize=20)  # domains are self-explanatory
ax.tick_params(axis='both', labelsize=18)

plt.xticks(rotation=45, ha='right')
ax.legend(fontsize=16, loc='upper left', framealpha=0.9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("Domain_Switch_Trends.png", dpi=600, bbox_inches='tight')
plt.savefig("Domain_Switch_Trends.pdf", bbox_inches='tight')
print(" \u2713 Saved Domain_Switch_Trends.png and .pdf")
plt.close()
