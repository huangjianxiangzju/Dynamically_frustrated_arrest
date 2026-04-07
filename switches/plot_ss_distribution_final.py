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
INPUT_CSV = "Global_SS_RawData.csv"
OUTPUT_CSV = "All_Residues_SS_Shift.csv"

# ==========================================
# 2. LOAD AND CALCULATE DATA
# ==========================================
print(f"Loading {INPUT_CSV}...")

try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_CSV}.")
    exit()

state_cols = [col for col in df.columns if '_Ordered_Pct' in col]
df['Max_Shift_Pct'] = df[state_cols].max(axis=1) - df[state_cols].min(axis=1)

df_sorted = df.sort_values(by='Max_Shift_Pct', ascending=False)
df_sorted.to_csv(OUTPUT_CSV, index=False)
print(f" \u2713 Saved complete residue shift list to: {OUTPUT_CSV}")

# ==========================================
# 3. CALCULATE PROPORTIONS
# ==========================================
total = len(df)
stable_count = len(df[df['Max_Shift_Pct'] < 15])
flexible_count = len(df[(df['Max_Shift_Pct'] >= 15) & (df['Max_Shift_Pct'] < 40)])
switch_count = len(df[df['Max_Shift_Pct'] >= 40])

stable_pct = stable_count / total * 100
flexible_pct = flexible_count / total * 100
switch_pct = switch_count / total * 100

# ==========================================
# 4. PLOTTING WITH BROKEN Y-AXIS
# ==========================================
print("Generating distribution plot...")

shifts = df['Max_Shift_Pct'].values
bins = np.arange(0, 102, 2)

# Create two subplots sharing x-axis (top = tall bar, bottom = short bars)
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.3})

# Plot histogram on both axes
for ax in [ax_top, ax_bot]:
    n, bins_out, patches = ax.hist(shifts, bins=bins,
                                    color='#BDBDBD', edgecolor='black', linewidth=0.5)
    
    # Color bars by zone
    for i, patch in enumerate(patches):
        bin_left = patch.get_x()
        if bin_left < 14:
            patch.set_facecolor('#1976D2')
        elif bin_left < 40:
            patch.set_facecolor('#FFF176')
        else:
            patch.set_facecolor('#D32F2F')
    
    # Threshold lines
    ax.axvline(x=15, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=40, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=18)

# Set y-axis ranges for the break
ax_top.set_ylim(200, 850)
ax_bot.set_ylim(0, 50)

# Hide the spines at the break
ax_top.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)
ax_top.tick_params(axis='x', bottom=False, labelbottom=False)

# Add break marks (diagonal lines)
d = 0.012
kwargs = dict(transform=ax_top.transAxes, color='black', clip_on=False, linewidth=1.2)
ax_top.plot((-d, +d), (-d*2, +d*2), **kwargs)
#ax_top.plot((1 - d, 1 + d), (-d*2, +d*2), **kwargs)

kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1 - d*2, 1 + d*2), **kwargs)
#ax_bot.plot((1 - d, 1 + d), (1 - d*2, 1 + d*2), **kwargs)

# Zone annotations (place on top axis where there's space)
ax_top.text(8.5, 700, f'Stable core\n( {stable_pct:.1f}%)',
            ha='center', va='top', fontsize=16, color='#0D47A1')

ax_top.text(27.5, 700, f'Flexible\n( {flexible_pct:.1f}%)',
            ha='center', va='top', fontsize=16, color='#F57F17')

ax_top.text(70, 700, f'Allosteric switches\n( {switch_pct:.1f}%)',
            ha='center', va='top', fontsize=16, color='#B71C1C')

# Axis labels
ax_bot.set_xlabel('Maximum secondary structure shift (%)', fontsize=20)
fig.text(0.04, 0.5, 'Number of residues', fontsize=20, va='center', rotation='vertical')

ax_bot.set_xlim(0, 100)

plt.savefig("SS_Shift_Distribution.png", dpi=600, bbox_inches='tight')
plt.savefig("SS_Shift_Distribution.pdf", bbox_inches='tight')
print(f" \u2713 Saved SS_Shift_Distribution.png and .pdf")
plt.close()

# Summary
print("\n--- Summary Statistics ---")
print(f"Total Residues: {total}")
print(f"Stable core (<15% shift): {stable_count} residues ({stable_pct:.1f}%)")
print(f"Flexible (15-40% shift): {flexible_count} residues ({flexible_pct:.1f}%)")
print(f"Allosteric switches (>=40% shift): {switch_count} residues ({switch_pct:.1f}%)")
