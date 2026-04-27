import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- FONT SETTINGS ---
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("Dynamic_Variance_Hubs_Ranked.csv")

states_raw = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']
display_states = ['6-nt', '8-nt', '10-nt', '12-nt', '14-nt', '16-nt', '18-nt']
intensity_cols = [f'{s}_Intensity' for s in states_raw]

bar_colors = ['#FF0000', '#FF8C00', '#FFD700', '#00CC00', '#0099FF', '#0000FF', '#8B00FF']

# ==========================================
# 2. BOOTSTRAP RESAMPLING
# ==========================================
N_BOOT = 1000
SAMPLE_SIZE = 100
np.random.seed(42)

all_residue_vals = {}  # raw values for all 1368 residues
boot_means = {}        # 1000 bootstrapped means per state
true_means = {}        # true mean from all residues

for state_col, state_label in zip(intensity_cols, display_states):
    vals = df[state_col].values
    all_residue_vals[state_label] = vals
    true_means[state_label] = np.mean(vals)

    # Bootstrap: sample 100 residues with replacement, 1000 times
    boots = []
    for _ in range(N_BOOT):
        sample = np.random.choice(vals, size=SAMPLE_SIZE, replace=True)
        boots.append(np.mean(sample))
    boot_means[state_label] = np.array(boots)

# ==========================================
# 3. COMPUTE STATS FROM BOOTSTRAP
# ==========================================
means = np.array([true_means[s] for s in display_states])

# 95% CI from bootstrap distribution
ci_low = np.array([np.percentile(boot_means[s], 2.5) for s in display_states])
ci_high = np.array([np.percentile(boot_means[s], 97.5) for s in display_states])
err_low = means - ci_low
err_high = ci_high - means

# ==========================================
# 4. BOOTSTRAP SIGNIFICANCE TEST
# ==========================================
def bootstrap_pvalue(boots_a, boots_b):
    """Two-sided bootstrap test: fraction of times difference crosses zero."""
    diffs = boots_a - boots_b
    p = np.mean(diffs <= 0) if np.mean(diffs) > 0 else np.mean(diffs >= 0)
    return p * 2  # two-sided

def p_to_label(p):
    if p < 0.001: return 'p < 0.001'
    elif p < 0.01: return f'p = {p:.3f}'
    elif p < 0.05: return f'p = {p:.3f}'
    else: return f'ns (p = {p:.2f})'

def p_to_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

# All adjacent pairs
all_pairs = []
for i in range(len(display_states) - 1):
    p = bootstrap_pvalue(boot_means[display_states[i]], boot_means[display_states[i + 1]])
    all_pairs.append((i, i + 1, p))

# Key non-adjacent
for i, j in [(0, 1), (1, 5), (4, 5), (5, 6)]:
    if not any(p[0] == i and p[1] == j for p in all_pairs):
        p = bootstrap_pvalue(boot_means[display_states[i]], boot_means[display_states[j]])
        all_pairs.append((i, j, p))

# ==========================================
# 5. PLOT
# ==========================================
fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(display_states))
bar_width = 0.6

# Bars
bars = ax.bar(x, means, width=bar_width, color=bar_colors, alpha=0.75,
              edgecolor='black', linewidth=0.8, zorder=3)

# Asymmetric error bars (95% CI)
ax.errorbar(x, means, yerr=[err_low, err_high], fmt='none', ecolor='black',
            elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

# Dots: show a random subset of bootstrap means (e.g., 30) for visual texture
N_DOTS = 30
np.random.seed(123)
for i, s in enumerate(display_states):
    dot_idx = np.random.choice(N_BOOT, size=N_DOTS, replace=False)
    dots = boot_means[s][dot_idx]
    jitter = np.random.uniform(-0.15, 0.15, size=N_DOTS)
    ax.scatter(x[i] + jitter, dots, color='black', s=18, alpha=0.5,
              edgecolors='white', linewidths=0.4, zorder=5)

# ==========================================
# 6. SIGNIFICANCE BRACKETS
# ==========================================
# Show key comparisons: 6vs8 (surge), 14vs16 (collapse), 16vs18 (rebound)
show_pairs = [(0, 1), (4, 5), (5, 6)]

y_max = ci_high.max() + 20
bracket_height = means.max() * 0.02

bracket_positions = {
    (0, 1): y_max + 15 ,           # 6 vs 8-nt (lowest bracket)
    (4, 5): y_max - 30,      # 14 vs 16-nt (middle)
    (5, 6): y_max - 60,      # 16 vs 18-nt (highest)
}


for bracket_idx, (i, j, p_val) in enumerate(
        [(pi, pj, pp) for pi, pj, pp in all_pairs if (pi, pj) in show_pairs]):

    y_bar = bracket_positions[(i, j)]

    # Bracket lines
    ax.plot([x[i], x[i], x[j], x[j]],
            [y_bar - bracket_height, y_bar, y_bar, y_bar - bracket_height],
            color='black', linewidth=1.2, zorder=6)

    # Label
    label = p_to_label(p_val)
    ax.text((x[i] + x[j]) / 2, y_bar + bracket_height * 0.5, label,
            ha='center', va='bottom', fontsize=11, zorder=6)

# ==========================================
# 7. FORMATTING
# ==========================================
ax.set_xticks(x)
ax.set_xticklabels(display_states, fontsize=18)
ax.set_xlabel('R-loop state', fontsize=20)
ax.set_ylabel(r'Mean allosteric coupling ($\overline{\Sigma|C_{ij}|}$)', fontsize=18)
ax.tick_params(axis='both', labelsize=18)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Y-axis: start from a sensible baseline, extend for brackets
y_bottom = min(ci_low) * 0.85
y_top = y_max + len(show_pairs) * bracket_height * 5.5
ax.set_ylim(y_bottom, 1000)

plt.tight_layout()
plt.savefig("GCCM_Compact_Summary.png", dpi=600, bbox_inches='tight')
plt.savefig("GCCM_Compact_Summary.pdf", bbox_inches='tight')
print("Saved: GCCM_Compact_Summary.png and .pdf")
plt.close()

# ==========================================
# 8. PRINT STATS
# ==========================================
print("\n--- Global Coupling Summary (Bootstrap, N=1000, sample=100) ---")
print(f"{'State':<10} {'Mean':>10} {'95% CI low':>12} {'95% CI high':>12}")
for i, s in enumerate(display_states):
    print(f"{s:<10} {means[i]:>10.1f} {ci_low[i]:>12.1f} {ci_high[i]:>12.1f}")

print("\n--- Statistical Comparisons (Bootstrap) ---")
for i, j, p_val in all_pairs:
    print(f"{display_states[i]} vs {display_states[j]}: {p_to_label(p_val)} ({p_to_stars(p_val)})")
