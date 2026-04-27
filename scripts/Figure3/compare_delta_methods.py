import os
import sys
import numpy as np

# --- 1. COMMAND LINE ARGUMENTS ---
if len(sys.argv) < 6:
    print("Usage: python compare_delta_methods.py <state_A> <state_B> <template_pdb> <out_02_pdb> <out_stat_pdb>")
    print("Example: python compare_delta_methods.py 6nt 8nt 8nt.pdb 6nt_8nt_0.2cut.pdb 6nt_8nt_statcut.pdb")
    sys.exit(1)

STATE_A = sys.argv[1]
STATE_B = sys.argv[2]
TEMPLATE_PDB = sys.argv[3]
OUT_PDB_02 = sys.argv[4]
OUT_PDB_STAT = sys.argv[5]

if not os.path.exists(TEMPLATE_PDB):
    print(f"ERROR: Template PDB '{TEMPLATE_PDB}' not found.")
    sys.exit(1)

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
    (1101, 1368, "PI")
]

num_replicas = 5
num_protein_residues = 1368
SCALE_FACTOR = 10.0  # nm to Angstroms

# --- 3. HELPER FUNCTIONS ---
def read_xvg_rmsf(filepath):
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')): continue
            parts = line.split()
            if len(parts) >= 2: values.append(float(parts[1]))
    return np.array(values)

def get_replica_rmsf_arrays(state_name):
    replica_arrays = []
    for rep in range(1, num_replicas + 1):
        filepath = os.path.join(state_name, f"replica{rep}", "rmsf.xvg")
        if os.path.exists(filepath):
            data = read_xvg_rmsf(filepath)
            if len(data) >= num_protein_residues:
                replica_arrays.append(data[:num_protein_residues] * SCALE_FACTOR)
    return replica_arrays

# --- 4. MAIN EXECUTION ---
print(f"\n--- Method Comparison: {STATE_A} -> {STATE_B} ---")

reps_A = get_replica_rmsf_arrays(STATE_A)
reps_B = get_replica_rmsf_arrays(STATE_B)

if not reps_A or not reps_B:
    print("Error: Could not load data for one or both states.")
    sys.exit(1)

# Dictionaries to store the mapped values for the PDBs
mapping_02 = {}
mapping_stat = {}

print(f"{'Domain':<10} | {'Δ Median':<10} | {'Noise (±)':<10} | {'0.2 Å Cutoff Logic':<22} | {'Statistical Logic'}")
print("-" * 80)

for start, end, name in domains:
    # Calculate medians per replica, then their mean and std dev
    medians_A = [np.median(rep[start-1 : end]) for rep in reps_A]
    medians_B = [np.median(rep[start-1 : end]) for rep in reps_B]
    
    mean_med_A = np.mean(medians_A)
    std_A = np.std(medians_A)
    
    mean_med_B = np.mean(medians_B)
    std_B = np.std(medians_B)
    
    delta = mean_med_B - mean_med_A
    combined_noise = np.sqrt(std_A**2 + std_B**2)
    
    # Logic 1: 0.2 Cutoff
    if abs(delta) > 0.2:
        conc_02 = "Rigidified" if delta < 0 else "Loosened"
        mapping_02[name] = delta
    else:
        conc_02 = "Stable (<= 0.2)"
        mapping_02[name] = 0.0  # Force to white
        
    # Logic 2: Statistical Noise
    if abs(delta) > combined_noise:
        conc_stat = "Rigidified" if delta < 0 else "Loosened"
        mapping_stat[name] = delta
    else:
        conc_stat = "Stable (<= Noise)"
        mapping_stat[name] = 0.0  # Force to white
        
    # Flag disagreements with an asterisk
    flag = "*" if conc_02.split()[0] != conc_stat.split()[0] else ""
    
    print(f"{name:<10} | {delta:>7.3f} Å   | {combined_noise:>7.3f} Å   | {conc_02:<22} | {conc_stat} {flag}")

print("\n(* indicates a disagreement between the two methods)")

# --- 5. PROJECT TO PDBS ---
def write_pdb(template, output, mapping_dict):
    with open(template, 'r') as f_in, open(output, 'w') as f_out:
        for line in f_in:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    res_num = int(line[22:26].strip())
                    if 1 <= res_num <= num_protein_residues:
                        mapped_val = 0.0
                        for start, end, name in domains:
                            if start <= res_num <= end:
                                mapped_val = mapping_dict[name]
                                break
                        new_line = line[:60] + f"{mapped_val:6.2f}" + line[66:]
                        f_out.write(new_line)
                    else:
                        f_out.write(line)
                except ValueError:
                    f_out.write(line)
            else:
                f_out.write(line)

write_pdb(TEMPLATE_PDB, OUT_PDB_02, mapping_02)
write_pdb(TEMPLATE_PDB, OUT_PDB_STAT, mapping_stat)

print(f"\nCreated: {OUT_PDB_02}")
print(f"Created: {OUT_PDB_STAT}")
