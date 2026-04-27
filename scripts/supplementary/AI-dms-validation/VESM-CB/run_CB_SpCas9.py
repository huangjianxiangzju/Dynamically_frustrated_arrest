"""
CB (Conformational Biasing) — SpCas9 R-loop state analysis
===========================================================
Scores all single mutants of SpCas9 against pairs of R-loop state structures.
Uses ProteinMPNN (default) + ESM-IF1 (cross-validation).

Hardware : Multi-GPU setup via --device argument
Input    : Two cleaned PDB files (single chain, protein only)
Output   : Per-variant CB scores + per-position bias summary

Usage (Run in separate terminal tabs for parallel processing):
    python run_CB_SpCas9.py --pdb1 7Z4E_8-nt_chainA_clean.pdb --pdb2 7Z4G_12-nt_chainA_clean.pdb --label 8nt_vs_12nt --device 0
    python run_CB_SpCas9.py --pdb1 7Z4G_12-nt_chainA_clean.pdb --pdb2 7Z4H_14-nt_chainA_clean.pdb --label 12nt_vs_14nt --device 1
    python run_CB_SpCas9.py --pdb1 7Z4H_14-nt_chainA_clean.pdb --pdb2 7Z4I_16-nt_chainA_clean.pdb --label 14nt_vs_16nt --device 2
"""
import torch
import jax
import jax.tree_util
jax.tree_map = jax.tree_util.tree_map  # 强行兼容旧版 ColabDesign

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import biotite.structure.io as bsio
from pathlib import Path

# Make sure you have cloned ConformationalBiasing and are running from inside it
sys.path.insert(0, str(Path(__file__).parent))

# ── MD Groups & Target Positions ─────────────────────────────────────────────

MD_GROUPS = {
    "Allosteric_Switch": [
        769, 771, 772, 773, 774, 775, 251, 254, 255, 256, 263, 271, 272, 284, 303, 304, 305,
        531, 538, 539, 624, 625, 712, 713, 714, 5, 23, 24, 25, 26, 35, 494, 495,
        1110, 1120, 1150, 1206, 1210, 1215, 1216, 1219, 1220, 1240,
        922, 957, 958, 960, 961, 981, 1002, 1003, 1004, 1008, 1017, 1018, 1019, 1020, 1021, 1022,
        1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 
        1039, 1040, 1041, 1049, 1050, 1051, 1057, 1058, 1059, 1087
    ],
    "GCCM_Hub": [
        170, 171, 168, 172, 175, 205, 204, 203, 183, 184, 297, 296, 181, 182, 298,
        185, 202, 295, 180, 300, 245, 186, 408
    ],
    "SaltBridge_Hub": [
        1200, 1114, 1118, 895, 905, 849, 782, 783, 850, 925, 919, 923, 762, 999,
        765, 63, 629, 661, 653, 274
    ],
    "Hydrophobic_Hub": [
        1036, 1039, 1004, 1022, 921, 1042, 1021, 1018, 1037, 1001, 917, 914,
        450, 451, 495, 492, 491, 256, 626, 846
    ],
    "Centrality_Hub": [
        450, 1200, 768, 916, 789, 794, 840, 692
    ]
}

# Flatten dictionary for target lookup
TARGET_POSITIONS = list(set([res for group in MD_GROUPS.values() for res in group]))
AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Global device variable, updated dynamically in main()
DEVICE = "cuda:0"

# ── Structure loading + alignment ─────────────────────────────────────────────

def load_structure(pdb_path, chain_id):
    import biotite.structure as struc
    atom_array = bsio.load_structure(pdb_path, model=1)
    mask = (atom_array.chain_id == chain_id) & (struc.filter_amino_acids(atom_array))
    atom_array = atom_array[mask]
    return atom_array

def extract_ca_coords_and_seq(atom_array):
    import biotite.structure as struc
    import biotite.sequence as bseq  # <--- 引入正确的 sequence 模块
    
    ca_mask  = atom_array.atom_name == "CA"
    ca_atoms = atom_array[ca_mask]
    coords   = ca_atoms.coord
    
    # 使用 bseq 来调用 convert_letter_3to1
    seq      = "".join([bseq.ProteinSequence.convert_letter_3to1(r) for r in ca_atoms.res_name])
    res_ids  = ca_atoms.res_id
    
    return coords, seq, res_ids


def align_sequences(seq1, res_ids1, seq2, res_ids2):
    id_set1 = {rid: i for i, rid in enumerate(res_ids1)}
    id_set2 = {rid: i for i, rid in enumerate(res_ids2)}
    shared  = sorted(set(id_set1.keys()) & set(id_set2.keys()))
    idx1    = [id_set1[r] for r in shared]
    idx2    = [id_set2[r] for r in shared]
    return idx1, idx2, shared

# ── ProteinMPNN scoring ───────────────────────────────────────────────────────

def score_with_proteinmpnn(pdb_path, mutant_sequences, chain_id="A"):
    from colabdesign.mpnn import mk_mpnn_model
    model = mk_mpnn_model() 
    model.prep_inputs(pdb_filename=pdb_path, chain=chain_id)
    
    scores = []
    for seq in mutant_sequences:
        s = model.score(seq=seq)
        
        # 兼容新版 ColabDesign (返回 dict) 和老版 (返回 float)
        if isinstance(s, dict):
            scores.append(s["score"])
        else:
            scores.append(s)
            
    return np.array(scores)


# ── ESM-IF1 scoring (cross-validation) ───────────────────────────────────────

def score_with_esmif1(pdb_path, mutant_sequences, chain_id="A"):
    import esm
    import esm.inverse_folding
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(DEVICE)
    structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    scores = []
    for seq in mutant_sequences:
        ll, _ = esm.inverse_folding.util.score_sequence(model, alphabet, coords, seq)
        scores.append(ll)
    return np.array(scores)

# ── Variant generation ────────────────────────────────────────────────────────

def generate_single_mutants(wt_seq, positions_0idx=None):
    variants = []
    scan_pos = positions_0idx if positions_0idx is not None else range(len(wt_seq))
    for pos in scan_pos:
        wt_aa = wt_seq[pos]
        for mut_aa in AAS:
            if mut_aa == wt_aa:
                continue
            mut_seq = wt_seq[:pos] + mut_aa + wt_seq[pos+1:]
            variants.append((pos + 1, wt_aa, mut_aa, mut_seq))
    return variants

# ── Main CB pipeline ──────────────────────────────────────────────────────────

def run_cb(pdb1, pdb2, chain, label, model_name, mode):

    print(f"\n{'='*60}")
    print(f"  CB Analysis: {label}")
    print(f"  State 1 : {pdb1}")
    print(f"  State 2 : {pdb2}")
    print(f"  Model   : {model_name}")
    print(f"  Mode    : {mode}")
    print(f"  Target  : {DEVICE}")
    #print(f"{='*60}\n")

    print("📂 Loading structures...")
    arr1 = load_structure(pdb1, chain)
    arr2 = load_structure(pdb2, chain)

    coords1, seq1, res_ids1 = extract_ca_coords_and_seq(arr1)
    coords2, seq2, res_ids2 = extract_ca_coords_and_seq(arr2)

    idx1, idx2, shared_res_ids = align_sequences(seq1, res_ids1, seq2, res_ids2)
    shared_seq = "".join(seq1[i] for i in idx1)
    
    if mode == "subset":
        shared_set = set(shared_res_ids)
        target_in_shared = []
        for p in TARGET_POSITIONS:
            if p in shared_set:
                local_idx = shared_res_ids.index(p)
                target_in_shared.append(local_idx)
        scan_positions = sorted(target_in_shared)
        print(f"  Scanning {len(scan_positions)} mapped MD switch residues")
    else:
        scan_positions = list(range(len(shared_seq)))
        print(f"  Full scan: {len(scan_positions)} positions")

    print("\n🧬 Generating single mutants...")
    variants = generate_single_mutants(shared_seq, positions_0idx=scan_positions)
    mutant_seqs = [v[3] for v in variants]
    print(f"  {len(variants)} variants generated")

    score_fn = score_with_proteinmpnn if model_name == "proteinmpnn" else score_with_esmif1

    print(f"\n🔬 Scoring against State 1 ({Path(pdb1).stem})...")
    scores1 = score_fn(pdb1, mutant_seqs, chain)

    print(f"🔬 Scoring against State 2 ({Path(pdb2).stem})...")
    scores2 = score_fn(pdb2, mutant_seqs, chain)

    cb_scores = scores2 - scores1

    def get_md_roles(pos):
        roles = []
        for group_name, res_list in MD_GROUPS.items():
            if pos in res_list:
                roles.append(group_name)
        return " | ".join(roles) if roles else "None"

    rows = []
    for (pos1idx, wt_aa, mut_aa, _), s1, s2, cb in zip(variants, scores1, scores2, cb_scores):
        rows.append({
            "position":        pos1idx,
            "wt":              wt_aa,
            "mut":             mut_aa,
            "variant":         f"{wt_aa}{pos1idx}{mut_aa}",
            "score_state1":    float(s1),
            "score_state2":    float(s2),
            "CB_bias":         float(cb),
            "MD_Roles":        get_md_roles(pos1idx),
            "is_MD_switch":    pos1idx in set(TARGET_POSITIONS),
            "model":           model_name,
            "comparison":      label,
        })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["position", "wt", "is_MD_switch", "MD_Roles"])
        .agg(
            mean_CB_bias  = ("CB_bias", "mean"),
            max_CB_bias   = ("CB_bias", "max"),
            min_CB_bias   = ("CB_bias", "min"),
            n_muts        = ("CB_bias", "count"),
        )
        .reset_index()
        .sort_values("mean_CB_bias", ascending=False)
    )

    mu  = summary["mean_CB_bias"].mean()
    sig = summary["mean_CB_bias"].std()
    summary["CB_bias_zscore"] = (summary["mean_CB_bias"] - mu) / sig

    out_dir = Path(f"CB_results_{label}_{model_name}")
    out_dir.mkdir(exist_ok=True)
    variants_file = out_dir / f"variants.csv"
    summary_file  = out_dir / f"position_summary.csv"
    df.to_csv(variants_file, index=False)
    summary.to_csv(summary_file, index=False)

    print(f"\n✓ Variant scores  → {variants_file}  ({len(df)} rows)")
    print(f"✓ Position summary → {summary_file}")

    print(f"\n── Top 15 State-2-biased MD positions (Positive CB_bias) ──")
    md_hits_forward = summary[summary["is_MD_switch"]].sort_values("mean_CB_bias", ascending=False)
    print(md_hits_forward[['position', 'wt', 'mean_CB_bias', 'MD_Roles']].head(15).to_string(index=False))

    print(f"\n── Top 15 State-1-biased MD positions (Negative CB_bias) ──")
    md_hits_reverse = summary[summary["is_MD_switch"]].sort_values("mean_CB_bias", ascending=True)
    print(md_hits_reverse[['position', 'wt', 'mean_CB_bias', 'MD_Roles']].head(15).to_string(index=False))

    return df, summary

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CB analysis for SpCas9 R-loop states")
    parser.add_argument("--pdb1",  required=True, help="PDB file for State 1")
    parser.add_argument("--pdb2",  required=True, help="PDB file for State 2")
    parser.add_argument("--label", required=True, help="Label for this comparison, e.g. 12nt_vs_18nt")
    parser.add_argument("--chain", default="A",   help="Chain ID (default: A)")
    parser.add_argument("--model", default="proteinmpnn", choices=["proteinmpnn", "esmif1"], help="Inverse folding model to use")
    parser.add_argument("--mode",  default="all", choices=["all", "subset"], help="'all' = full scan; 'subset' = TARGET_POSITIONS only")
    parser.add_argument("--device", type=int, default=0, choices=[0, 1, 2, 3], help="GPU device ID (0 to 3)")
    args = parser.parse_args()

    # CRITICAL: Force PyTorch and dependent libraries onto the target GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if torch.cuda.is_available():
        torch.cuda.set_device(0) # Because we masked via env var, it is always 0 to PyTorch
    
    global DEVICE
    DEVICE = "cuda:0" 

    df, summary = run_cb(
        pdb1       = args.pdb1,
        pdb2       = args.pdb2,
        chain      = args.chain,
        label      = args.label,
        model_name = args.model,
        mode       = args.mode,
    )
    print("\n🎉 Done.")

if __name__ == "__main__":
    main()
