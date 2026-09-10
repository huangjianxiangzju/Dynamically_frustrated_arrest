# Integrated CB–VESM–MD ranking (Supplementary Figure S14)

## Inputs and calculation

Inputs are the six adjacent-state `CB_results_*_proteinmpnn/position_summary.csv` files, `SpCas9_VESM3B_full_position_summary.csv` and `full_superset.csv`, all in this directory.

For each of 1,368 SpCas9 positions:

1. CB driving force is the arithmetic mean of `−CB_bias_zscore` over the six state transitions.
2. VESM substitution intolerance is `−mean_LLR`. The retained column name `vesm_constraint` is a historical identifier; it does not imply experimentally measured sequence conservation.
3. MD support (`md_n`) is the number of positive category flags among Switch, GCCM, SB_hub, Hydro_hub and BC; positions absent from the 311-residue MD union have zero support. The five category sizes are 84, 54, 46, 90 and 89; 52 residues have at least two categories.
4. Each metric is converted to an ascending percentile rank: average tied rank divided by 1,368, multiplied by 100. The integrated score is the arithmetic mean of these three percentiles.
5. `evidence_count` is the number of **unrounded** percentiles strictly greater than 50. Counts for 0/1/2/3 are 267/606/412/83. Display rounding must not be used to reconstruct a strict threshold.

MD annotation roles are assembled directly from one row per residue in `full_superset.csv`; historical role-label synonyms are not grouping keys. Input observations and the scoring definitions are unchanged.

## Outputs

- `integrated_rank_table.csv`: display-rounded table matching the approved figure.
- `integrated_rank_table_full_precision.csv`: unrounded means, percentiles and integrated scores for numerical reuse.
- `three_method_consensus_residues.csv`: the 83 positions above the median in all three metrics, with residue identity and domain.

## Reproduction

From the repository root:

```bash
python scripts/supplementary/AI-dms-validation/plot_integrated_rank.py
python scripts/supplementary/AI-dms-validation/validate_integrated_rank.py
```

The plotting script reads repository-relative inputs regardless of the working directory. It writes tables here and artwork to `figures/submission/Supplementary_figures/`. To avoid overwriting supplied outputs when checking reproduction, pass `--output /path/to/check_figures --table-output /path/to/check_tables`. Required packages: numpy, pandas, scipy, matplotlib, Pillow and adjustText.

The deposited PNG is the author-approved artwork. Label positions may vary slightly when rerendered with different fonts or adjustText versions; numerical outputs must agree.
