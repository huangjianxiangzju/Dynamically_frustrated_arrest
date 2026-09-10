# Changelog

## Version 2.1 — 2026-09-10

- Corrected the integrated CB–VESM–MD ranking script to merge one MD annotation record per residue. Historical role-label synonyms could produce multiple records at a position; metadata now derive from the five category flags in `full_superset.csv`, with one-to-one merge checks.
- Added the corrected 1,368-position integrated ranking table, its full-precision companion and an explicit list of 83 three-method consensus residues. Percentile ranks use average ties over all 1,368 positions; the combined score remains the mean of the three percentiles. Evidence counts are calculated before display rounding.
- Preserved the MD category sizes (84, 54, 46, 90 and 89), the 311-residue union and the 52 multi-category key residues. The 90 hydrophobic-category residues are distinct from the 83 three-method consensus residues; that category count is not corrected to 83.
- Added the approved corrected integrated-ranking figure as Supplementary Figure S14, with PNG, TIFF and vector PDF versions. Added all eight main and 18 supplementary manuscript-numbered PNGs, a figure index, captions and the supplementary figure/note numbering map.
- Made the integrated-ranking script independent of the current working directory and added an independent numerical validation script.
- Aligned documentation with the supplementary renumbering following placement of Materials and Methods before Results. Main figure and supplementary table numbers are unchanged.

The raw simulation outputs, trajectories and input CB/VESM scores are unchanged. This release supersedes the integrated-ranking script in Zenodo record 22278694; it does not change the earlier Figure 7 correction or rerun the MD or neural scoring calculations.

## Version 2.0 — 2026-09-02

This release aligns the deposited data, scripts, figures, and documentation with the revised manuscript under review at *Nucleic Acids Research*.

- Replaced the stale exploratory Louvain community table with the weighted Girvan–Newman assignments used for the reported results. The deposited partitions use GCCM-derived impedance as the edge distance and retain the maximum-modularity partition for each state (Q = 0.79–0.82).
- Added the weighted Girvan–Newman recomputation script and regenerated community-membership, community-size, and modularity outputs.
- Added the complete 1,368-residue weighted betweenness-centrality matrix and Kneedle audit. The verified centrality elbow is rank 89, with peak centrality 0.0823457336; the earlier value of 98 was a preliminary caption error.
- Documented all five independently filtered MD evidence categories: 84 structural switches, 54 GCCM-variance hubs, 46 salt-bridge partner-switching residues, 90 hydrophobic partner-switching residues, and 89 centrality hubs. Their union contains 311 residues.
- Added `MD_key_residues_multi_evidence.csv`, containing the final 52 MD key residues supported by at least two independent categories, together with a script that recreates the table from `full_superset.csv`.
- Replaced "evolutionary constraint" with "VESM substitution intolerance" in code, tables, figures, and documentation. VESM uses an ESM2-3B backbone loaded with VESM distilled weights; its sign-inverted mean LLR is not direct sequence conservation.
- Corrected the Figure 7 aggregation to enforce one record per SpCas9 position. Historical synonyms for the structural-switch category had duplicated some rows in the plotting workflow. The corrected output contains exactly 1,368 positions and reproduces the manuscript's super-hub counts: Q1 = 13, Q2 = 15, Q3 = 13, and Q4 = 11.

The underlying MD trajectories and raw simulation outputs are unchanged from version 1.
