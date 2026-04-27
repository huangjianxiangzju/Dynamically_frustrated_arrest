# scripts/Figure3/ — Domain-Level RMSF Dynamics

## Purpose
Generates the multi-panel RMSF domain dynamics figure showing how the median backbone flexibility of each SpCas9 domain evolves across the seven R-loop states.

## Scripts

| Script | Description |
|--------|-------------|
| `Fig_Mechanistic_Domain_Dynamics_median.py` | **Main figure script.** Reads the aggregated RMSF summary, computes per-domain medians, and produces a three-panel plot grouping domains as "Rigid Scaffold", "Sensors and Transducers", and "Catalytic Payload" (HNH). |
| `plot_rmsf_diff.py` | Generates a per-residue RMSF difference plot between consecutive state pairs. |
| `compare_delta_methods.py` | Compares different approaches to calculating ΔRMSF (mean vs median) for sensitivity analysis. |

## Inputs

| File | Description |
|------|-------------|
| `data/RMSF/rmsf_summary_all_states.csv` | Per-residue RMSF mean ± SD for all 7 states |

## Outputs

| File | Description |
|------|-------------|
| `figures/Figure3/Fig_Mechanistic_Domain_Dynamics_Clean.png` | Main panel figure (600 dpi) |

## Execution

```bash
cd data/RMSF
python ../../scripts/Figure3/Fig_Mechanistic_Domain_Dynamics_median.py
```

> **Note:** The script reads `rmsf_summary_all_states.csv` as a bare filename and must be run from the `data/RMSF/` directory.  
> Output `figures/Figure3/Fig_Mechanistic_Domain_Dynamics_Clean.png` is written relative to the working directory; the script saves to `../../figures/Figure3/` or the local directory depending on the configured output path.  
> `plot_rmsf_diff.py` and `compare_delta_methods.py` are utility/diagnostic scripts and follow the same path convention.
