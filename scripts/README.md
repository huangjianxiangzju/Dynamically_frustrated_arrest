# scripts/

This folder contains all analysis and figure-generation Python scripts, organised by figure number.

## Sub-folder Overview

| Folder | Description |
|--------|-------------|
| `Figure2/` | PCA porcupine plot of the dominant conformational transition vector across R-loop states |
| `Figure3/` | Domain-level RMSF dynamics — median backbone flexibility per domain per state |
| `Figure4/` | Global secondary structure heatmap and classification of allosteric switch residues |
| `Figure5/` | Allosteric interaction network pipeline (salt bridges, hydrophobic contacts) |
| `Figure6/` | GCCM network analysis — community structure, centrality evolution, path impedance |
| `Figure7/` | CB vs VESM discordance analysis for identifying allosteric hotspots |
| `supplementary/` | Supplementary figure scripts and the AI validation pipeline (VESM-CB scoring) |

Each sub-folder has its own `README.md` with detailed input/output descriptions and execution instructions.

## General Notes

- All scripts are written in **Python 3.10+**.
- Scripts expect processed data files to be present in `data/` and raw data in `raw_data/` (relative to the repository root). Adjust path variables at the top of each script if running from a different working directory.
- Figure output files are saved to `figures/` (see `figures/README.md`).
- The `Figure6/build_network.py` script is the only one that requires the original MD trajectory files.
