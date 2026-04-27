# scripts/Figure7/ — CB vs VESM Discordance Analysis

## Purpose
Integrates ProteinMPNN conditional-probability (CB) scores with VESM-3B evolutionary constraint scores to identify residues showing discordance between structural context (CB) and evolutionary pressure (VESM). These discordant residues are interpreted as mechanistically important allosteric sites.

## Scripts

| Script | Description |
|--------|-------------|
| `plot_CB_VESM_discordance.py` | Generates a two-panel figure: (A) CB driving-force vs VESM evolutionary-constraint scatter plot coloured by MD hub category and quadrant-annotated at medians; (B) quadrant composition bar plot (total / MD-annotated / super-hub). |

## Inputs

| File | Description |
|------|-------------|
| `data/AI-validation-[CB,VESM]/CB_results_*_proteinmpnn/position_summary.csv` | Per-position CB score summaries for each state transition |
| `data/AI-validation-[CB,VESM]/SpCas9_VESM3B_full_position_summary.csv` | VESM-3B per-position mean LLR scores |
| `data/Interaction_Network/SB_All_Dynamic_Hubs_Ranked.csv` | MD-derived dynamic hub annotations |
| `data/AI-validation-[CB,VESM]/DMS_reference/spencer-zhang-data.csv` | DMS reference data from Spencer & Zhang 2017 (see reference below) |

## Outputs

Discordance scatter plot and quadrant bar chart, saved to the `scripts/Figure7/` working directory (600 dpi PNG and PDF).  
The quadrant assignment table is written to `data/AI-validation-[CB,VESM]/CB_VESM_discordance/cb_vesm_quadrant_table.csv`, which is used downstream by `data/AI-validation-[CB,VESM]/cb_vesm_dms_triple.py`.

## Execution

```bash
cd data/AI-validation-[CB,VESM]
python ../../scripts/Figure7/plot_CB_VESM_discordance.py
```

> **Note:** The script uses bare relative paths (`CB_results_*_proteinmpnn/`, `SpCas9_VESM3B_full_position_summary.csv`) and must therefore be executed from the `data/AI-validation-[CB,VESM]/` directory, not from `scripts/Figure7/`.  
> `CB_VESM_Discordance.png/.pdf` are written to the working directory.  
> `cb_vesm_quadrant_table.csv` is also written to the working directory; move it to `CB_VESM_discordance/` afterwards (it is required by `data/AI-validation-[CB,VESM]/cb_vesm_dms_triple.py`).

## References

- Spencer, J.M., Zhang, X. Deep mutational scanning of S. pyogenes Cas9 reveals important functional domains. *Sci Rep* 7, 16836 (2017). https://doi.org/10.1038/s41598-017-17081-y

