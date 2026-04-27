# scripts/Figure5/ — Allosteric Interaction Network Pipeline

## Purpose
Analyses the state-dependent competitive interaction networks (salt bridges and hydrophobic contacts) between SpCas9 domains, tracking how inter-domain communication changes as the R-loop elongates.

## Scripts

| Script | Description |
|--------|-------------|
| `Allosteric_Interaction_Pipeline_final.py` | **Main pipeline.** Takes an interaction type as a command-line argument. Reads per-state raw interaction CSVs, applies occupancy thresholds, identifies constitutive/persisting/appearing/disappearing interactions, classifies dynamic hubs, generates Sankey diagrams, bar/line plots of net flux, and saves processed tables. |
| `plot_network_series_updated.py` | Generates the macroscopic inter-domain interaction network diagram showing aggregate connection strengths across states. |

## Inputs

| File | Description |
|------|-------------|
| `raw_data/Interaction_Network/{state}/competitive_salt_bridges.csv` | Per-state salt-bridge occupancy data |
| `raw_data/Interaction_Network/{state}/competitive_hydrophobic.csv` | Per-state hydrophobic contact occupancy data |

## Outputs

| File | Description |
|------|-------------|
| `data/Interaction_Network/SB_Constitutive_Interactions.csv` | Salt bridges present in all states |
| `data/Interaction_Network/SB_All_Dynamic_Hubs_Ranked.csv` | Ranked dynamic hub residues (salt bridges) |
| `data/Interaction_Network/SB_Clustered_Data.csv` | Binary occupancy matrix for salt bridges with cluster assignment |
| `data/Interaction_Network/SB_Stage_Classified_Data.csv` | Salt bridges classified by active R-loop stage |
| `data/Interaction_Network/SB_Stage_Summary.csv` | Stage-category summary statistics for salt bridges |
| `data/Interaction_Network/Hydro_Constitutive_Interactions.csv` | Hydrophobic contacts present in all states |
| `data/Interaction_Network/Hydro_All_Dynamic_Hubs_Ranked.csv` | Ranked dynamic hub residues (hydrophobic) |
| `data/Interaction_Network/Hydro_Clustered_Data.csv` | Binary occupancy matrix for hydrophobic contacts with cluster assignment |
| `data/Interaction_Network/Hydro_Stage_Classified_Data.csv` | Hydrophobic contacts classified by active R-loop stage |
| `data/Interaction_Network/Hydro_Stage_Summary.csv` | Stage-category summary statistics for hydrophobic contacts |
| `figures/Figure5/SB_Macroscopic_Domain_Network_Final.png` | Inter-domain network diagram |
| `figures/Figure5/SB_B_Net_Flux.png` | Net flux plot |

## Execution

```bash
cd scripts/Figure5
python Allosteric_Interaction_Pipeline_final.py saltbridge
python Allosteric_Interaction_Pipeline_final.py hydro
python plot_network_series_updated.py
```

**Occupancy thresholds (defaults):**
- Salt bridges: 0.50
- Hydrophobic contacts: 0.40
