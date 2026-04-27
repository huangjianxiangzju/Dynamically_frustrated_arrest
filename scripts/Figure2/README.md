# scripts/Figure2/ — Conformational Transition: PCA Porcupine Plot

## Purpose
Generates a porcupine plot showing the principal component 1 (PC1) displacement vectors across the SpCas9 backbone, visualising the dominant conformational transition along the R-loop progression trajectory.

## Scripts

| Script | Description |
|--------|-------------|
| `transition_pca_porcupine.py` | Loads a backbone PDB and a subsampled trajectory (every 50th frame), performs PCA with MDAnalysis, and outputs a porcupine-style vector plot of PC1 motions. |

## Inputs

| File | Description |
|------|-------------|
| `first_frame_backbone.pdb` | Reference backbone structure (first frame of the trajectory) |
| `every50th_frame.xtc` | Subsampled trajectory (every 50th frame from the concatenated multi-state simulation) |

## Outputs

Porcupine plot figure saved to the working directory.

## Requirements

```
MDAnalysis
numpy
matplotlib
```
