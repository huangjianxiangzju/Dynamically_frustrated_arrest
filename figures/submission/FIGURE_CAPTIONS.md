# Supplementary figure captions

## Figure S1

Convergence of HNH domain backbone RMSD for the 12-nt R-loop state. HNH coordinates in both states were grafted from the 14-nt undocked conformation (PDB 7Z4H) prior to simulation. Raw RMSD values are shown in transparent shading and a running average (5 ns window) as a solid line. All five replicas converge to stable RMSD values of ~1.0 Å within the first 100 ns, with no systematic drift or replica-dependent divergence over the 1,000 ns production runs, confirming that the grafted HNH domain relaxes into a well-defined conformational basin independent of the starting model. 

## Figure S2

Supplementary dynamical network analysis. (A) Residue-level global allosteric coupling profiles (Σ|Cij|) for the three stages of R-loop maturation. (B) Modularity Q values from weighted Girvan–Newman edge-betweenness decomposition across all seven states. GCCM-derived impedance weights were used, and the maximum-modularity partition along each decomposition dendrogram was retained (Q = 0.79–0.82). Louvain optimization was not used for the reported communities. (C) Community-size distribution showing redistribution of residues across states. The number of communities increases from 14 at 6-nt to 18 at 16-nt before consolidating to 11 at 18-nt. (D) Normalized mutual information matrix quantifying pairwise similarity of community structure between states.

## Figure S3

Topological deformation of target DNA and sgRNA strand invasion during progressive R-loop extension. 3D spatial conformational evolution of the nucleic acid backbone from 6-nt to 18-nt. Yellow, sgRNA; blue, target strand (TS); red, non-target strand (NTS). Progressive kinking of the distal DNA duplex and extrusion of the NTS into a flexible single-stranded loop at 18-nt are illustrated across the seven R-loop states.

## Figure S4

Global dynamics and the flexibility checkpoint. Differential RMSF (ΔRMSF) profiles plotted as a function of residue number for each state transition from 6-nt to 18-nt. Positive values indicate increased flexibility; negative values indicate decreased flexibility. Domain boundaries are indicated along the x-axis.

## Figure S5

Spatial distribution of structural switch residues in the RuvC-III domain, mapped onto the representative 18-nt R-loop structure (6-nt structure shown as transparent overlay for reference). Structural switch residues partition into a spatially segregated pattern: a central Folder cluster (dark green spheres) flanked by two Melter clusters (purple spheres) on either side, with more isolated Transient residues (pink spheres). The zoom-in inset highlights the residues 1024–1028 in the Folder cluster, illustrating their close contact with the target-strand DNA (red) in the 18-nt R-loop state.

## Figure S6

Per-residue mechanical force on RuvC-III increases sharply from 16-nt to 18-nt and drives opposing secondary structure transitions in switch residues. (A) Mean resultant force magnitude (|F|) averaged over all 182 RuvC-III residues (919–1100) across seven R-loop states; shading denotes ±SD across residues. (B) Mean |F| computed separately for the 26 Folders and 15 Melters within the 41 RuvC-III switch residues; both subsets experience the same abrupt force increase at the 16-nt to 18-nt transition. (C) Mean ordered secondary structure fraction (helix + strand) for Folders and Melters across R-loop states. Gray shading highlights the 16-nt to 18-nt transition.

## Figure S7

Dynamics of hydrophobic interaction residues during progressive R-loop formation in the SpCas9-sgRNA complex. (A) Sankey diagram of hydrophobic contact residue state transitions from 6-nt to 18-nt. (B) Bar and line plots of persisting, disappearing, and appearing hydrophobic contact residues and net flux at each transition. (C) Inter-domain hydrophobic contact diagrams for each R-loop state, with edge thickness proportional to the number of contacts. (D) Representative hydrophobic contact residues.

## Figure S8

Stage-based classification and occupancy heatmaps of dynamic non-covalent interactions. Heatmaps display the continuous occupancy of dynamic salt bridges (left) and hydrophobic contacts (right) across the seven R-loop progression states (6-nt to 18-nt). Interactions were filtered to exclude constitutive contacts and grouped into seven mutually exclusive categories based on their presence across three major structural phases: Stage 1 (unlocking, 6-nt to 10-nt), Stage 2 (pre-organization, 12-nt to 14-nt), and Stage 3 (pre-catalytic gating, 16-nt to 18-nt). The color map represents the fractional occupancy of each interaction over the trajectory, ranging from 0.0 (light yellow, absent) to 1.0 (dark blue, fully formed). Colored vertical bars on the left of each heatmap denote the categorical assignment of the interaction subsets, with the total number of residue pairs (n) indicated for each class. The staggered clustering illustrates the systematic disassembly of early-stage networks and the distinct emergence of late-stage gating contacts. For stage assignment, each state was grouped by the transition it completes: 6-, 8-, and 10-nt to Stage 1; 12- and 14-nt to Stage 2; 16- and 18-nt to Stage 3.

## Figure S9

Intersection of dynamic interaction networks across R-loop progression stages. Venn diagrams illustrate the temporal overlap of dynamic salt bridges (left) and hydrophobic contacts (right) across the three primary phases of R-loop maturation: Stage 1 (6-nt to 10-nt), Stage 2 (12-nt to 14-nt), and Stage 3 (16-nt to 18-nt). Numerical values represent the absolute count of distinct residue-residue interactions within each intersection. Stage 2 hosts relatively few exclusive interactions compared with Stages 1 and 3, consistent with its role as a transitional phase between two structurally distinct energetic endpoints rather than a stable energetic minimum.

## Figure S10

Constitutive salt bridge and hydrophobic interactions maintained throughout R-loop progression. Bar charts show the number of rigid residues participating in constitutive salt bridges (left) and constitutive hydrophobic contacts (right) within each SpCas9 domain and the bound nucleic acids, pooled across all seven R-loop extension states (6-nt to 18-nt). Interactions are classified as constitutive if their occupancy exceeds the mode-specific threshold (≥50% for salt bridges; ≥40% for hydrophobic contacts) in every R-loop state examined. 

## Figure S11

Y450–sgRNA (residue 1385) hydrophobic contact occupancy as a function of R-loop length. Contact occupancy values are plotted for 6-nt to 18-nt R-loop states. Maximum occupancy is observed at 8-nt (0.977), with a sharp decline at 10-nt (0.765) and relatively stable occupancy (~0.85–0.91) from 12-nt to 18-nt, consistent with the transition from base-stacking gatekeeper at the seed region to persistent sgRNA anchor across the remainder of R-loop maturation.

## Figure S12

Multi-metric summary of the three-stage allosteric trajectory during R-loop maturation. (Top) Three independent metrics, each normalized to [0, 1], tracked across the seven R-loop states: overall RMSF (red circles, mean backbone flexibility across all 1,368 protein residues), structured residues (blue squares, number of residues with secondary structure occupancy exceeding 40%), and mean GCCM coupling (purple triangles, global allosteric coupling intensity from Figure 6a). Shaded backgrounds denote the three stages. (Bottom) Representative structural snapshots from each stage boundary. At 8-nt (left), Y450 senses the initial R-loop wedge and the REC lobe loosens, coinciding with peak global coupling. At 12-nt (center), the sharply kinked distal DNA duplex drives REC2 drift and overall flexibility reaches its maximum. At 16-nt (right), the system achieves maximal rigidification as HNH is primed for the 18-nt checkpoint, where dynamical frustration prevents productive catalytic activation.

## Figure S13

Elbow analysis of MD-derived allosteric hub rankings. (A) All 1,368 SpCas9 residues ranked by cross-state variance of global coupling intensity (Σ|Cij|). The Kneedle algorithm identifies a cutoff at k = 54 (variance score = 27,094.5; red dashed line). (B) All 1,368 residues ranked by peak weighted betweenness centrality across the seven R-loop states. The Kneedle cutoff is k = 89 (peak centrality = 0.0823457336; red dashed line). Red dots mark the elbow points. These are category-specific cutoffs; the final stringent MD key-residue set contains 52 residues supported by at least two independent MD categories.

## Figure S14

Integrated percentile-rank analysis combining CB, VESM substitution intolerance, and MD evidence. (A) Scatter plot of CB percentile rank (PCB) versus VESM substitution-intolerance percentile rank (PVESM) for all 1,368 SpCas9 positions. Dot size reflects MD-category evidence count; dot color indicates the number of methods above the 50th percentile (evidence count 0–3). Dashed lines mark the 50th percentile on each axis. The top 15 residues by combined score are labeled. (B) Stacked bar chart of percentile contributions (PCB, orange; PVESM, blue; PMD, red) for the top 30 residues ranked by the arithmetic mean of the three percentile ranks. VESM −meanLLR is model-inferred substitution intolerance, not direct evolutionary conservation.

## Figure S15

Position-level cross-validation of the CB × VESM substitution-intolerance quadrant partition against deep mutational scanning data. (A) Distribution of DMS tolerance (median log2FC under positive selection) across the four quadrants. Boxplots show the median, interquartile range, and 1.5× IQR whiskers; individual positions are overlaid. Q4 exhibits higher DMS tolerance than Q1 (one-sided Mann-Whitney p = 1.3 × 10−10) and Q2 (p = 1.4 × 10−8); Kruskal-Wallis H = 57.9, p = 1.7 × 10−12, n = 1,163 positions (Q1 = 284; Q2 = 311; Q3 = 272; Q4 = 296). (B) DMS tolerance versus CB driving force (−ZCB,bias), colored by VESM substitution intolerance (−meanLLR). Point size scales with sampled substitutions (1–7). Circled points are triple-convergent candidates. Dashed lines mark median CB force and neutral DMS tolerance. Spearman ρ(DMS, CB) = +0.013, p = 0.66. (C) Scatter plot of DMS tolerance versus VESM substitution intolerance (−meanLLR), colored by CB driving force. The negative correlation (Spearman ρ = −0.254, p = 1.5 × 10−18) indicates that positions assigned higher model-inferred substitution intolerance by VESM are less tolerant in the DMS assay; this axis is not a direct measurement of evolutionary conservation. Triple-convergent candidates are circled and labeled as in (B).

## Figure S16

Substitution-level cross-validation of CB, VESM masked-marginal scores, and experimental DMS fitness. (A) VESM LLR versus DMS log2FC for 2,346 matched substitutions (Spearman ρ = +0.267, p = 1.4 × 10−39; AUROC = 0.646). Open circles mark published fidelity-variant substitutions. (B) CB driving force versus DMS log2FC. CB is uncorrelated with fitness both unconditionally (ρ = −0.019, p = 0.37) and after conditioning on VESM LLR (partial ρ = −0.019, p = 0.37), consistent with CB reporting state-dependent conformational selectivity rather than substitution fitness. With n = 2,346, the minimum detectable correlation at 80% power is |ρ| = 0.058. (C) ROC curves for substitution tolerance: VESM LLR alone (AUROC = 0.646), CB alone (0.505), and cross-validated VESM + CB logistic regression (0.640). Adding CB does not improve fitness classification because CB was not designed as a fitness predictor; this null result does not by itself demonstrate statistical independence, and CB's contribution is its state-directional structural criterion.

## Figure S17

CB directionality analysis across six stepwise R-loop transitions. (A) Heatmap of conformational bias direction for the 138 flipper residues that reverse bias direction at least once during R-loop progression. Residues (x-axis, sorted by hub overlap count) are shown across all six transitions (y-axis). Blue, pre-transition bias; red, post-transition bias; white, neutral or not observed. (B, C) Fraction of pre-transition (source-state-favoring) and post-transition (target-state-favoring) residues belonging to each allosteric hub category across the six transitions. Bar colors indicate hub category as shown in the legend.

## Figure S18

Kinetics of reporter activation for all constructs at 24, 48, and 72 hours. Constructs are grouped by their 72-hour behaviour. (A) Constructs with increased activity relative to wild-type. (B) Constructs at or below wild-type. (C) The two constructs forming the eSpCas9-like low-ratio group. The wild-type trajectory, shown as a dashed black line in every panel, is included for reference. Each trace is the mean of three replicates.
