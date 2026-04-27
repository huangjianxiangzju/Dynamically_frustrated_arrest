# raw_data/RMSF/

Per-replica GROMACS RMSF output files for all seven SpCas9 R-loop states.

## Structure

```
RMSF/
└── {state}/          # 6-nt, 8-nt, 10-nt, 12-nt, 14-nt, 16-nt, 18-nt
    ├── replica1/rmsf.xvg
    ├── replica2/rmsf.xvg
    ├── replica3/rmsf.xvg
    ├── replica4/rmsf.xvg
    └── replica5/rmsf.xvg
```

## File Format

Each `rmsf.xvg` is a plain-text GROMACS XVG file produced by `gmx rmsf -res`:
- Column 1: Residue number (1–1368)
- Column 2: RMSF in **nanometres** (nm)

Lines beginning with `#` or `@` are header/metadata.

## How These Files Are Used

The 35 files (7 states × 5 replicas) are averaged to produce:

`data/RMSF/rmsf_summary_all_states.csv`

which reports per-residue mean and standard deviation (in Å) for each state.
