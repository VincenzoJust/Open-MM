# BRD4 BD1 + (+)-JQ1 — OpenMM MD

First OpenMM project: 3 × 20 ns replicas of BRD4 BD1 bound to (+)-JQ1 (PDB 3MXF), with replica-comparison analysis (RMSD/RMSF, Asn140 contact, PCA + clustering).

## Run
Windows (conda-forge `openmmforcefields` needs AmberTools, which has no win-64 build, so it goes in with pip):
```bash
mamba env create -f environment.yml -p ~/.conda/envs/brd4-md   # add CONDA_PKGS_DIRS=~/.conda/pkgs if C:\Miniforge3 is read-only
mamba install -p ~/.conda/envs/brd4-md "cuda-version=12" libcufft   # OpenMM CUDA platform
~/.conda/envs/brd4-md/python -m pip install --no-deps openmmforcefields
# run python with <env>/Library/bin on PATH (or `mamba activate`), else the CUDA plugin fails to load (error 126)
python -c "import openmm as m; print([m.Platform.getPlatform(i).getName() for i in range(m.Platform.getNumPlatforms())])"  # needs CUDA
python 01_prepare.py                        # -> system.pdb, system.xml
python 02_run.py --seed 1 --scale 0.05      # smoke test (~45 s)
for s in 1 2 3; do python 02_run.py --seed $s; done   # ~2 h each on an RTX 2060 Super (~250 ns/day measured)
python 03_analysis.py                       # -> figures/
```
On Linux, `mamba env create -f environment.yml` plus a plain `mamba install openmmforcefields` is enough.

## Method
- **System**: 3MXF chain A, crystal waters/additives removed, PDBFixer (missing atoms, H at pH 7.4). Ligand bond orders from the JQ1 SMILES template; stereochemistry checked against (+)-JQ1.
- **Force field**: Amber14 protein, TIP3P, OpenFF Sage 2.1 ligand with NAGL (AM1-BCC-like) charges. (AmberTools/GAFF2 has no win-64 conda build, hence OpenFF.)
- **Box**: 1 nm padding, NaCl 0.15 M, neutralized. PME, 1 nm cutoff, HBonds constrained, 2 fs, LangevinMiddle 300 K, MonteCarloBarostat 1 bar.
- **Protocol**: minimize → 100 ps NVT heating 50→300 K (k = 1000 kJ/mol/nm²) → 1 ns NPT with restraints 1000/100/10/0 → 20 ns production per seed (seeds 1–3), frame every 10 ps, solute only.
- **Analysis**: CA-aligned RMSD (protein, ligand), RMSF, ligand–pocket contact occupancy, Asn140 ND2 to ligand N/O distance (heavy-atom proxy, no angle criterion), PCA on CA coordinates of pooled replicas, KMeans (k = 3).

## Limitations
- 20 ns × 3 is short: no convergence claim, no free energies, cluster populations are descriptive only.
- One force field combination (Amber14 + OpenFF Sage + NAGL charges); no sensitivity check.
- Disordered termini not modelled; protonation fixed at pH 7.4 by PDBFixer (no pKa calc); conserved crystal waters (e.g. near Tyr97) removed, so water-mediated contacts must re-form.
- H-bond metric is a distance cutoff, not distance + angle.
- No apo control.
- No checkpointing; trajectories are not versioned (`*.dcd` ignored).
