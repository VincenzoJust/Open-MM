"""Download 3MXF (BRD4 BD1 + (+)-JQ1), fix, parametrize ligand, solvate -> system.pdb / system.xml."""
import urllib.request
from io import StringIO

from openff.toolkit import Molecule
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
from openmm import XmlSerializer, unit
from openmm.app import PDBFile, Modeller, HBonds, PME
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem

PDB_ID, LIG = "3MXF", "JQ1"
JQ1_SMILES = "CC1=C(SC2=C1C(=N[C@H](C3=NN=C(N32)C)CC(=O)OC(C)(C)C)C4=CC=C(C=C4)Cl)C"

raw = urllib.request.urlopen(f"https://files.rcsb.org/download/{PDB_ID}.pdb").read().decode()
lig_lines = [l for l in raw.splitlines() if l.startswith("HETATM") and l[17:20] == LIG and l[21] == "A"]
assert lig_lines, f"{LIG} not found in chain A of {PDB_ID}; check rcsb.org"

# Protein: chain A only, no crystal waters/additives, pH 7.4
fixer = PDBFixer(pdbfile=StringIO(raw))
fixer.removeChains([c.index for c in fixer.topology.chains() if c.id != "A"])
fixer.removeHeterogens(keepWater=False)
fixer.findMissingResidues()
fixer.missingResidues = {}  # ponytail: skip modelling disordered termini
fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues()
fixer.findMissingAtoms(); fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.4)

# Ligand: bond orders from SMILES template, hydrogens, NAGL charges
rd = Chem.MolFromPDBBlock("\n".join(lig_lines), removeHs=False)
rd = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(JQ1_SMILES), rd)
rd = Chem.AddHs(rd, addCoords=True)
Chem.AssignStereochemistryFrom3D(rd)
for at in rd.GetAtoms():
    at.SetMonomerInfo(None)  # else added Hs end up in separate residues
lig = Molecule.from_rdkit(rd)
assert lig.to_smiles() == Molecule.from_smiles(JQ1_SMILES).to_smiles(), "ligand stereo mismatch: not (+)-JQ1"
lig.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.1.pt", toolkit_registry=NAGLToolkitWrapper())

gen = SystemGenerator(
    forcefields=["amber14-all.xml", "amber14/tip3p.xml"],
    small_molecule_forcefield="openff-2.1.0",
    molecules=[lig],
    forcefield_kwargs=dict(constraints=HBonds, rigidWater=True, removeCMMotion=True),
    periodic_forcefield_kwargs=dict(nonbondedMethod=PME, nonbondedCutoff=1 * unit.nanometer),
)

m = Modeller(fixer.topology, fixer.positions)
m.add(lig.to_topology().to_openmm(), lig.conformers[0].to_openmm())
m.addSolvent(gen.forcefield, model="tip3p", padding=1 * unit.nanometer,
             ionicStrength=0.15 * unit.molar, neutralize=True)

system = gen.create_system(m.topology, molecules=[lig])
PDBFile.writeFile(m.topology, m.positions, open("system.pdb", "w"), keepIds=True)
open("system.xml", "w").write(XmlSerializer.serialize(system))
print("atoms:", m.topology.getNumAtoms(), "| residues:", sorted({r.name for r in m.topology.residues()} - {"HOH"})[-5:])
