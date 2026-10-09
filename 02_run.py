"""Minimize -> NVT 100 ps (heating) -> NPT 1 ns (restraints released in stages) -> production.
Usage: python 02_run.py --seed 1 [--ns 20] [--scale 1]   (--scale 0.02 for a smoke test)"""
import argparse

from openmm import (CustomExternalForce, LangevinMiddleIntegrator, MonteCarloBarostat, Platform,
                    XmlSerializer, unit)
from openmm.app import DCDFile, Modeller, PDBFile, Simulation

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--ns", type=float, default=20)
ap.add_argument("--scale", type=float, default=1.0, help="multiplies every duration")
a = ap.parse_args()

DT, T = 0.002, 300  # ps, K
steps = lambda ps: max(1, int(ps * a.scale / DT))

pdb = PDBFile("system.pdb")
system = XmlSerializer.deserialize(open("system.xml").read())

# Positional restraints on protein heavy atoms (not the ligand: restraining it at k=1000 blew up its H); k is a global parameter so stages just change it
rest = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
rest.addGlobalParameter("k", 1000.0)  # kJ/mol/nm^2
for p in ("x0", "y0", "z0"):
    rest.addPerParticleParameter(p)
solute = [r for r in pdb.topology.residues() if r.name not in ("HOH", "NA", "CL")]
for r in solute:
    for at in r.atoms():
        if at.element.symbol != "H" and r.name != "UNK":
            rest.addParticle(at.index, pdb.positions[at.index].value_in_unit(unit.nanometer))
system.addForce(rest)

integ = LangevinMiddleIntegrator(50 * unit.kelvin, 1 / unit.picosecond, DT * unit.picosecond)
integ.setRandomNumberSeed(a.seed)
sim = Simulation(pdb.topology, system, integ, Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
sim.context.setPositions(pdb.positions)

sim.minimizeEnergy()
sim.context.setVelocitiesToTemperature(50 * unit.kelvin, a.seed)

# NVT: heat 50 -> 300 K in 10 chunks, k = 1000
for i in range(1, 11):
    integ.setTemperature((50 + 25 * i) * unit.kelvin)
    sim.step(steps(10))

# NPT: add barostat, then release restraints in stages
baro = MonteCarloBarostat(1 * unit.bar, T * unit.kelvin, 25)
baro.setRandomNumberSeed(a.seed)
system.addForce(baro)
sim.context.reinitialize(preserveState=True)
for k in (1000, 100, 10, 0):
    sim.context.setParameter("k", k)
    sim.step(steps(250))

# Production: save solute only (no water/ions), unwrapped, every 10 ps
keep = Modeller(pdb.topology, pdb.positions)
keep.delete([r for r in pdb.topology.residues() if r.name in ("HOH", "NA", "CL")])
idx = [at.index for r in solute for at in r.atoms()]
PDBFile.writeFile(keep.topology, keep.positions, open("solute.pdb", "w"), keepIds=True)
every = steps(10)
n_frames = int(a.ns * 1000 / 10 * a.scale)
with open(f"traj_seed{a.seed}.dcd", "wb") as f:
    dcd = DCDFile(f, keep.topology, 10 * unit.picosecond)
    for _ in range(n_frames):
        sim.step(every)
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        dcd.writeModel(pos[idx])
# ponytail: no checkpointing; a crash means rerunning that seed (~hours on a 2060 Super)
