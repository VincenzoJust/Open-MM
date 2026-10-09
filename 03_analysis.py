"""RMSD / RMSF / pocket contacts / PCA + clustering across replicas -> figures/"""
import glob

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis.analysis import align, distances, rms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

SEEDS = sorted(int(f[9:-4]) for f in glob.glob("traj_seed*.dcd"))
LIG = "not protein"  # solute.pdb only has protein + ligand
N140_ND2 = "resid 140 and name ND2"
POCKET_CUTOFF = 4.0   # A, residue within this of ligand = contact
HBOND_CUTOFF = 3.5    # A, heavy-atom N/O proxy (no angle criterion)

ref = mda.Universe("solute.pdb", f"traj_seed{SEEDS[0]}.dcd")
ref.trajectory[0]
assert ref.select_atoms(N140_ND2).n_atoms == 1, "Asn140 ND2 not found: check residue numbering"
assert ref.select_atoms(LIG).n_atoms > 0, "no ligand found"

rmsd, rmsf, nh, occ, ca_xyz = {}, {}, {}, {}, {}
for s in SEEDS:
    u = mda.Universe("solute.pdb", f"traj_seed{s}.dcd")
    t_ns = np.arange(len(u.trajectory)) * 0.01
    R = rms.RMSD(u, ref, select="name CA", groupselections=[f"({LIG}) and not name H*"]).run()
    rmsd[s] = (t_ns, R.results.rmsd[:, 2], R.results.rmsd[:, 3])  # CA, ligand

    align.AlignTraj(u, ref, select="name CA", in_memory=True).run()
    ca = u.select_atoms("name CA")
    rmsf[s] = (ca.resids, rms.RMSF(ca).run().results.rmsf)
    ca_xyz[s] = np.array([ca.positions.ravel() for _ in u.trajectory])

    lig, prot = u.select_atoms(f"({LIG}) and not name H*"), u.select_atoms("protein")
    n140, no = u.select_atoms(N140_ND2), lig.select_atoms("element N O")
    nh[s] = np.array([distances.distance_array(n140.positions, no.positions).min() for _ in u.trajectory])
    resids = np.unique(prot.resids)
    contacts = np.zeros(len(resids))
    for _ in u.trajectory:
        near = prot.select_atoms(f"around {POCKET_CUTOFF} group lig", lig=lig).resids
        contacts += np.isin(resids, near)
    occ[s] = (resids, contacts / len(u.trajectory))
    print(f"seed {s}: ligand RMSD {rmsd[s][2].mean():.2f} A | Asn140 H-bond occupancy {(nh[s] < HBOND_CUTOFF).mean():.2f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for s in SEEDS:
    ax[0].plot(rmsd[s][0], rmsd[s][1], label=f"seed {s}")
    ax[1].plot(rmsd[s][0], rmsd[s][2])
    ax[2].plot(*rmsf[s])
for a_, t in zip(ax, ["Protein CA RMSD (A)", "Ligand RMSD, protein-aligned (A)", "CA RMSF (A)"]):
    a_.set_title(t)
ax[0].set_xlabel("ns"); ax[1].set_xlabel("ns"); ax[2].set_xlabel("residue"); ax[0].legend()
fig.tight_layout(); fig.savefig("figures/rmsd_rmsf.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s in SEEDS:
    ax[0].plot(rmsd[s][0], nh[s], label=f"seed {s}")
    ax[1].plot(*occ[s])
ax[0].axhline(HBOND_CUTOFF, ls="--", c="k"); ax[0].set_title("Asn140 ND2 - ligand N/O min distance (A)"); ax[0].set_xlabel("ns"); ax[0].legend()
ax[1].set_title(f"Pocket contact occupancy (<{POCKET_CUTOFF} A)"); ax[1].set_xlabel("residue")
fig.tight_layout(); fig.savefig("figures/contacts.png", dpi=150)

# PCA on CA coordinates of all replicas pooled, KMeans on the first 3 PCs
X = np.vstack([ca_xyz[s] for s in SEEDS])
lab = np.concatenate([[s] * len(ca_xyz[s]) for s in SEEDS])
pcs = PCA(n_components=3).fit_transform(X)
cl = KMeans(n_clusters=min(3, len(X)), n_init=10, random_state=0).fit_predict(pcs)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s in SEEDS:
    ax[0].scatter(*pcs[lab == s, :2].T, s=4, label=f"seed {s}")
ax[1].scatter(*pcs[:, :2].T, s=4, c=cl)
for a_, t in zip(ax, ["PCA by replica", "KMeans clusters (k=3)"]):
    a_.set_title(t); a_.set_xlabel("PC1"); a_.set_ylabel("PC2")
ax[0].legend(); fig.tight_layout(); fig.savefig("figures/pca_clusters.png", dpi=150)
print("cluster occupancy per replica:")
for s in SEEDS:
    print(f"  seed {s}:", np.bincount(cl[lab == s], minlength=3) / (lab == s).sum())
