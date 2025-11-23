#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np

# ========================= Reader (yours, with 2 tiny tweaks) =========================
def readTimesteps(inputFile, timestepsCount=None):
    """
    Reads a LAMMPS dump with:
      ITEM: TIMESTEP
      <timestep>
      ITEM: NUMBER OF ATOMS
      <natoms>
      ITEM: BOX BOUNDS ...
      <xlo xhi>
      <ylo yhi>
      <zlo zhi>
      ITEM: ATOMS id mol type xu yu zu
    Returns: dict[timestep] -> {
        'atoms': N, '-x','x','-y','y','-z','z',
        'headers': str,
        'mols': { mol_id: {
            'atoms': [(id,typ,xu,yu,zu), ...],
            'length': int, 'xcm','ycm','zcm'
        }}
    }
    """
    counter = 0
    timesteps = {}
    with open(inputFile) as f:
        line = f.readline()
        while line != '':
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(f.readline().split()[0])
                timesteps[timestep] = {}

                _ = f.readline()  # ITEM: NUMBER OF ATOMS
                atoms = int(f.readline().split()[0])
                timesteps[timestep]["atoms"] = atoms

                _ = f.readline()  # ITEM: BOX BOUNDS ...
                xlo, xhi = map(float, f.readline().split()[:2])
                ylo, yhi = map(float, f.readline().split()[:2])
                zlo, zhi = map(float, f.readline().split()[:2])
                timesteps[timestep]["-x"] = xlo; timesteps[timestep]["x"]  = xhi
                timesteps[timestep]["-y"] = ylo; timesteps[timestep]["y"]  = yhi
                timesteps[timestep]["-z"] = zlo; timesteps[timestep]["z"]  = zhi

                headers = f.readline().strip()  # "ITEM: ATOMS id mol type xu yu zu"
                timesteps[timestep]["headers"] = headers
                timesteps[timestep]["mols"] = {}

                for _ in range(atoms):
                    parts = f.readline().split()
                    a_id  = int(parts[0]); mol = int(parts[1]); a_typ = int(parts[2])
                    xu    = float(parts[3]); yu  = float(parts[4]);  zu = float(parts[5])
                    timesteps[timestep]["mols"].setdefault(mol, {"atoms": []})
                    timesteps[timestep]["mols"][mol]["atoms"].append((a_id, a_typ, xu, yu, zu))

                # aggregate per-molecule stats
                for mol, mdata in timesteps[timestep]["mols"].items():
                    # sort atoms by id to approximate chain order (safer than file order)
                    mdata["atoms"].sort(key=lambda r: r[0])
                    length = len(mdata["atoms"])
                    mdata["length"] = length
                    xs = sum(a[2] for a in mdata["atoms"])
                    ys = sum(a[3] for a in mdata["atoms"])
                    zs = sum(a[4] for a in mdata["atoms"])
                    mdata["xcm"] = xs/length; mdata["ycm"] = ys/length; mdata["zcm"] = zs/length

                counter += 1
                if timestepsCount is not None and counter >= timestepsCount:
                    break
            line = f.readline()
    return timesteps

# ========================= Birefringence computation =========================
def bond_unit_vectors_from_chain(chain_atoms: List[Tuple[int,int,float,float,float]],
                                 stride: int = 1) -> np.ndarray:
    """
    Build unit bond vectors from a single chain's atom list (sorted by id).
    stride=1 -> consecutive beads; stride>1 -> coarse-grained segments (reduces noise).
    Returns array of shape (Nbonds, 3). If <2 points available, returns empty (0,3).
    """
    if len(chain_atoms) < 2*stride:
        return np.zeros((0,3), dtype=float)
    # positions array
    X = np.array([[a[2], a[3], a[4]] for a in chain_atoms], dtype=float)
    # coarse-grained segment vectors: X[i+stride]-X[i]
    vecs = X[stride:] - X[:-stride]
    # keep every 'stride' to avoid overlapping segments if desired (optional)
    # vecs = vecs[::stride]
    norms = np.linalg.norm(vecs, axis=1)
    mask = norms > 1e-12
    if not np.any(mask):
        return np.zeros((0,3), dtype=float)
    u = vecs[mask] / norms[mask, None]
    return u

def orientation_tensor_from_units(U: np.ndarray) -> np.ndarray:
    """
    Given N x 3 unit vectors, compute orientation tensor S = <uu> - I/3.
    Returns 3x3 array. If no vectors, returns zeros.
    """
    if U.size == 0:
        return np.zeros((3,3), dtype=float)
    C = (U.T @ U) / U.shape[0]         # second moment
    S = C - np.eye(3)/3.0
    return S

def birefringence_from_S(S: np.ndarray, axis: int = 2) -> float:
    """
    For uniaxial extension along 'axis' (0=x, 1=y, 2=z),
    Δn ∝ S_aa - 0.5*(S_bb + S_cc) with (a,b,c) a permutation of (x,y,z).
    """
    a = axis
    b = (axis + 1) % 3
    c = (axis + 2) % 3
    return float(S[a, a] - 0.5*(S[b, b] + S[c, c]))

def compute_birefringence_time_series(timesteps: Dict[int, dict],
                                      stride: int = 1,
                                      axis: int = 2,
                                      per_chain: bool = False,
                                      z_filter_min_length: int = 2) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[int,float]]]:
    """
    Computes Δn per frame from bond orientations.
    - stride: bond spacing (1 = nearest-neighbor, >1 = coarse-grain).
    - axis: extension axis index (2 = z).
    - per_chain: if True, also returns Δn per chain per frame.
    - z_filter_min_length: ignore chains shorter than this many beads (after considering stride).
    Returns:
      timesteps_sorted: np.array of timestep ids (sorted),
      delta_n: np.array of Δn per frame (same order),
      delta_n_per_chain: dict[timestep][mol_id] = Δn_chain (if per_chain=True), else {}
    """
    ts_sorted = np.array(sorted(timesteps.keys()), dtype=int)
    delta_n = []
    delta_n_per_chain: Dict[int, Dict[int,float]] = {}

    for t in ts_sorted:
        mols = timesteps[t]["mols"]

        # collect all unit bond vectors in the frame
        U_all = []
        chain_map = {}

        for mol_id, mdata in mols.items():
            if mdata["length"] < z_filter_min_length:
                continue
            U_chain = bond_unit_vectors_from_chain(mdata["atoms"], stride=stride)
            if U_chain.shape[0] == 0:
                continue
            U_all.append(U_chain)
            if per_chain:
                S_chain = orientation_tensor_from_units(U_chain)
                dn_chain = birefringence_from_S(S_chain, axis=axis)
                chain_map[mol_id] = dn_chain

        if len(U_all) == 0:
            delta_n.append(0.0)
            if per_chain:
                delta_n_per_chain[t] = {}
            continue

        U_all = np.vstack(U_all)   # (Ntotal,3)
        S_tot = orientation_tensor_from_units(U_all)
        dn_tot = birefringence_from_S(S_tot, axis=axis)
        delta_n.append(dn_tot)

        if per_chain:
            delta_n_per_chain[t] = chain_map

    return ts_sorted, np.array(delta_n, dtype=float), delta_n_per_chain

# ========================= Convenience I/O =========================
def write_birefringence_csv(out_csv: str,
                            ts_sorted: np.ndarray,
                            delta_n: np.ndarray,
                            delta_n_per_chain: Optional[Dict[int, Dict[int,float]]] = None):
    """
    Writes:
      - frame-level Δn
      - (optional) per-chain Δn as additional columns (mol_<id>)
    Note: per-chain columns are sparse if chains appear/disappear; absent entries are blank.
    """
    # Gather full set of mol IDs if we want per-chain output
    all_mols = set()
    if delta_n_per_chain:
        for t in delta_n_per_chain:
            all_mols.update(delta_n_per_chain[t].keys())
    mol_list = sorted(all_mols)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        header = ["Timestep", "DeltaN_orientation"] + [f"mol_{m}" for m in mol_list]
        wr.writerow(header)
        for i, t in enumerate(ts_sorted):
            row = [t, f"{delta_n[i]:.8f}"]
            if delta_n_per_chain:
                cmap = delta_n_per_chain.get(int(t), {})
                row += [f"{cmap.get(m, ''):.8f}" if m in cmap else "" for m in mol_list]
            wr.writerow(row)

# ========================= Example usage =========================
if __name__ == "__main__":
    # ---- user inputs ----
    DUMP_FILE = "dump.flow-frame_1.0_360_R1.lammpstrj"   # already in flow frame
    MAX_FRAMES = None              # set to an int to cap frames (e.g., 500), or None for all
    STRIDE = 1                     # 1 = nearest-neighbor bonds; try 2–10 for Kuhn-like coarse-grain
    EXT_AXIS = 2                   # 0=x, 1=y, 2=z (uniaxial extension along z)
    OUTPUT_CSV = "birefringence_from_orientation.csv"
    WANT_PER_CHAIN = True          # also write per-chain Δn columns

    # ---- read and (optionally) cap frames ----
    ts = readTimesteps(DUMP_FILE, timestepsCount=MAX_FRAMES)

    # ---- compute Δn(t) ----
    ts_sorted, delta_n, delta_n_pc = compute_birefringence_time_series(
        ts, stride=STRIDE, axis=EXT_AXIS, per_chain=WANT_PER_CHAIN, z_filter_min_length=max(2, STRIDE+1)
    )

    # ---- write CSV ----
    write_birefringence_csv(OUTPUT_CSV, ts_sorted, delta_n, delta_n_pc if WANT_PER_CHAIN else None)

    print(f"[done] Wrote {OUTPUT_CSV} with {len(ts_sorted)} frames. "
          f"(Δn is in relative units ∝ orientation; scale later if you want absolute optics.)")
