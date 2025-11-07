#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, csv, os
from typing import Dict, List, Tuple, Optional
import numpy as np

# ==================== USER SETTINGS ====================
# ---- SEQUENCE 1 (required) ----
SP_FILE        = "Z1+SP_1.dat"                   # Z1+ -SP+ output (per-frame Zi)
DUMP_FILE      = "dump.lammps-frame_part1.lammpstrj" # trajectory with id mol x*/y*/z*
ROT_FILE       = "rotation.txt"                # optional; lab→flow rotation, 3x3 / line
REE_FILE       = "Ree_values_1.dat"              # single-column |Ree| per chain per frame
LPP_FILE       = "Lpp_values_1.dat"              # single-column Lpp per chain per frame
N_FILE         = "N_values_1.dat"                # single-column chain length N per chain per frame

# ---- SEQUENCE 2 (optional continuation) ----
# If you have a second (continued) trajectory/analysis, point these to the files.
# Leave as None if you don't have sequence 2.
SP_FILE2       = "Z1+SP_2.dat" #None          # e.g., "Z1+SP_2.dat"
DUMP_FILE2     = "dump.lammps-frame_part2.lammpstrj" # None          # e.g., "dump2.lammpstrj"
ROT_FILE2      = "rotation_cont.txt" # None          # e.g., "rotation_2.txt"
REE_FILE2      = "Ree_values_2.dat" #None          # e.g., "Ree_values_2.dat"
LPP_FILE2      = "Lpp_values_2.dat" #None          # e.g., "Lpp_values_2.dat"
N_FILE2        = "N_values_2.dat" #None          # e.g., "N_values_2.dat"

# ---- GLOBAL LIMITS ----
# Stop after this many frames *across both sequences*. Use None to process all.
MAX_USE_FRAMES = 595          # set to an int (e.g., 250) to cap, or None to use all

# ---- Physics / Model ----
b_stat        = 0.966
rho_monomer   = 0.85
C_INF         = 2.7
kBT           = 1.0
MIN_ZI        = 1
CLAMP_X       = 0.995
INVL_TYPE     = "pade"        # "pade" | "kroger" | "jedynak"
USE_KUHN_LEN  = False

OUTPUT_CSV    = "rdp_stress_vs_strain_no_window_andZ.csv"
# =======================================================


# ---------- inverse Langevin ----------
def inv_langevin(x: float) -> float:
    ax = abs(x)
    if ax >= 1.0:
        return math.copysign(1e12, x)  # guard
    if INVL_TYPE == "pade":
        return x*(3 - ax*ax)/(1 - ax*ax)
    elif INVL_TYPE == "kroger":
        return x*(3 - ax*ax)/(1 - ax*ax) + 0.2*x
    elif INVL_TYPE == "jedynak":
        c = 1.1
        return x*(3 - ax*ax)/(1 - ax*ax) + (x/3.0)*(ax*ax)/(1 - c*ax*ax + 1e-12)
    return x*(3 - ax*ax)/(1 - ax*ax)

# ---------- small utils ----------
def _first_present(idx: Dict[str,int], options: List[str]) -> Optional[str]:
    for name in options:
        if name in idx: return name
    return None

# ---------- readers ----------
def parse_sp_plus_multiframe(path: str):
    """Z1+ -SP+ (multi-frame). Returns list of frames: ((Lx,Ly,Lz), {mol: Zi})."""
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line: break
            s = line.strip()
            if not s or not s.split()[0].lstrip('-').isdigit():
                continue
            C = int(s.split()[0])
            Lx, Ly, Lz = map(float, f.readline().split()[:3])
            chains = {}
            for mol in range(1, C+1):
                sn = f.readline().strip()
                if not sn: break
                try:
                    n_nodes = int(sn.split()[0])
                except:
                    sn = f.readline().strip()
                    if not sn: break
                    n_nodes = int(sn.split()[0])
                Zi = 0
                for _ in range(n_nodes):
                    p = f.readline().split()
                    if len(p) >= 5 and int(float(p[4])) == 1: Zi += 1
                chains[mol] = Zi
            frames.append(((Lx, Ly, Lz), chains))
    return frames

def read_dump(path: str):
    """LAMMPS dump (xu/yu/zu or x/y/z) grouped by molecule, keyed by timestep."""
    ts = {}
    with open(path) as f:
        while True:
            line = f.readline()
            if not line: break
            if "ITEM: TIMESTEP" not in line: continue
            t = int(f.readline().strip())

            if "ITEM: NUMBER OF ATOMS" not in f.readline():
                raise ValueError("Dump malformed (NUMBER OF ATOMS).")
            n = int(f.readline().strip())

            bb = f.readline().strip()
            if not bb.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Dump malformed (BOX BOUNDS).")
            _ = f.readline().split(); _ = f.readline().split()
            zline = f.readline().split()
            zlo, zhi = float(zline[0]), float(zline[1])
            Lz = zhi - zlo

            hdr = f.readline().strip()
            if not hdr.startswith("ITEM: ATOMS"):
                raise ValueError("Dump malformed (ATOMS header).")
            cols = hdr.split()[2:]
            idx = {name:i for i,name in enumerate(cols)}
            mol_key = _first_present(idx, ["mol", "molecule-ID", "molecule", "molid", "molID"])
            x_key   = _first_present(idx, ["xu", "x"])
            y_key   = _first_present(idx, ["yu", "y"])
            z_key   = _first_present(idx, ["zu", "z"])
            missing = [lab for lab,key in [("molecule id", mol_key), ("x/xu", x_key),
                                           ("y/yu", y_key), ("z/zu", z_key)] if key is None]
            if missing: raise KeyError(f"Dump reader missing {missing}. Found: {cols}")

            mols: Dict[int, List[List[float]]] = {}
            for _ in range(n):
                p = f.readline().split()
                mol = int(p[idx[mol_key]])
                x = float(p[idx[x_key]]); y = float(p[idx[y_key]]); z = float(p[idx[z_key]])
                mols.setdefault(mol, []).append([x,y,z])
            for m in mols: mols[m] = np.array(mols[m], dtype=float)

            ts[t] = {"Lz": Lz, "mols": mols}
    return ts

def read_rotations(path: str):
    """rotation.txt: timestep R11..R13 R21..R23 R31..R33 (lab→flow)."""
    R = {}
    if not path or not os.path.exists(path): return R
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            p = s.split()
            if len(p) != 10: continue
            t = int(p[0]); mat = np.array(list(map(float, p[1:])), float).reshape(3,3)
            R[t] = mat
    return R

def read_single_column(fname: str) -> np.ndarray:
    vals = []
    with open(fname, "r") as f:
        for line in f:
            s = line.strip()
            if s: vals.append(float(s.split()[0]))
    return np.array(vals, dtype=float)

# ---------- Ne(mod–S–coil) per-chain, per-frame ----------
def build_Ne_mod_scoil_per_chain(ree_col: np.ndarray,
                                 lpp_col: np.ndarray,
                                 N_col:   np.ndarray,
                                 n_frames: int,
                                 n_chains: int) -> np.ndarray:
    """
    Per-chain Ne_i(t) = (N_i - 1) / ( (Lpp_i^2 / Ree_i^2) - 1 ), safely clipped.
    Inputs are per-chain, per-frame stacked as single columns of length ~ frames*chains.
    Returns array of shape (n_frames, n_chains).
    """
    expect = n_frames * n_chains
    if min(ree_col.size, lpp_col.size, N_col.size) < expect:
        print(f"[warn] fewer rows than frames*chains; truncating to common length.")
    m = min(ree_col.size, lpp_col.size, N_col.size, expect)
    R  = ree_col[:m].reshape(n_frames, n_chains)
    Lp = lpp_col[:m].reshape(n_frames, n_chains)
    Ni =  N_col [:m].reshape(n_frames, n_chains)

    denom = ((Lp*Lp)/(R*R + 1e-30)) - 1.0
    denom = np.maximum(denom, 1e-8)
    Ne = (Ni - 1.0) / denom
    Ne = np.clip(Ne, 1.0, 1e6)
    return Ne

# ---------- stress routes ----------
def frame_stress_Z1prime(Lz0: float, Lz: float,
                         dump_mols: Dict[int, np.ndarray],
                         Zi_dict: Dict[int, int],
                         Rmat: Optional[np.ndarray],
                         PREF: float,
                         b_used: float) -> Tuple[float,float,float]:
    """
    Z1+ route: heterogeneous per-chain N_e' = N/(Z_i+1) - 1 (uses kink counts).
    Chains with Z_i < MIN_ZI are SKIPPED.
    """
    lam = Lz / Lz0
    eH  = math.log(lam)
    sigma_terms, P2_terms = [], []

    for mol, coords in dump_mols.items():
        Zi = Zi_dict.get(mol, 0)
        if Zi < MIN_ZI:
            continue
        Ni = coords.shape[0]

        # --- robust Ne' (avoid zero/neg/NaN) ---
        Ne_prime_raw = Ni / (Zi + 1.0) - 1.0
        if not np.isfinite(Ne_prime_raw):
            continue
        Ne_prime = max(1.0, float(Ne_prime_raw))  # clamp to >= 1.0

        w = max(1, int(round(Ne_prime)))
        if w >= Ni:
            continue

        X = coords if Rmat is None else (Rmat.T @ coords.T).T  # lab→flow assumed
        for i in range(0, Ni - w, w):
            Rv = X[i+w] - X[i]
            Rmag = float(np.linalg.norm(Rv))
            if Rmag <= 1e-12:
                continue
            denom = max(Ne_prime * b_used, 1e-12)  # guard vs 0
            x = min(CLAMP_X, max(0.0, Rmag / denom))
            Linv = inv_langevin(x)
            cos2 = (Rv[2]*Rv[2]) / (Rmag*Rmag)
            P2   = 0.5*(3.0*cos2 - 1.0)
            sigma_terms.append(Linv * P2)
            P2_terms.append(P2)

    if not sigma_terms:
        return eH, 0.0, 0.0
    sigma = PREF * float(np.mean(sigma_terms))
    P2ne  = float(np.mean(P2_terms)) if P2_terms else 0.0
    return eH, sigma, P2ne

def frame_stress_modS(Lz0: float, Lz: float,
                      dump_mols: Dict[int, np.ndarray],
                      Ne_row: np.ndarray,               # (n_chains,) for this frame
                      Zi_dict: Dict[int, int],          # Z-filter
                      Rmat: Optional[np.ndarray],
                      PREF: float,
                      b_used: float) -> Tuple[float,float,float]:
    """
    Modified S–coil route: heterogeneous per-chain Ne_i(t).
    Chains with Z_i < MIN_ZI are SKIPPED.
    """
    lam = Lz / Lz0
    eH  = math.log(lam)
    sigma_terms, P2_terms = [], []

    for mol, coords in dump_mols.items():
        Zi = Zi_dict.get(mol, 0)
        if Zi < MIN_ZI:
            continue

        idx = mol - 1  # assumes mol IDs are 1..M
        if idx < 0 or idx >= Ne_row.size:
            continue
        Ne_i = float(Ne_row[idx])

        w = max(1, int(round(Ne_i)))
        if w >= coords.shape[0]:
            continue

        X = coords if Rmat is None else (Rmat.T @ coords.T).T
        for i in range(0, coords.shape[0] - w, w):
            Rv = X[i+w] - X[i]
            Rmag = float(np.linalg.norm(Rv))
            if Rmag <= 1e-12:
                continue
            x = min(CLAMP_X, max(0.0, Rmag / (max(Ne_i, 1.0) * b_used + 1e-12)))
            Linv = inv_langevin(x)
            cos2 = (Rv[2]*Rv[2]) / (Rmag*Rmag)
            P2   = 0.5*(3.0*cos2 - 1.0)
            sigma_terms.append(Linv * P2)
            P2_terms.append(P2)

    if not sigma_terms:
        return eH, 0.0, 0.0
    sigma = PREF * float(np.mean(sigma_terms))
    P2ne  = float(np.mean(P2_terms)) if P2_terms else 0.0
    return eH, sigma, P2ne


# ---------- helpers for merging two sequences ----------
def concat_rotations(R1: Dict[int, np.ndarray], R2: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    if not R2: return R1
    R = dict(R1)
    for t, M in R2.items():  # sequence-2 wins on overlap
        R[t] = M
    return R

def concat_sp_frames(F1: List[Tuple[Tuple[float,float,float], Dict[int,int]]],
                     F2: Optional[List[Tuple[Tuple[float,float,float], Dict[int,int]]]]) \
                     -> List[Tuple[Tuple[float,float,float], Dict[int,int]]]:
    if not F2: return F1
    return F1 + F2

def concat_Ne_blocks(n_chains:int,
                     ree1: np.ndarray, lpp1: np.ndarray, N1: np.ndarray, n_frames1: int,
                     ree2: Optional[np.ndarray], lpp2: Optional[np.ndarray], N2: Optional[np.ndarray], n_frames2: int) -> np.ndarray:
    A1 = build_Ne_mod_scoil_per_chain(ree1, lpp1, N1, n_frames1, n_chains)
    if ree2 is None or lpp2 is None or N2 is None or n_frames2 == 0:
        return A1
    A2 = build_Ne_mod_scoil_per_chain(ree2, lpp2, N2, n_frames2, n_chains)
    return np.vstack([A1, A2])


# -------------------- MAIN --------------------
def main():
    # Load dump(s)
    dump1 = read_dump(DUMP_FILE)
    if not dump1:
        raise RuntimeError(f"No frames parsed from {DUMP_FILE}")
    dump = dict(dump1)

    if DUMP_FILE2 is not None:
        dump2 = read_dump(DUMP_FILE2)
        if not dump2:
            raise RuntimeError(f"No frames parsed from {DUMP_FILE2}")
        # merge dumps; assume continuation (later timesteps), but allow overlap (second wins)
        for t, rec in dump2.items():
            dump[t] = rec

    # Global timestep list (sorted)
    tlist = sorted(dump.keys())
    Lz0 = dump[tlist[0]]["Lz"]

    # Rotations
    R1 = read_rotations(ROT_FILE)
    R2 = read_rotations(ROT_FILE2) if ROT_FILE2 else {}
    R  = concat_rotations(R1, R2)

    PREF   = (rho_monomer / C_INF) * kBT
    b_used = (C_INF*b_stat) if USE_KUHN_LEN else b_stat

    # Z1+ frames
    sp_frames1 = parse_sp_plus_multiframe(SP_FILE)
    sp_frames2 = parse_sp_plus_multiframe(SP_FILE2) if SP_FILE2 else None
    sp_frames  = concat_sp_frames(sp_frames1, sp_frames2)

    # Determine chains/frame from dump
    n_chains = len(dump[tlist[0]]["mols"])

    # Per-sequence frame counts for Ne(mod-S-coil)
    n_frames1 = len(sp_frames1)
    n_frames2 = len(sp_frames2) if sp_frames2 else 0

    # Read per-chain columns (sequence 1)
    ree1 = read_single_column(REE_FILE)
    lpp1 = read_single_column(LPP_FILE)
    Ncol1= read_single_column(N_FILE)

    # Optional per-chain columns (sequence 2)
    ree2 = read_single_column(REE_FILE2) if REE_FILE2 else None
    lpp2 = read_single_column(LPP_FILE2) if LPP_FILE2 else None
    Ncol2= read_single_column(N_FILE2) if N_FILE2 else None

    # Build Ne(mod–S–coil), concatenated across sequences
    Ne_modS = concat_Ne_blocks(n_chains, ree1, lpp1, Ncol1, n_frames1,
                               ree2, lpp2, Ncol2, n_frames2)

    total_frames_from_sp   = len(sp_frames)
    total_frames_from_ne   = Ne_modS.shape[0]
    total_frames_from_dump = len(tlist)

    # Align all three; take the minimum and cap by MAX_USE_FRAMES
    n_frames = min(total_frames_from_sp, total_frames_from_ne, total_frames_from_dump)
    if MAX_USE_FRAMES is not None:
        n_frames = min(n_frames, MAX_USE_FRAMES)
    if n_frames == 0:
        raise RuntimeError("No common frames to process (check inputs).")

    print(f"[info] frames available -> dump={total_frames_from_dump}, Z1+={total_frames_from_sp}, Ne(modS)={total_frames_from_ne}")
    print(f"[info] using n_frames={n_frames} (capped by MAX_USE_FRAMES={MAX_USE_FRAMES})")
    print(f"[info] chains/frame={n_chains}")

    rows = []
    for k in range(n_frames):
        t   = tlist[k]
        Lz  = dump[t]["Lz"]
        Rmt = R.get(t, None)

        # Correctly unpack SP frame: ((Lx,Ly,Lz), Zi_dict)
        Zi_dict = sp_frames[k][1]

        # Route 1: mod–S–coil with Z-filter
        eH_A, sigma_modS, P2_modS = frame_stress_modS(
            Lz0, Lz, dump[t]["mols"], Ne_modS[k, :], Zi_dict, Rmt, PREF, b_used
        )

        # Route 2: Z1+ (kink-based) with robust Ne' handling
        eH_B, sigma_Neprime, P2_Neprime = frame_stress_Z1prime(
            Lz0, Lz, dump[t]["mols"], Zi_dict, Rmt, PREF, b_used
        )

        rows.append([k, t, eH_A, sigma_modS, P2_modS, sigma_Neprime, P2_Neprime])

        if k % 10 == 0:
            kept = sum(1 for m in dump[t]["mols"] if Zi_dict.get(m, 0) >= MIN_ZI)
            print(f"[{k:03d}] t={t} eH={eH_A:.3f} kept_chains(Z≥{MIN_ZI})={kept}  "
                  f"σ_modS={sigma_modS:.3f}  σ_Ne'={sigma_Neprime:.3f}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Frame","Timestep","Hencky_Strain",
                     "sigma_modS_hetero_Zfiltered","avgP2_modS",
                     "sigma_Neprime_hetero","avgP2_Neprime"])
        wr.writerows(rows)

    print(f"Saved {OUTPUT_CSV} | INVL={INVL_TYPE} | prefactor=rho/C_inf | "
          f"b_used={'Kuhn' if USE_KUHN_LEN else 'stat'} | CLAMP_X={CLAMP_X} | MIN_ZI={MIN_ZI}")


if __name__ == "__main__":
    main()
