#!/usr/bin/env python3
# Tube radius evolution δr(n)/δr_eq(n) from dump + Z1+ (1-column files)
# Robust to Z=0 at t0 and allows manual Ne0 override.

import math, csv
import numpy as np
from collections import defaultdict

# ---------------- user inputs ----------------
DUMP_FILE      = "dump.lammps-frame.lammpstrj"      # must contain: id mol xu yu zu
Z1_Z_FILE      = "Z_values.dat"        # 1-column, flattened over (chains, times)
Z1_N_FILE      = "N_values.dat"        # 1-column, flattened over (chains, times)
SERIES_ORDER   = "chain_major"         # or "time_major"

Ne_multipliers = (0.2, 0.5, 1, 2, 6, 12) # (0.5, 1, 2, 5)        # n = multiplier * Ne0
central_frac   = 0.20                   # central 20% of window
Nmin, Nmax     = 340, 380               # keep chains with N in [Nmin, Nmax]
ASSUME_SORTED_BY_ID = True
FRAME_SUBSAMPLE = 1
OUT_CSV         = "tube_radius_ratio_vs_strain_2.csv"

# If Z(t0) has zeros and you want to force Ne0, set this (else None)
MANUAL_NE0 = None    # e.g., MANUAL_NE0 = 28.0

# ---------------- utils ----------------
def read_dump_frames(path):
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line: break
            if not line.startswith("ITEM: TIMESTEP"): continue
            t = int(f.readline().split()[0])

            assert f.readline().startswith("ITEM: NUMBER OF ATOMS")
            nat = int(f.readline().split()[0])

            bb = f.readline().strip()
            _ = f.readline().split()
            _ = f.readline().split()
            l3 = f.readline().split()
            if "xy" in bb and "xz" in bb and "yz" in bb:
                zlo, zhi, yz = map(float, l3[:3])
            else:
                zlo, zhi = map(float, l3[:2])
            Lz = zhi - zlo

            cols = f.readline().split()[2:]
            idx = {name:i for i,name in enumerate(cols)}
            for need in ("id","mol","xu","yu","zu"):
                if need not in idx:
                    raise ValueError("Dump must have columns: id mol xu yu zu")
            i_id, i_mol, ix, iy, iz = idx["id"], idx["mol"], idx["xu"], idx["yu"], idx["zu"]

            chains = {}
            for _ in range(nat):
                p = f.readline().split()
                mol = int(float(p[i_mol])); ida = int(float(p[i_id]))
                x,y,z = float(p[ix]), float(p[iy]), float(p[iz])
                chains.setdefault(mol, []).append((ida, x, y, z))

            frame = {}
            for m, lst in chains.items():
                if not ASSUME_SORTED_BY_ID:
                    lst.sort(key=lambda r: r[0])
                coords = np.array([[x,y,z] for _,x,y,z in lst], float)
                frame[m] = coords
            yield t, Lz, frame

def first_frame_info(dump_path):
    gen = read_dump_frames(dump_path)
    t0, Lz0, frame0 = next(gen)
    N0 = {m: coords.shape[0] for m, coords in frame0.items()}
    return t0, Lz0, frame0, N0

def load_1col(path):
    vals = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            vals.append(float(s.split(",")[0]))
    return np.asarray(vals, float)

def reshape_TC(vec, C, T, order):
    if vec.size != C*T:
        raise ValueError(f"Vector length {vec.size} != C*T={C*T}")
    return vec.reshape(C, T).T if order=="chain_major" else vec.reshape(T, C)

def window_delta_r(coords, i0, i1, central_frac=0.2):
    r0 = coords[i0]; r1 = coords[i1]
    R  = r1 - r0; R2 = float(np.dot(R,R))
    if R2 <= 1e-30: return np.nan
    n_bonds = i1 - i0
    c0 = i0 + int(round(0.5*(1.0 - central_frac) * n_bonds))
    c1 = i0 + int(round(0.5*(1.0 + central_frac) * n_bonds))
    c0 = max(c0, i0); c1 = min(c1, i1)
    if c1 <= c0: return np.nan
    seg = coords[c0:c1+1] - r0
    u2  = np.einsum("ij,ij->i", seg, seg)
    uR  = seg @ R
    d2  = u2 - (uR*uR)/R2
    d2  = d2[d2 >= 0.0]
    return math.sqrt(float(np.mean(d2))) if d2.size else np.nan

# ---------------- main ----------------
def compute_tube_radius_ratio():
    # First frame
    t0, Lz0, frame0, N0_map = first_frame_info(DUMP_FILE)
    mol_ids = sorted(N0_map.keys())
    C = len(mol_ids)

    # Build time list and Lz(t)
    times, Lz_of_t = [], {}
    for i, (t, Lz, _) in enumerate(read_dump_frames(DUMP_FILE)):
        if i % FRAME_SUBSAMPLE: continue
        times.append(t); Lz_of_t[t] = Lz
    times = sorted(times); T = len(times)
    eps_t = {t: math.log(Lz_of_t[t]/Lz0) for t in times}

    # Z1+ series
    Z_tc = reshape_TC(load_1col(Z1_Z_FILE), C, T, SERIES_ORDER)
    N_tc = reshape_TC(load_1col(Z1_N_FILE), C, T, SERIES_ORDER)

    # Select chains by N window (at t0)
    N0_by_chain = N_tc[0, :]
    keep = np.ones(C, dtype=bool)
    if Nmin is not None: keep &= (N0_by_chain >= Nmin)
    if Nmax is not None: keep &= (N0_by_chain <= Nmax)
    idx = np.where(keep)[0]
    if idx.size == 0:
        raise RuntimeError("No chains in requested N-range")

    # ----- Robust Ne0 -----
    if MANUAL_NE0 is not None:
        Ne0 = float(MANUAL_NE0)
    else:
        Z0 = Z_tc[0, idx]
        N0 = N0_by_chain[idx]
        msk = (Z0 > 0.0) & np.isfinite(Z0) & (N0 > 1.0)
        if not np.any(msk):
            raise RuntimeError("All Z(t0) are zero/invalid in selection; set MANUAL_NE0.")
        Ne0 = float(np.nanmedian((N0[msk] - 1.0) / Z0[msk]))
    if not np.isfinite(Ne0) or Ne0 <= 0:
        raise RuntimeError(f"Computed Ne0 invalid: {Ne0}. Set MANUAL_NE0.")

    # Determine n(bonds) for each multiplier, clipped to feasible windows
    shortest_N = int(np.min(N0_by_chain[idx]))
    max_window = max(1, shortest_N - 1)  # need at least n+1 atoms
    n_list = []
    for m in Ne_multipliers:
        n_raw = int(round(m * Ne0))
        n = min(max(1, n_raw), max_window)
        n_list.append(n)
    labels = [f"n={m:g}Ne0({n} bonds)" for m, n in zip(Ne_multipliers, n_list)]

    # δr_eq(n) from first frame
    dreq = np.zeros(len(n_list))
    for j, n in enumerate(n_list):
        vals = []
        for k in idx:
            mol = mol_ids[k]; coords = frame0[mol]
            N = coords.shape[0]; win = n + 1
            if N < win: continue
            for i0 in range(0, N - win + 1):
                v = window_delta_r(coords, i0, i0+n, central_frac=central_frac)
                if not (v != v): vals.append(v)
        dreq[j] = float(np.mean(vals)) if vals else float("nan")

    # Walk the dump again to compute ratios
    rows = [["timestep","Hencky_epsilon"] + [f"delta_r/req ({lab})" for lab in labels]]

    # map timesteps to frames for quick access
    frame_iter = read_dump_frames(DUMP_FILE)
    frame_cache = {t: fr for t, _, fr in frame_iter for _ in [0]}

    for t in times:
        frame = frame_cache[t]
        ratios = []
        for j, n in enumerate(n_list):
            vals = []
            for k in idx:
                mol = mol_ids[k]; coords = frame[mol]
                N = coords.shape[0]; win = n + 1
                if N < win: continue
                for i0 in range(0, N - win + 1):
                    v = window_delta_r(coords, i0, i0+n, central_frac=central_frac)
                    if not (v != v): vals.append(v)
            dr = float(np.mean(vals)) if vals else float("nan")
            denom = dreq[j]
            ratios.append(dr/denom if denom and denom==denom else float("nan"))
        rows.append([t, eps_t[t]] + ratios)

    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Saved {OUT_CSV}")
    print(f"Chains used: {idx.size}/{C} | Frames: {T} | Ne0 = {Ne0:.2f} | n(bonds)={n_list}")

if __name__ == "__main__":
    compute_tube_radius_ratio()
