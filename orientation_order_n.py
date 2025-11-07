# Hermans orientation for UEF with GKR boundary conditions
# Computes P2(n) for subchains n in {0.2,0.5,1,2,5,10} Ne (Ne=28)
# Uses rotation/uefex matrix to de-rotate subchain vectors into the co-deforming frame.

import math
import numpy as np
import csv
import time

start_time = time.time()

# ----------------- Readers -----------------

def read_dump_triclinic_xu(path, frame_limit=None):
    ts = {}
    frames = 0
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM:"):
                continue
            if "TIMESTEP" in line:
                t = int(f.readline().strip())
                ts[t] = {}

                if not f.readline().startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError("Dump malformed: expected 'ITEM: NUMBER OF ATOMS'")
                n = int(f.readline().strip())
                ts[t]["atoms"] = n

                bb = f.readline().strip()
                if not (bb.startswith("ITEM: BOX BOUNDS") and all(k in bb for k in ("xy","xz","yz"))):
                    raise ValueError("Expect 'ITEM: BOX BOUNDS xy xz yz ...'")
                xlo, xhi, xy = map(float, f.readline().split())
                ylo, yhi, xz = map(float, f.readline().split())
                zlo, zhi, yz = map(float, f.readline().split())

                lx = xhi - xlo; ly = yhi - ylo; lz = zhi - zlo
                a = (lx, 0.0, 0.0); b = (xy, ly, 0.0); c = (xz, yz, lz)
                La = math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
                Lb = math.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
                Lc = math.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
                ts[t]["a"], ts[t]["b"], ts[t]["c"] = a, b, c
                ts[t]["La"], ts[t]["Lb"], ts[t]["Lc"] = La, Lb, Lc

                atoms_hdr = f.readline().strip()
                if not atoms_hdr.startswith("ITEM: ATOMS"):
                    raise ValueError("Dump malformed: expected 'ITEM: ATOMS ...'")
                cols = atoms_hdr.split()[2:]
                idx = {name:i for i,name in enumerate(cols)}
                for need in ("id","mol","type","xu","yu","zu"):
                    if need not in idx:
                        raise ValueError(f"Dump missing column '{need}'")

                ts[t]["mols"] = {}
                for _ in range(n):
                    p = f.readline().split()
                    ida  = int(p[idx["id"]]); mol = int(p[idx["mol"]]); typ = int(p[idx["type"]])
                    x    = float(p[idx["xu"]]); y  = float(p[idx["yu"]]); z  = float(p[idx["zu"]])
                    ts[t]["mols"].setdefault(mol, {"atoms":[]})["atoms"].append((ida,typ,x,y,z))

                for m, md in ts[t]["mols"].items():
                    md["length"] = len(md["atoms"])

                frames += 1
                if frame_limit and frames >= frame_limit:
                    break
    return ts

def read_rotation_file(path):
    R = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 10:
                continue
            t = int(parts[0])
            vals = list(map(float, parts[1:10]))
            R[t] = np.array(vals, dtype=float).reshape(3,3)
    return R

# ----------------- Orientation P2(n) -----------------

def _unit_vec(v):
    n2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    if n2 <= 0.0: return (0.0,0.0,0.0)
    n = math.sqrt(n2);  return (v[0]/n, v[1]/n, v[2]/n)

def _round_n(x, mode="nearest"):
    if mode == "floor":  return max(1, math.floor(x))
    if mode == "ceil":   return max(1, math.ceil(x))
    return max(1, int(round(x)))  # nearest

def compute_P2_subchains(
    ts, min_chain_len, max_chain_len,
    Ne=28, n_multipliers=(0.2,0.5,1,2,5,10), rounding="nearest",
    use_rotation=True, Rdict=None, assume_dump_sorted_by_id=True
):
    """
    Returns:
      t_sorted: list of timesteps
      hencky:   list of Hencky strain values
      p2_cols:  dict[label] -> list of P2(n) per timestep
                label is like f'P2(n={m}Ne)'
    """
    if not ts:
        return [], [], {}

    t_sorted = sorted(ts.keys())
    Lc0 = ts[t_sorted[0]]["Lc"]
    ez = np.array([0.0, 0.0, 1.0])
    n_bonds_list = [_round_n(m*Ne, rounding) for m in n_multipliers]
    labels = [f"P2(n={m}Ne)" for m in n_multipliers]
    p2_cols = {lab: [] for lab in labels}
    hencky = []

    for t in t_sorted:
        Lc = ts[t]["Lc"]
        hencky.append(math.log(Lc / Lc0))

        # choose projection direction
        use_R = (use_rotation and Rdict is not None and t in Rdict)
        Rt = Rdict[t].T if use_R else None
        chat = np.array(_unit_vec(ts[t]["c"]), dtype=float) if not use_R else None

        # accumulators per n
        sums_cos2 = [0.0 for _ in n_bonds_list]
        counts    = [0    for _ in n_bonds_list]

        for _, md in ts[t]["mols"].items():
            n_atoms = md["length"]
            if n_atoms < min_chain_len or n_atoms > max_chain_len:
                continue

            atoms_sorted = md["atoms"] if assume_dump_sorted_by_id else sorted(md["atoms"], key=lambda r: r[0])

            # slide windows for each n (n bonds -> n+1 atoms)
            for j, n_b in enumerate(n_bonds_list):
                win = n_b + 1
                if n_atoms < win:
                    continue
                for i in range(n_atoms - win + 1):
                    _, _, x1, y1, z1 = atoms_sorted[i]
                    _, _, x2, y2, z2 = atoms_sorted[i + n_b]
                    AB = np.array([x2-x1, y2-y1, z2-z1], dtype=float)
                    AB2 = float(AB.dot(AB))
                    if AB2 <= 1e-30:
                        continue

                    if use_R:
                        ABp = Rt @ AB
                        cos2 = (ABp[2]*ABp[2]) / AB2   # z-component in co-deforming frame
                    else:
                        dot = float(AB.dot(chat))
                        cos2 = (dot*dot) / AB2         # projection on c-hat in lab frame

                    sums_cos2[j] += cos2
                    counts[j]    += 1

        # finalize P2 for this timestep
        for lab, j in zip(labels, range(len(n_bonds_list))):
            if counts[j] > 0:
                p2 = (3.0*(sums_cos2[j]/counts[j]) - 1.0)/2.0
            else:
                p2 = float('nan')
            p2_cols[lab].append(p2)

    return t_sorted, hencky, p2_cols

def write_csv_multi(tsteps, hencky, p2_cols, path):
    labels = list(p2_cols.keys())
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(["Timestep", "Hencky_Strain"] + labels)
        for i, t in enumerate(tsteps):
            row = [t, hencky[i]] + [p2_cols[lab][i] for lab in labels]
            wr.writerow(row)
    print("Saved:", path)

# ----------------- Parameters -----------------

DUMP_FILE       = "dump.lammps-frame.lammpstrj"   # triclinic xu/yu/zu dump
ROTATION_FILE   = "rotation.txt"                  # c_rmatrix[*] ave/time output
OUTPUT_CSV      = "P2_subchains_UEF_GKR_1.0_360_R1_v1.csv"

MIN_CHAIN_LEN   = 345
MAX_CHAIN_LEN   = 375
FRAME_LIMIT     = None

USE_ROTATION    = True
Ne              = 28
N_MULTIPLIERS   = (0.2, 0.5, 1, 2, 6, 12)
ROUNDING        = "nearest"      # or "floor"/"ceil"

# ----------------- Run -----------------

ts    = read_dump_triclinic_xu(DUMP_FILE, frame_limit=FRAME_LIMIT)
Rdict = read_rotation_file(ROTATION_FILE) if USE_ROTATION else None

tsteps, eH, p2cols = compute_P2_subchains(
    ts, MIN_CHAIN_LEN, MAX_CHAIN_LEN,
    Ne=Ne, n_multipliers=N_MULTIPLIERS, rounding=ROUNDING,
    use_rotation=USE_ROTATION, Rdict=Rdict,
    assume_dump_sorted_by_id=True
)

write_csv_multi(tsteps, eH, p2cols, OUTPUT_CSV)

# Timing / info
elapsed = (time.time() - start_time)/60.0
print(f"Execution time: {elapsed:.2f} min | Frames: {len(tsteps)} | Chains window: [{MIN_CHAIN_LEN},{MAX_CHAIN_LEN}] | Rotation used: {USE_ROTATION and (Rdict is not None)}")