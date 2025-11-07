import math

# ---------- reader (unchanged) ----------
def readTimesteps(inputFile, timestepsCount=None):
    if timestepsCount is None:
        timestepsCount = 5
    counter = 0
    timesteps = {}
    with open(inputFile) as f:
        line = f.readline()
        while line != '':
            if line.startswith("ITEM:"):
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

                for mol, mdata in timesteps[timestep]["mols"].items():
                    length = len(mdata["atoms"])
                    mdata["length"] = length
                    xs = sum(a[2] for a in mdata["atoms"])
                    ys = sum(a[3] for a in mdata["atoms"])
                    zs = sum(a[4] for a in mdata["atoms"])
                    mdata["xcm"] = xs/length; mdata["ycm"] = ys/length; mdata["zcm"] = zs/length

                counter += 1
                if counter >= timestepsCount:
                    break
            line = f.readline()
    return timesteps

# ---------- helpers ----------
def _round_n(x, mode="nearest"):
    if mode == "floor":
        return max(1, math.floor(x))
    if mode == "ceil":
        return max(1, math.ceil(x))
    return max(1, int(round(x)))  # nearest

def _rms_R_of_n_for_timestep(ts_t, n_bonds_list, chain_length_range):
    """Return rms R(n) for each n at a single timestep (sqrt(mean(R^2)))."""
    sums_R2 = [0.0 for _ in n_bonds_list]
    counts  = [0   for _ in n_bonds_list]

    for _, mdata in ts_t["mols"].items():
        L_atoms = mdata["length"]
        if chain_length_range and len(chain_length_range) == 2:
            mn, mx = chain_length_range
            if not (mn <= L_atoms <= mx):
                continue

        atoms_sorted = sorted(mdata["atoms"], key=lambda a: a[0])

        for j, n_bonds in enumerate(n_bonds_list):
            win_atoms = n_bonds + 1
            if L_atoms < win_atoms:
                continue
            for i in range(L_atoms - win_atoms + 1):
                _, _, x1, y1, z1 = atoms_sorted[i]
                _, _, x2, y2, z2 = atoms_sorted[i + n_bonds]
                dx, dy, dz = x2-x1, y2-y1, z2-z1
                R2 = dx*dx + dy*dy + dz*dz
                sums_R2[j] += R2
                counts[j]  += 1

    # rms = sqrt(mean(R^2))
    rms_vals = [math.sqrt(sums_R2[j]/counts[j]) if counts[j] > 0 else float('nan')
                for j in range(len(n_bonds_list))]
    return rms_vals, counts

# ---------- write R(n)/Req(n) ----------
def writeStretchingRatioPerTimestep(
    timesteps,
    outFile,
    Ne=28,
    n_multipliers=(0.2, 0.5, 1, 2, 5, 10),
    rounding="nearest",
    chain_length_range=None  # optional (min_atoms, max_atoms)
):
    """
    For each timestep, compute R(n)/Req(n), where:
      R(n)   = sqrt( mean_over_windows( |r(i+n)-r(i)|^2 ) ) at that timestep
      Req(n) = the same quantity evaluated at the **first** timestep
    """
    # set n values
    n_bonds_list = [_round_n(m*Ne, rounding) for m in n_multipliers]

    # establish baseline Req(n) from the earliest timestep
    t0 = sorted(timesteps.keys())[0]
    Req_n, base_counts = _rms_R_of_n_for_timestep(
        timesteps[t0], n_bonds_list, chain_length_range
    )

    with open(outFile, "w") as f:
        ratio_cols = [f"R_over_Req(n={m}Ne)" for m in n_multipliers]
        nb_cols    = [f"n_bonds(n={m}Ne)"      for m in n_multipliers]
        req_cols   = [f"Req_rms(n={m}Ne)"      for m in n_multipliers]
        f.write("timestep, " + ", ".join(ratio_cols + nb_cols + req_cols) + "\n")

        for t in sorted(timesteps.keys()):
            Rn_rms, counts = _rms_R_of_n_for_timestep(
                timesteps[t], n_bonds_list, chain_length_range
            )

            # ratios relative to time-zero rms
            ratios = []
            for j, Rrms in enumerate(Rn_rms):
                denom = Req_n[j]
                if (denom is None) or (denom != denom) or denom == 0.0:  # NaN or zero
                    ratios.append(float('nan'))
                else:
                    ratios.append(Rrms / denom)

            row = (
                [str(t)] +
                [f"{v:.8g}" for v in ratios] +
                [str(n) for n in n_bonds_list] +
                [f"{v:.8g}" for v in Req_n]
            )
            f.write(", ".join(row) + "\n")

# --- Example usage ---
result = readTimesteps("dump.lammps-frame.lammpstrj", timestepsCount=1000)
writeStretchingRatioPerTimestep(
    result,
    outFile="R_over_Req_vs_timestep.csv",
    Ne=28,
    n_multipliers=(0.2, 0.5, 1, 2, 6, 12),
    rounding="nearest",
    chain_length_range=(345, 375)  # adjust as needed
)
