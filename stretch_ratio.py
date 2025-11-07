import math

# --- your reader, unchanged ---
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

# --- helper ---
def _round_n(x, mode="nearest"):
    if mode == "floor":
        return max(1, math.floor(x))
    if mode == "ceil":
        return max(1, math.ceil(x))
    return max(1, int(round(x)))  # nearest

# --- stretching ratio with constant b ---
def writeStretchingRatioPerTimestep(
    timesteps,
    outFile,
    Ne=28,
    b=0.96,                         # <-- constant bond length
    n_multipliers=(0.2, 0.5, 1, 2, 5, 10),
    rounding="nearest",
    chain_length_range=None         # optional (min_atoms, max_atoms)
):
    """
    For each timestep, compute mean R(n)/(n*b) across all chains and all
    sliding windows of length n bonds, for n in n_multipliers*Ne (rounded).
    Output columns: timestep, one value per n, then the integer n used.
    """
    n_bonds_list = [_round_n(m*Ne, rounding) for m in n_multipliers]

    with open(outFile, "w") as f:
        ratio_cols = [f"R_over_nb(n={m}Ne)" for m in n_multipliers]
        nb_cols    = [f"n_bonds(n={m}Ne)"  for m in n_multipliers]
        f.write("timestep, " + ", ".join(ratio_cols + nb_cols) + "\n")

        for t in sorted(timesteps.keys()):
            sums   = [0.0 for _ in n_bonds_list]
            counts = [0   for _ in n_bonds_list]

            for mol, mdata in timesteps[t]["mols"].items():
                L_atoms = mdata["length"]
                if chain_length_range and len(chain_length_range) == 2:
                    mn, mx = chain_length_range
                    if not (mn <= L_atoms <= mx):
                        continue

                atoms_sorted = sorted(mdata["atoms"], key=lambda a: a[0])

                for idx, n_bonds in enumerate(n_bonds_list):
                    win_atoms = n_bonds + 1
                    if L_atoms < win_atoms:
                        continue
                    # slide a window of n_bonds along the chain
                    for i in range(L_atoms - win_atoms + 1):
                        _, _, x1, y1, z1 = atoms_sorted[i]
                        _, _, x2, y2, z2 = atoms_sorted[i + n_bonds]
                        dx, dy, dz = x2-x1, y2-y1, z2-z1
                        Ree = math.sqrt(dx*dx + dy*dy + dz*dz)
                        ratio = Ree / (n_bonds * b)
                        sums[idx] += ratio
                        counts[idx] += 1

            means = [(sums[i]/counts[i]) if counts[i] > 0 else float('nan')
                     for i in range(len(n_bonds_list))]
            row = [str(t)] + [str(v) for v in means] + [str(n) for n in n_bonds_list]
            f.write(", ".join(row) + "\n")

#--- Example usage ---
result = readTimesteps("dump.lammps-frame.lammpstrj", timestepsCount=1000)
writeStretchingRatioPerTimestep(
    result,
    outFile="stretching_ratio_vs_timestep.csv",
    Ne=28,
    b=0.96,                         # constant bond length
    n_multipliers=(0.2, 0.5, 1, 2, 6, 12),
    rounding="nearest",
    chain_length_range=(350, 370)   # optional
)