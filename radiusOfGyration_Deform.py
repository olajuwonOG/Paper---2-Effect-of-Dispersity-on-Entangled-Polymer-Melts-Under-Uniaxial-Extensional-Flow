
import math
import csv

def readTimesteps(inputFile, timestepsCount=None):
    if timestepsCount:
        counter = 0
    else:
        timestepsCount = 5  # Default count if not specified
    timesteps = {}  # Created an empty dictionary
    with open(inputFile) as f:
        line = f.readline()
        while line != '':
            if line.split()[0] == "ITEM:":
                timestep = int(f.readline().split()[0])  # Timestep number
                timesteps[timestep] = {}
                filler = f.readline()
                atoms = int(f.readline().split()[0])  # Number of atoms
                timesteps[timestep]["atoms"] = atoms
                filler = f.readline()
                x = f.readline()  # Box dimension in x direction
                x_pos = float(x.split()[1])
                x_neg = float(x.split()[0])
                timesteps[timestep]["-x"] = x_neg
                timesteps[timestep]["x"] = x_pos
                y = f.readline()
                y_pos = float(y.split()[1])
                y_neg = float(y.split()[0])
                timesteps[timestep]["-y"] = y_neg
                timesteps[timestep]["y"] = y_pos
                z = f.readline()
                z_pos = float(z.split()[1])
                z_neg = float(z.split()[0])
                timesteps[timestep]["-z"] = z_neg
                timesteps[timestep]["z"] = z_pos
                headers = f.readline()
                timesteps[timestep]["headers"] = headers
                timesteps[timestep]["mols"] = {}
                for i in range(atoms):
                    line = f.readline()
                    id = int(line.split()[0])
                    mol = int(line.split()[1])
                    type = int(line.split()[2])
                    #q = int(line.split()[3])
                    xs = float(line.split()[3]) * (timesteps[timestep]['x'] - timesteps[timestep]['-x'])
                    ys = float(line.split()[4]) * (timesteps[timestep]['y'] - timesteps[timestep]['-y'])
                    zs = float(line.split()[5]) * (timesteps[timestep]['z'] - timesteps[timestep]['-z'])
                    ix = int(line.split()[6])
                    iy = int(line.split()[7])
                    iz = int(line.split()[8])
                    xbox = (timesteps[timestep]['x'] - timesteps[timestep]['-x'])
                    ybox = (timesteps[timestep]['y'] - timesteps[timestep]['-y'])
                    zbox = (timesteps[timestep]['z'] - timesteps[timestep]['-z'])
                    if xs > xbox:
                        xs -= xbox
                    elif xs < 0:
                        xs += xbox
                    if ys > ybox:
                        ys -= ybox
                    elif ys < 0:
                        ys += ybox
                    if zs > zbox:
                        zs -= zbox
                    elif zs < 0:
                        zs += zbox
                    xs = xbox * float(line.split()[3]) + ix * xbox
                    ys = ybox * float(line.split()[4]) + iy * ybox
                    zs = zbox * float(line.split()[5]) + iz * zbox
                    if mol in timesteps[timestep]["mols"]:
                        timesteps[timestep]["mols"][mol]["atoms"].append((id, type, xs, ys, zs, ix, iy, iz))
                    else:
                        timesteps[timestep]["mols"][mol] = {"atoms": [(id, type, xs, ys, zs, ix, iy, iz)]}
                for mol in timesteps[timestep]["mols"]:
                    x = y = z = 0
                    length = len(timesteps[timestep]["mols"][mol]["atoms"])
                    timesteps[timestep]["mols"][mol]["length"] = length
                    for atom in timesteps[timestep]["mols"][mol]["atoms"]:
                        x += atom[2] / length
                        y += atom[3] / length
                        z += atom[4] / length
                    timesteps[timestep]["mols"][mol]["xcm"] = x
                    timesteps[timestep]["mols"][mol]["ycm"] = y
                    timesteps[timestep]["mols"][mol]["zcm"] = z
            line = f.readline()
            counter += 1
            if counter == timestepsCount:
                return timesteps
    return timesteps

def Rg2Calculator(timesteps):
    for timestep in timesteps:
        for mol in timesteps[timestep]["mols"]:
            length = len(timesteps[timestep]["mols"][mol]["atoms"])
            rg2_x, rg2_y, rg2_z = 0, 0, 0
            
            for atom in timesteps[timestep]["mols"][mol]["atoms"]:
                x = atom[2] - timesteps[timestep]["mols"][mol]["xcm"]
                y = atom[3] - timesteps[timestep]["mols"][mol]["ycm"]
                z = atom[4] - timesteps[timestep]["mols"][mol]["zcm"]
                
                rg2_x += (x ** 2)
                rg2_y += (y ** 2)
                rg2_z += (z ** 2)
            
            rg2_x /= length
            rg2_y /= length
            rg2_z /= length

            timesteps[timestep]["mols"][mol]["rg2_x"] = rg2_x
            timesteps[timestep]["mols"][mol]["rg2_y"] = rg2_y
            timesteps[timestep]["mols"][mol]["rg2_z"] = rg2_z
    return timesteps


#z - direction
def writeRg2AverageToFile(timesteps, outFile, frames=1, min_length=None, max_length=None):
    with open(outFile, 'w') as f:
        f.write("chain_ID, rg2_x, rg2_y, rg2_z, Rg_parallel_z, Rg_perpendicular, Rg_total, chain_length\n")
        mols = {}
        frameCount = 0

        for timestep in timesteps:
            if frameCount < frames:
                for mol in timesteps[timestep]["mols"]:
                    length = len(timesteps[timestep]["mols"][mol]["atoms"])
                    
                    if (min_length is not None and length < min_length) or (max_length is not None and length > max_length):
                        continue

                    if mol not in mols:
                        mols[mol] = {
                            'rg2_x_sum': timesteps[timestep]["mols"][mol]["rg2_x"],
                            'rg2_y_sum': timesteps[timestep]["mols"][mol]["rg2_y"],
                            'rg2_z_sum': timesteps[timestep]["mols"][mol]["rg2_z"],
                            'length': length
                        }
                    else:
                        mols[mol]['rg2_x_sum'] += timesteps[timestep]["mols"][mol]["rg2_x"]
                        mols[mol]['rg2_y_sum'] += timesteps[timestep]["mols"][mol]["rg2_y"]
                        mols[mol]['rg2_z_sum'] += timesteps[timestep]["mols"][mol]["rg2_z"]

                frameCount += 1

        for mol in mols:
            rg2_x_avg = mols[mol]['rg2_x_sum'] / frames
            rg2_y_avg = mols[mol]['rg2_y_sum'] / frames
            rg2_z_avg = mols[mol]['rg2_z_sum'] / frames
            rg_parallel_z_avg = math.sqrt(rg2_z_avg)  # Parallel direction changed to z
            rg_perpendicular_avg = math.sqrt((rg2_x_avg + rg2_y_avg) / 2)  # Perpendicular in x-y plane
            rg_total_avg = math.sqrt(rg2_x_avg + rg2_y_avg + rg2_z_avg)
            f.write(f"{mol}, {rg2_x_avg}, {rg2_y_avg}, {rg2_z_avg}, {rg_parallel_z_avg}, {rg_perpendicular_avg}, {rg_total_avg}, {mols[mol]['length']}\n")

        overall_avg_rg2 = rg_total_avg / len(mols)
        rg_square_root = math.sqrt(overall_avg_rg2)
        f.write('------------------AVERAGE Square radius of gyration----------------\n')
        f.write(str(overall_avg_rg2))
        f.write('\n--------------Rg------------\n')
        f.write(str(rg_square_root))
        

def writeTotalRgPerTimestep(timesteps, outFile, chain_length_range=None):
    with open(outFile, 'w') as f:
        f.write("timestep, total_rg_parallel_z, total_rg_perpendicular, total_rg\n")
        for timestep in timesteps:
            total_rg_parallel_z, total_rg_perpendicular, total_rg = 0, 0, 0
            mol_count = 0  # Initialize molecule count for normalization
            
            for mol in timesteps[timestep]["mols"]:
                length = len(timesteps[timestep]["mols"][mol]["atoms"])
                
                # Validate that chain_length_range is defined and has two elements
                if chain_length_range and len(chain_length_range) == 2:
                    min_length, max_length = chain_length_range
                    
                    # Check if the molecule's length is within the specified range
                    if min_length <= length <= max_length:
                        #rg_parallel_z = math.sqrt(timesteps[timestep]["mols"][mol]["rg2_z"])  # Parallel in z-direction
                        rg_parallel_z = (timesteps[timestep]["mols"][mol]["rg2_z"])  # Parallel in z-direction - square
                        #rg_perpendicular = math.sqrt((timesteps[timestep]["mols"][mol]["rg2_x"] + timesteps[timestep]["mols"][mol]["rg2_y"]) / 2)  # Perpendicular in x-y plane
                        rg_perpendicular = ((timesteps[timestep]["mols"][mol]["rg2_x"] + timesteps[timestep]["mols"][mol]["rg2_y"]) / 2)  # Perpendicular in x-y plane
                        #rg_total = math.sqrt(timesteps[timestep]["mols"][mol]["rg2_x"] + 
                        #                     timesteps[timestep]["mols"][mol]["rg2_y"] +
                        #                    timesteps[timestep]["mols"][mol]["rg2_z"])
                        rg_total = (timesteps[timestep]["mols"][mol]["rg2_x"] + 
                                             timesteps[timestep]["mols"][mol]["rg2_y"] +
                                             timesteps[timestep]["mols"][mol]["rg2_z"])
                        total_rg_parallel_z += rg_parallel_z
                        total_rg_perpendicular += rg_perpendicular
                        total_rg += rg_total
                        
                        mol_count += 1  # Increment molecule count
            
            # Normalize by the number of selected molecules
            if mol_count > 0:
                total_rg_parallel_z /= mol_count
                total_rg_perpendicular /= mol_count
                total_rg /= mol_count
            
            f.write(f"{timestep}, {total_rg_parallel_z}, {total_rg_perpendicular}, {total_rg}\n")



# x - direction
# def writeRg2AverageToFile(timesteps, outFile, frames=1, min_length=None, max_length=None):
#     with open(outFile, 'w') as f:
#         f.write("chain_ID, rg2_x, rg2_y, rg2_z, Rg_parallel_x, Rg_perpendicular, Rg_total, chain_length\n")
#         mols = {}
#         frameCount = 0

#         for timestep in timesteps:
#             if frameCount < frames:
#                 for mol in timesteps[timestep]["mols"]:
#                     length = len(timesteps[timestep]["mols"][mol]["atoms"])
                    
#                     if (min_length is not None and length < min_length) or (max_length is not None and length > max_length):
#                         continue

#                     if mol not in mols:
#                         mols[mol] = {
#                             'rg2_x_sum': timesteps[timestep]["mols"][mol]["rg2_x"],
#                             'rg2_y_sum': timesteps[timestep]["mols"][mol]["rg2_y"],
#                             'rg2_z_sum': timesteps[timestep]["mols"][mol]["rg2_z"],
#                             'length': length
#                         }
#                     else:
#                         mols[mol]['rg2_x_sum'] += timesteps[timestep]["mols"][mol]["rg2_x"]
#                         mols[mol]['rg2_y_sum'] += timesteps[timestep]["mols"][mol]["rg2_y"]
#                         mols[mol]['rg2_z_sum'] += timesteps[timestep]["mols"][mol]["rg2_z"]

#                 frameCount += 1

#         for mol in mols:
#             rg2_x_avg = mols[mol]['rg2_x_sum'] / frames
#             rg2_y_avg = mols[mol]['rg2_y_sum'] / frames
#             rg2_z_avg = mols[mol]['rg2_z_sum'] / frames
#             rg_parallel_x_avg = math.sqrt(rg2_x_avg)
#             #rg_perpendicular_avg = math.sqrt(rg2_y_avg + rg2_z_avg)
#             rg_perpendicular_avg = math.sqrt((rg2_y_avg + rg2_z_avg)/2)
#             rg_total_avg = math.sqrt(rg2_x_avg + rg2_y_avg + rg2_z_avg)
#             f.write(f"{mol}, {rg2_x_avg}, {rg2_y_avg}, {rg2_z_avg}, {rg_parallel_x_avg}, {rg_perpendicular_avg}, {rg_total_avg}, {mols[mol]['length']}\n")

#         #f.write("timestep, total_rg_parallel_x, total_rg_perpendicular, total_rg\n")
#         overall_avg_rg2 = rg_total_avg / len(mols)
#         rg_square_root = math.sqrt(overall_avg_rg2)
#         f.write('------------------AVERAGE Square radius of gyration----------------\n')
#         f.write(str(overall_avg_rg2))
#         f.write('\n--------------Rg------------\n')
#         f.write(str(rg_square_root))
        
        
# def writeTotalRgPerTimestep(timesteps, outFile, chain_length_range=None):
#     with open(outFile, 'w') as f:
#         f.write("timestep, total_rg_parallel_x, total_rg_perpendicular, total_rg\n")
#         for timestep in timesteps:
#             total_rg_parallel_x, total_rg_perpendicular, total_rg = 0, 0, 0
#             mol_count = 0  # Initialize molecule count for normalization
            
#             for mol in timesteps[timestep]["mols"]:
#                 length = len(timesteps[timestep]["mols"][mol]["atoms"])
                
#                 # Validate that chain_length_range is defined and has two elements
#                 if chain_length_range and len(chain_length_range) == 2:
#                     min_length, max_length = chain_length_range
                    
#                     # Check if the molecule's length is within the specified range
#                     if min_length <= length <= max_length:
#                         rg_parallel_x = math.sqrt(timesteps[timestep]["mols"][mol]["rg2_x"])
#                         rg_perpendicular = math.sqrt((timesteps[timestep]["mols"][mol]["rg2_y"] + timesteps[timestep]["mols"][mol]["rg2_z"]) / 2)
#                         rg_total = math.sqrt(timesteps[timestep]["mols"][mol]["rg2_x"] + 
#                                              timesteps[timestep]["mols"][mol]["rg2_y"] +
#                                              timesteps[timestep]["mols"][mol]["rg2_z"])
#                         total_rg_parallel_x += rg_parallel_x
#                         total_rg_perpendicular += rg_perpendicular
#                         total_rg += rg_total
                        
#                         mol_count += 1  # Increment molecule count
            
#             # Normalize by the number of selected molecules
#             if mol_count > 0:
#                 total_rg_parallel_x /= mol_count
#                 total_rg_perpendicular /= mol_count
#                 total_rg /= mol_count
            
#             f.write(f"{timestep}, {total_rg_parallel_x}, {total_rg_perpendicular}, {total_rg}\n")

# Example usage with chain length range
result = readTimesteps("production.dump", 2000)
result = Rg2Calculator(result)
writeRg2AverageToFile(result, "rg2AverageResults_SQ.csv", 2000, min_length=345, max_length=378)
writeTotalRgPerTimestep(result, "totalRgPerTimestep_SQ.csv", chain_length_range=(345, 378))