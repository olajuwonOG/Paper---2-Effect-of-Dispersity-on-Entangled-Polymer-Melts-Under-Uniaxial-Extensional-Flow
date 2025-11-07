import csv
import numpy as np
import matplotlib.pyplot as plt
import time

# Increase the figure DPI for better resolution
plt.rcParams["figure.dpi"] = 400
#fig, ax = plt.subplots()

# fig, ax = plt.subplots(figsize=(4, 4))
# plt.rcParams["figure.dpi"] = 250

plt.rcParams['font.family'] = 'Times New Roman'
start_time = time.time()

# List of CSV files for different polymers along with their dispersity values


csv_files = [
    ('Mov_Ave_1.0_360_normal_new_R1_k=1.5.csv', 1.0),
    ('Mov_Ave_1.4_360_normal_new_R1_k=1.5.csv', 1.4),
    ('Mov_Ave_2.0_360_normal_new_R1_k=1.5.csv', 2.0)

]


# Define a range of frequencies (adjust as needed)
frequencies = np.logspace(-7, 0, 1000)  # Logarithmic frequency range

# Constants (adjust as needed)
k_B = 1  # Boltzmann constant
T = 1    # Temperature in Kelvin
V = 1    # Volume in m^3

# Define markers for G' for each polymer
markers_G_prime = ['s', 'o', '^']  # Square, Diamond, Triangle Up
# Define markers for G'' for each polymer
markers_G_double_prime = ['s', 'o', '^']  # Circle, Triangle Down, X

# Define colors for each polymer
colors = ['purple', 'orange', 'blue']  # Red, Green, Blue

# Define the interval for plotting (e.g., every 10th point)
plot_interval = 20

# Initialize the plot
plt.figure(figsize=(12, 8))

# Iterate over each polymer's CSV file
for idx, (csv_file_path, dispersity) in enumerate(csv_files):
    # Initialize lists to store the data
    stress = []
    time_multi_Tau = []

    # Open the CSV file and read the data
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header if present
        for row in csvreader:
            if len(row) < 2:
                continue  # Skip incomplete rows
            try:
                time_multi_Tau.append(float(row[0]))
                stress.append(float(row[1]))
            except ValueError:
                continue  # Skip rows with non-numeric data

    # Convert to numpy arrays
    t = np.array(time_multi_Tau)
    G = np.array(stress)

    # Initialize arrays to store G' and G''
    G_prime = np.zeros_like(frequencies)
    G_double_prime = np.zeros_like(frequencies)

    # Calculate G' and G'' for each frequency
    for i, omega in enumerate(frequencies):
        G_complex = 0.0 + 0.0j  # Initialize the complex modulus as zero

        for k in range(1, len(time_multi_Tau)):
            delta_t = time_multi_Tau[k] - time_multi_Tau[k-1]
            if delta_t == 0:
                continue  # Avoid division by zero

            term1 = (np.exp(-1j * omega * delta_t) - 1) / (omega**2 * delta_t)
            term2 = np.exp(-1j * omega * delta_t) / (1j * omega)

            integrand = (
                G[k] * (term1 - term2) - G[k-1] * (term1 - 1 / (1j * omega))
            )

            # Multiply the integrand by i * omega and accumulate
            G_complex += np.exp(-1j * omega * time_multi_Tau[k-1]) * (1j * omega) * integrand

        # Multiply the result by V / (k_B * T) to get the complex modulus
        G_complex *= V / (k_B * T)

        # Store the real and imaginary parts
        G_prime[i] = G_complex.real
        G_double_prime[i] = G_complex.imag

    # Write the results to a CSV file
    output_csv_file = csv_file_path.replace('.csv', '_moduli.csv')
    with open(output_csv_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Frequency (rad/s)', "G' (Pa)", "G'' (Pa)"])
        for freq, g_prime, g_double_prime in zip(frequencies, G_prime, G_double_prime):
            writer.writerow([freq, g_prime, g_double_prime])

    # Select points at the specified interval for plotting
    freq_plot = frequencies[::plot_interval]
    Gp_plot = G_prime[::plot_interval]
    Gpp_plot = G_double_prime[::plot_interval]

    # Plot G' with unique markers for each polymer
    plt.scatter(
        freq_plot,
        Gp_plot,
        s=250,  # Increased size for better visibility
        marker=markers_G_prime[idx % len(markers_G_prime)],
        facecolors='none', linewidth =4.0, # Hollow markers
        edgecolors=colors[idx % len(colors)],
        label=f"$G'(\omega)$ \u0110 = {dispersity} " #label=f"$G'(\omega)$ \u0110 = {dispersity}
    )

    # Plot G'' with distinct markers
    plt.scatter(
        freq_plot,
        Gpp_plot,
        s=250, #60
        linewidth =4.0, 
        marker=markers_G_double_prime[idx % len(markers_G_double_prime)],
        facecolors=colors[idx % len(colors)],  # Filled markers
        edgecolors=colors[idx % len(colors)],
        label=f"$G''(\omega)$ \u0110 = {dispersity} "
    )

# Customize the plot
plt.ylabel(r"$\mathregular{G'}, \mathregular{G''}$", fontsize=80, fontname='Times New Roman')
plt.xlabel(r"$\mathregular{\omega}$", fontsize=80, fontname='Times New Roman')

# Ensure fontsize of tick labels is set

plt.xscale("log")
plt.yscale("log")
plt.ylim(0.0001, 0.13)
plt.xlim(0.0000001, 0.01)

#plt.legend(fontsize=26, loc='lower right', frameon=True)
plt.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=6, width=1, labelsize=24, pad = 14)
plt.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=3, width=1)
plt.minorticks_on()

#plt.legend(loc='lower right', prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 26},frameon=False, bbox_to_anchor=(1, -0.00))
#plt.legend([], [], frameon=False)
# Adjust layout for better spacing
#plt.tight_layout()
plt.xticks(fontsize=50, fontname='Times New Roman')
plt.yticks(fontsize=50, fontname='Times New Roman')
plt.rcParams["axes.linewidth"] = 5

# Save and display the plot
plt.savefig('Moduli_Frequency_Dependence.png', dpi=600)
plt.show()
end_time = time.time()
#print(f"Total execution time: {end_time - start_time:.2f} seconds")