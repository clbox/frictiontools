import numpy as np
import matplotlib.pyplot as plt


def calculate_dos(eigenvalues, chemical_potential, k_weights, num_bins=500, energy_range=None, sigma=0.01):
    """
    Calculate the Density of States (DOS) from eigenvalues with k-weights.

    Args:
        eigenvalues (ndarray): Array of shape (n_states, n_k_points) with eigenvalues in eV.
        k_weights (ndarray): Array of shape (n_k_points,) with weights for each k-point.
        num_bins (int): Number of bins for DOS calculation.
        energy_range (tuple): Energy range (min, max) in eV for the DOS.
        sigma (float): Broadening parameter (Gaussian smearing) in eV.

    Returns:
        energies (ndarray): Energy values for DOS.
        dos (ndarray): Calculated DOS values.
    """
    n_k_points, n_states = eigenvalues.shape
    eigenvalues = eigenvalues.flatten()  # Flatten states and k-points into one array
    k_weights = np.repeat(k_weights, n_states)  # Match k-weights to eigenvalues

    if energy_range is None:
        energy_min, energy_max = eigenvalues.min(), eigenvalues.max()
    else:
        energy_min, energy_max = energy_range

    energies = np.linspace(energy_min, energy_max, num_bins)
    dos = np.zeros_like(energies)

    # Calculate DOS with Gaussian smearing and k-weights
    for e, w in zip(eigenvalues, k_weights):
        dos += w * np.exp(-((energies - e) ** 2) / (2 * sigma ** 2))
    dos /= np.sqrt(2 * np.pi) * sigma

    return energies, dos

def save_dos(filename, energies, dos):
    """
    Save the DOS to a file.

    Args:
        filename (str): Output file name.
        energies (ndarray): Energy values.
        dos (ndarray): DOS values.
    """
    with open(filename, "w") as file:
        file.write("# Energy (eV)   DOS\n")
        for e, d in zip(energies, dos):
            file.write(f"{e:.6f}   {d:.6e}\n")

def plot_dos(energies, dos, chemical_potential, output_image="dos.pdf"):
    """
    Plot the DOS.

    Args:
        energies (ndarray): Energy values.
        dos (ndarray): DOS values.
        output_image (str): Filename to save the plot (optional).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(energies, dos, label="DOS", color="blue")
    plt.axvline(chemical_potential, color="black", linestyle="--", label="Fermi level")
    plt.xlabel("Energy (eV)", fontsize=14)
    plt.ylabel("DOS", fontsize=14)
    plt.title("Density of States", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)

def output_dos(eigenvalues, chemical_potential, k_weights, num_bins=500, energy_range=None, sigma=0.01 ):

    energies, dos = calculate_dos(eigenvalues, chemical_potential, k_weights, num_bins, energy_range, sigma)

    save_dos("dos.txt", energies, dos)

    plot_dos(energies, dos, chemical_potential, "dos.pdf")

    return energies, dos

def calculate_dos_fermi(eigenvalues, k_weights, fermi_energy, sigma=0.01):
    """
    Calculate the density of states (DOS) at the Fermi level.
    
    Args:
        eigenvalues (ndarray): Array of shape (n_k_points, n_states) containing eigenvalues in eV.
        k_weights (ndarray): Array of shape (n_k_points,) containing the k-point weights.
        fermi_energy (float): Fermi level energy in eV.
        sigma (float): Gaussian smearing parameter in eV (default: 0.01 eV).
        
    Returns:
        float: The density of states at the Fermi level (in states/eV).
    """
    n_k_points, n_states = eigenvalues.shape
    dos = 0.0
    
    # Iterate over k-points
    for i_k in range(n_k_points):
        # Iterate over states
        for i_state in range(n_states):
            # Gaussian smearing function
            delta = np.exp(-((eigenvalues[i_k, i_state] - fermi_energy) ** 2) / (2 * sigma ** 2))
            delta /= np.sqrt(2 * np.pi) * sigma
            
            # Weighted contribution to DOS
            dos += k_weights[i_k] * delta
    
    return dos