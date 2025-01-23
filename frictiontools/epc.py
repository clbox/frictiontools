import numpy as np
from scipy.integrate import simps  # For numerical integration
from frictiontools.constants import *
from frictiontools.utils import gaussian_function
import matplotlib.pyplot as plt

def calculate_epc_strength_mode(linewidths, dos_fermi, perturbing_energies, n_spin=1):
    """
    Calculate the electron-phonon coupling strength (lambda) for each mode and total lambda.
    
    Args:
        linewidths (ndarray): Array of shape (n_modes) containing phonon linewidths in eV.
        dos_fermi (float): Density of states at the Fermi level in states/eV.
        perturbing_energies (ndarray): Array of shape (n_modes) containing phonon energies in eV.
        
    Returns:
        tuple:
            - ndarray: Mode-resolved electron-phonon coupling strength (lambda) for each mode.
            - float: Total electron-phonon coupling strength (lambda).
    """
    # Ensure inputs are arrays
    linewidths = np.asarray(linewidths)
    perturbing_energies = np.asarray(perturbing_energies)

    # Avoid division by zero for zero-energy phonons
    nonzero_mask = perturbing_energies > 0
    linewidths = linewidths[nonzero_mask]
    perturbing_energies = perturbing_energies[nonzero_mask]

    non_zero_frequencies = perturbing_energies / hbar

    # Calculate mode-resolved lambda
    #lambda_modes = ((n_spin/2.0) / (dos_fermi * perturbing_energies)) * linewidths
    lambda_modes = ((n_spin/2.0) / (np.pi * dos_fermi)) * (linewidths/non_zero_frequencies**2)

    # Calculate total lambda
    lambda_total = np.sum(lambda_modes)

    return lambda_modes, lambda_total

def calculate_alpha2_F_gaussian(omega_values, linewidths, perturbing_energies, dos_fermi, n_q_points, sigma):
    """
    Calculate the α²F(ω) using a Gaussian approximation for the delta function.

    Args:
        omega_values (ndarray): Array of ω values (in eV) at which to evaluate α²F(ω).
        linewidths (ndarray): Vibrational linewidths (γ_q) in eV, shape (n_modes,).
        perturbing_energies (ndarray): Phonon perturbing energies (ℏω_q) in eV, shape (n_modes,).
        dos_fermi (float): Density of states at the Fermi level (N(0)) in states/eV.
        n_q_points (int): Number of q-points (N_q) used in the calculation.
        sigma (float): Broadening width for the Gaussian approximation (in eV).
        hbar (float): Planck's constant (default is in eV·s).
        
    Returns:
        ndarray: The value of α²F(ω) for each ω in omega_values.
    """
    frequencies = perturbing_energies / hbar  # Convert perturbing energies to frequencies
    alpha2_F_values = np.zeros_like(omega_values)
    
    for i, omega in enumerate(omega_values):
        alpha2_F_omega = 0.0
        for j in range(len(frequencies)):
            # Use Gaussian function to approximate the delta function
            alpha2_F_omega += (linewidths[j] / frequencies[j]) * gaussian_function(omega, frequencies[j], sigma)
        
        # Normalize the result
        alpha2_F_values[i] = (1 / (2 * np.pi * dos_fermi * n_q_points)) * alpha2_F_omega
    
    return alpha2_F_values


def plot_eliashberg_function(omega_values, alpha2_F_values, output_file="eliashberg_function.pdf"):
    """
    Plot the Eliashberg spectral function α²F(ω).

    Args:
        omega_values (ndarray): Array of frequencies (ω) in eV.
        alpha2_F_values (ndarray): Corresponding values of α²F(ω).
        output_file (str): File name for saving the plot (default: "eliashberg_function.pdf").
    """
    plt.figure(figsize=(8, 6))
    plt.plot(omega_values, alpha2_F_values, label=r"$\alpha^2 F(\omega)$", color='blue', linewidth=2)
    plt.xlabel(r'$\hbar\omega$ (eV)', fontsize=14)
    plt.ylabel(r'$\alpha^2 F(\omega)$', fontsize=14)
    plt.title(r'Eliashberg Spectral Function $\alpha^2 F(\omega)$', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot as a PDF
    plt.savefig(output_file)
    print(f"Eliashberg function plot saved as: {output_file}")
    
    # Show the plot
