import os
import numpy as np
from frictiontools.constants import boltzmann_kB

def gaussian_function(x,x0,s):
    return 1./np.sqrt(2*np.pi*s*s)*np.exp(-0.5*(((x-x0) / (s))**2))

def fermi_pop(x, x0, T):
    """
    Compute the Fermi population based on the given parameters.

    Parameters:
        x (float): Energy value.
        x0 (float): Fermi level.
        T (float): Temperature.
        n_spin (int): Number of spins.

    Returns:
        float: Fermi population.
    """


    if T < np.finfo(float).tiny:  # Check if temperature is near zero
        if x < x0:
            return 1.0  # Fermi population for T = 0
        else:
            return 0.0
    else:
        fermi_exponent = (x - x0) / (boltzmann_kB * T)

        if fermi_exponent > 100.0:  # Prevent floating-point overflow for large exponents
            return 0.0
        else:
            fermi_denominator = np.exp(fermi_exponent) + 1
            return 1 / fermi_denominator
        
def fermi_derivative(energy, chem_pot, temperature):
    """
    Calculate the derivative of the Fermi-Dirac distribution with respect to energy.

    Args:
        energy (float or ndarray): Energy value(s) in eV.
        chem_pot (float): Chemical potential (Fermi energy) in eV.
        temperature (float): Temperature in Kelvin.

    Returns:
        float or ndarray: The derivative of the Fermi-Dirac distribution.
    """
    # Boltzmann constant in eV/K
    boltzmann_kB = 8.617333262145e-5  # Boltzmann constant in eV/K

    # Fermi-Dirac distribution
    f = 1.0 / (np.exp((energy - chem_pot) / (boltzmann_kB * temperature)) + 1.0)

    # Derivative of Fermi-Dirac distribution
    f_derivative = -f * (1 - f) / (boltzmann_kB * temperature)
    
    return f_derivative

def print_system_parameters(n_k_points, k_weights, perturbing_energies, friction_masses, friction_indices, friction_symbols, n_atoms,
                            ndims, sigma, temperature, chem_pot, friction_max_energy, friction_aimsout, normal_mode_filename):
    """
    Nicely print out the system parameters.

    Args:
        n_k_points (int): Number of k-points.
        k_weights (ndarray): k-point weights.
        perturbing_energies (ndarray): Perturbing energies for the system.
        friction_masses (ndarray): Friction masses for each atom.
        friction_indices (list): Indices of atoms involved in the friction calculation.
        n_atoms (int): Total number of atoms in the system.
        ndims (int): Total number of dimensions (3 * number of friction atoms).
        sigma (float): Broadening parameter (eV).
        temperature (float): Temperature (K).
        friction_max_energy (float): Maximum energy considered for friction (eV).
    """
    # Resolve absolute paths
    abs_friction_aimsout = os.path.abspath(friction_aimsout)
    abs_normal_mode_filename = os.path.abspath(normal_mode_filename)

    print("===================")
    print("\nFile Paths:")
    print(f"Aims output file: {abs_friction_aimsout}")
    print(f"Normal mode file: {abs_normal_mode_filename}")
    print(f"\n")

    print("System Parameters:")
    print(f"Number of k-points: {n_k_points}")
    print(f"k-point weights: {k_weights}")
    print(f"Perturbing energies: {perturbing_energies}")
    print(f"Friction masses: {friction_masses}")
    print(f"Friction indices (atoms involved): {friction_indices}")
    print(f"Friction symbols (atoms involved): {friction_symbols}")
    print(f"Number of atoms in system: {n_atoms}")
    print(f"Dimensions of friction tensor: {ndims}")
    print(f"Sigma (Broadening parameter): {sigma} eV")
    print(f"Temperature: {temperature} K")
    print(f"Maximum energy for friction calculation: {friction_max_energy} eV")
    

    print(f"Total number of friction atoms: {len(friction_masses)}")
    print(f"Chemical potential: {chem_pot} eV") 
    print(f"Energy window for friction calculations: {chem_pot - 2 * friction_max_energy} eV to {chem_pot + 2 * friction_max_energy} eV")
    print("\n===================")