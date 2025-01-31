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

