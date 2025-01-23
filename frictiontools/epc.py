import numpy as np
from scipy.integrate import simps  # For numerical integration
from frictiontools.constants import *

def calculate_eliashberg(vibrational_linewidths, perturbing_energies):
    """
    Calculate Eliashberg spectral function, electron-phonon coupling strength, and Tc.

    Args:
        vibrational_linewidths (ndarray): Vibrational linewidths (Gamma) of shape (n_modes) in eV.
        perturbing_energies (ndarray): Corresponding energies (hbar omega) of shape (n_modes) in eV.

    Returns:
        tuple: (eliashberg_function, lambda_, Tc)
            - eliashberg_function (ndarray): The α²F(ω) array of shape (n_modes).
            - lambda_ (float): Electron-phonon coupling strength.
            - Tc (float): Superconducting critical temperature in Kelvin.
    """
    # Convert energies to frequencies in Hz
    omega = perturbing_energies / hbar  # Frequency in Hz

    # Calculate the Eliashberg spectral function α²F(ω)
    eliashberg_function = (vibrational_linewidths / (np.pi * omega)) * perturbing_energies

    # Calculate λ (electron-phonon coupling strength)
    lambda_ = 2 * simps(eliashberg_function / perturbing_energies, x=perturbing_energies)

    # Use the McMillan formula to estimate Tc
    mu_star = 0.1  # Coulomb pseudopotential (common value)
    omega_log = np.exp(np.sum(np.log(perturbing_energies) * eliashberg_function) / np.sum(eliashberg_function))
    Tc = (hbar / boltzmann_kB) * (omega_log / 1.2) * np.exp(-1.04 * (1 + lambda_) / (lambda_ - mu_star * (1 + 0.62 * lambda_)))

    return eliashberg_function, lambda_, Tc