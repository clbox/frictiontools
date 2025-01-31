import numpy as np
from frictiontools.constants import *
from frictiontools.utils import *
from scipy.special import hermite, factorial

def find_evs_sensible_bounds(evs,chem_pot,max_energy_from_fermi):
    # We find indices for upper and lower bounds of eigenvalues
    # to reduce size of problem
    min_bounds = np.zeros((np.shape(evs)[0]),dtype=int)
    max_bounds = np.zeros((np.shape(evs)[0]),dtype=int)


    for i_k_point in range(np.shape(evs)[0]):
        for i,ev in enumerate(evs[i_k_point,:]):
            if ev < (chem_pot-max_energy_from_fermi):
                min_bounds[i_k_point]= i
                continue
            elif ev > (chem_pot+max_energy_from_fermi):
                max_bounds[i_k_point]= i
                break
    return min_bounds,max_bounds

def delta_function_gaussian(x, x0, s):
    """
    Compute the Gaussian delta function.
    
    Parameters:
        x (float or np.ndarray): Input value(s).
        x0 (float): Mean or center value.
        s (float): Standard deviation.
        
    Returns:
        float or np.ndarray: Result of the Gaussian delta function.
    """
    sqrt_pi = np.sqrt(np.pi)
    one_over_sqrt2 = 1 / np.sqrt(2)
    gaussian = (np.exp(-0.5 * ((x - x0) ** 2) / (s ** 2)) / (s * sqrt_pi)) * one_over_sqrt2
    return gaussian

def delta_function_methfessel_paxton(x, x0, sigma, order=1):
    """
    Compute the Methfessel-Paxton smearing function.

    Parameters:
    - epsilon: Energy difference from the Fermi level (array-like)
    - sigma: Smearing width (float, in eV)
    - order: Order of the Methfessel-Paxton expansion (int)

    Returns:
    - f: Smearing function values
    """
    epsilon = x-x0

    eta = epsilon / sigma  # Scaled energy
    gauss = np.exp(-eta**2) / np.sqrt(np.pi)  # Gaussian core function

    # Initialize smearing function
    f = 0.5 * (1 + np.erf(eta))  # Zeroth order: Gaussian smearing

    # Add higher order corrections
    for n in range(1, order + 1):
        Hn = hermite(n)(eta)  # Hermite polynomial H_n(eta)
        term = (-1) ** n * Hn * gauss / (2 ** n * factorial(n))
        f += term

    return f


def delta_function(epsilon, x0, sigma, order=1, type="gaussian"):

    if type=="gaussian":
        result = delta_function_gaussian(epsilon, x0, sigma)
    elif type=="methfessel_paxton":
        result = delta_function_methfessel_paxton(epsilon, x0, sigma, order)
    else:
        print("unknown type of delta function")
        

    return result

def find_max_occupied_state(evs, chem_pot, temperature, tolerance=1e-3):
    n_states = len(evs)
    max_occupied_state = n_states
    for i in range(n_states):
        if (fermi_pop(evs[i], chem_pot, temperature) >=tolerance):
            max_occupied_state = i
    return max_occupied_state

def find_min_unoccupied_state(evs, chem_pot, temperature,  tolerance=1e-3):
    n_states = len(evs)
    min_unoccupied_state = 0
    for i in range(n_states):
        if (fermi_pop(evs[i], chem_pot, temperature)>=(1-tolerance)):
            min_unoccupied_state = i
    return min_unoccupied_state

def find_bounded_states(evs, chem_pot, energy_cutoff):
    """
    Find the minimum and maximum state indices that lie within a specified energy window around the chemical potential.

    Args:
        evs (ndarray): Energy eigenvalues of the system.
        chem_pot (float): The chemical potential.
        energy_cutoff (float): The energy cutoff for the window around the chemical potential.

    Returns:
        tuple: Indices of the minimum and maximum states within the energy window.
    """
    n_states = len(evs)
    # Define the energy window bounds
    lower_bound = chem_pot - 2 * energy_cutoff
    upper_bound = chem_pot + 2 * energy_cutoff

    # Initialize the minimum and maximum state indices to None
    minimum_state = 0
    maximum_state = 0

    # # Iterate over the energy eigenvalues to find the states within the window
    # for i, ev in enumerate(evs):
    #     if ev >= lower_bound and ev <= upper_bound:
    #         if minimum_state is None or ev < evs[minimum_state]:
    #             minimum_state = i
    #         if maximum_state is None or ev > evs[maximum_state]:
    #             maximum_state = i

    for i, ev in enumerate(evs):
        if  ev <= lower_bound:
            minimum_state = i
    
    for i, ev in enumerate(evs):
        if  ev <= upper_bound:
            maximum_state = i

    return minimum_state, maximum_state

def evaluate_fermi_factor(ei, ej, chem_pot, temperature, perturbing_energies, mode="diff"):

    if mode=="diff":
        epsilon = ej - ei
        fermi_factor = (fermi_pop(ei, chem_pot, temperature) - fermi_pop(ej, chem_pot, temperature))
        fermi_factor = fermi_factor/epsilon
        return fermi_factor
    elif mode =="derivative":
        fermi_factor = -1.0 * fermi_derivative(ei, chem_pot, temperature)
        return fermi_factor
    else:
        print("Unrecognised fermi factor mode")