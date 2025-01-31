from dataclasses import dataclass
from frictiontools.aims_parse import *
import os

@dataclass
class FrictionParams:
    friction_aimsout : str
    friction_dirname: str
    n_spin: int
    sigma: float
    temperature: float
    friction_max_energy: float
    perturbing_energies: any
    fermi_mode: str
    expression: any
    delta_order: int
    n_jobs: int

@dataclass
class SystemProps:
    n_atoms : int
    n_friction_atoms : int 
    ndims : int
    n_k_points : int
    k_weights : any
    chem_pot : float
    evs : any
    friction_masses : any
    friction_indices : any
    friction_symbols : any

def get_system_properties(params):
    friction_aimsout = params.friction_aimsout
    friction_dirname = params.friction_dirname
    friction_masses = get_friction_masses(friction_aimsout)
    friction_indices = get_friction_indices(friction_aimsout)
    friction_symbols = extract_atomic_symbols(friction_aimsout)

    n_atoms = get_number_of_atoms(friction_aimsout)
    n_friction_atoms = len(friction_masses)
    ndims = 3 * n_friction_atoms

    n_k_points, k_weights = parse_kpoints_and_weights(friction_aimsout)
    chem_pot, evs = parse_ev_data(friction_dirname)

    system_properties = SystemProps(
        n_atoms=n_atoms,
        n_friction_atoms=n_friction_atoms,
        ndims=ndims,
        n_k_points=n_k_points,
        k_weights=k_weights,
        chem_pot=chem_pot,
        evs=evs,
        friction_masses=friction_masses,
        friction_indices=friction_indices,
        friction_symbols=friction_symbols
    )

    return system_properties



def print_friction_params(params):
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

    friction_aimsout = params.friction_aimsout
    friction_dirname = params

    n_spin = params.n_spin
    sigma = params.sigma
    temperature = params.temperature
    friction_max_energy = params.friction_max_energy
    perturbing_energies = params.perturbing_energies
    fermi_mode = params.fermi_mode
    expression = params.expression
    delta_order = params.delta_order
    n_jobs = params.n_jobs

    # Resolve absolute paths
    abs_friction_aimsout = os.path.abspath(friction_aimsout)


    print("===================")
    print("\nFile Paths:")
    print(f"Aims output file: {abs_friction_aimsout}")
    print(f"\n")

    print("Friction Parameters:")
    print(f"Sigma (Broadening parameter): {sigma} eV")
    print(f"Temperature: {temperature} K")
    print(f"Maximum energy for friction calculation: {friction_max_energy} eV")
    print("\n===================")

def print_system_properties(system_properties):
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

    n_atoms = system_properties.n_atoms
    ndims = system_properties.ndims
    n_k_points = system_properties.n_k_points
    k_weights = system_properties.k_weights
    chem_pot = system_properties.chem_pot
    friction_masses = system_properties.friction_masses
    friction_indices = system_properties.friction_indices
    friction_symbols = system_properties.friction_symbols

    print("===================")
    print("System Parameters:")
    print(f"Number of k-points: {n_k_points}")
    print(f"k-point weights: {k_weights}")
    print(f"Friction masses: {friction_masses}")
    print(f"Friction indices (atoms involved): {friction_indices}")
    print(f"Friction symbols (atoms involved): {friction_symbols}")
    print(f"Number of atoms in system: {n_atoms}")
    print(f"Dimensions of friction tensor: {ndims}")    
    print(f"Total number of friction atoms: {len(friction_masses)}")
    print(f"Chemical potential: {chem_pot} eV") 
    print("\n===================")