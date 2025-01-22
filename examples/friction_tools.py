
import numpy as np
import scipy.sparse as sp
import struct
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import frictiontools
from frictiontools.aims_parse import get_friction_masses, get_friction_indices, extract_atomic_symbols, get_number_of_atoms, parse_kpoints_and_weights, parse_ev_data
from frictiontools.vib_parse import parse_normal_modes, parse_perturbing_energies
from frictiontools.utils import print_system_parameters
from frictiontools.dos import output_dos
from frictiontools.jdos import calculate_and_plot_jdos_q0_parallel, calculate_and_plot_jdos_allq_parallel
from frictiontools.friction_tensor import * 
from frictiontools.constants import *


# Todo check bounds on looping states (base 0)
# Todo:  FHI-aims friction EPC based restart
# Todo: Detailed information on what excitations are being included

# TODO: Density of States
# TODO: Joint - Density of States - q=0
# TODO: Joint - Density of States - all q

#
if __name__ == "__main__":
    friction_dirname = "friction/"
    friction_aimsout = friction_dirname+"aims.out"
    normal_mode_filename = "vibrations/normal_modes"

    n_spin = 1 # Number of spin channels, 1 = spin none calculation
    sigma = 0.05 #eV  # Broadening width of delta function(s) in friction evaluation
    temperature = 300 #K 
    friction_max_energy = 3.2 # eV
    fermi_mode = "diff"  #Fermi factor = fi - fj / ei - ej

    friction_masses = get_friction_masses(friction_aimsout)
    friction_indices = get_friction_indices(friction_aimsout)
    friction_symbols = extract_atomic_symbols(friction_aimsout)

    n_atoms = get_number_of_atoms(friction_aimsout)
    n_friction_atoms = len(friction_masses)
    ndims = 3 * n_friction_atoms

    n_k_points, k_weights = parse_kpoints_and_weights(friction_aimsout)
    modes = parse_normal_modes(normal_mode_filename, ndims)
    chem_pot, evs = parse_ev_data(friction_dirname)

    # Get hbar omega for each mode
    #perturbing_energies = parse_perturbing_energies(normal_mode_filename)
    # perturbing_energies = np.array([0.245])
    perturbing_energies = np.linspace(-1,1,100)

    print_system_parameters(n_k_points, k_weights, perturbing_energies, friction_masses, friction_indices, friction_symbols, n_atoms, 
                            ndims, sigma, temperature, chem_pot, friction_max_energy, friction_aimsout, normal_mode_filename)

    dos_grid, dos = output_dos(evs, chem_pot, k_weights, num_bins=500, energy_range=np.array([-6,2]), sigma = 0.05)
    calculate_and_plot_jdos_q0_parallel(friction_aimsout, friction_dirname,-1, 1, 500, 0.01, n_spin, temperature, fermi_mode)
    calculate_and_plot_jdos_allq_parallel(friction_aimsout, friction_dirname,-1, 1, 500, 0.01, n_spin, temperature, fermi_mode)

    #friction_tensor = calculate_friction_tensor(friction_aimsout, friction_dirname, n_spin, sigma, temperature, friction_max_energy, perturbing_energies)
    friction_tensor = calculate_friction_tensor_parallel(friction_aimsout, friction_dirname, n_spin, sigma, temperature, friction_max_energy, perturbing_energies, fermi_mode)


    relaxation_tensor = np.zeros_like(friction_tensor) # s-1
    for p, pe in enumerate(perturbing_energies):
        relaxation_tensor[:,:,p] = mass_weight_tensor(friction_tensor[:,:,p], friction_masses)
    np.save("mass-weighted_friction.npy", relaxation_tensor/1e12) # convert to ps-1
    plot_relaxation_rates(np.real(relaxation_tensor),perturbing_energies, output_pdf="cartesian_tensor.pdf", output_txt="cartesian_tensor.txt")

    projected_tensor = np.zeros_like(relaxation_tensor) # s-1
    for p, pe in enumerate(perturbing_energies):
        projected_tensor[:,:,p] = project_tensor(relaxation_tensor[:,:,p], modes)

    plot_relaxation_rates(np.real(projected_tensor),perturbing_energies)

    linewidths = output_linewidth_data(projected_tensor, perturbing_energies, mode="grid")#, mode="normal_modes")


