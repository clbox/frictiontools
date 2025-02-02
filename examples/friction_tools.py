
import numpy as np
import matplotlib.pyplot as plt
import time

from frictiontools.aims_parse import get_friction_masses, get_friction_indices, extract_atomic_symbols, get_number_of_atoms, parse_kpoints_and_weights, parse_ev_data
from frictiontools.vib_parse import parse_normal_modes, parse_perturbing_energies
from frictiontools.dos import output_dos, calculate_dos_fermi
from frictiontools.jdos import calculate_and_plot_jdos_q0_parallel, calculate_and_plot_jdos_allq_parallel
from frictiontools.friction_tensor import * 
from frictiontools.constants import *
from frictiontools.epc import *
from frictiontools.properties import * 
# Todo check bounds on looping states (base 0)
# Todo:  FHI-aims friction EPC based restart
# Todo: Detailed information on what excitations are being included

# TODO: Vibrational energy Evib
# TODO: 

#
if __name__ == "__main__":
    

    normal_mode_filename = "vibrations/normal_modes"
    # Get hbar omega for each mode
    
    perturbing_energies = parse_perturbing_energies(normal_mode_filename)
    ndims = len(perturbing_energies)
    modes = parse_normal_modes(normal_mode_filename,ndims)
    # perturbing_energies = np.array([0.245])
    perturbing_energy_grid = np.linspace(-1,1,100)



    friction_dirname = "friction/"
    n_q_points = 1
    sigma = 0.6 # eV
    n_spin = 1
    
    params = FrictionParams(
        friction_aimsout=friction_dirname+"aims.out",
        friction_dirname=friction_dirname,
        n_spin=n_spin,
        sigma=sigma, # eV
        temperature=0, # K
        friction_max_energy=3.2, # eV
        perturbing_energies=perturbing_energies,
        fermi_mode="diff", #diff = fi - fj
        #expression="allen_low_temperature",
        expression="default",
        delta_order=1,
        n_jobs=-1, # parallelism
    )

    system_properties = get_system_properties(params)
    
    print_friction_params(params)
    print_system_properties(system_properties)

    chem_pot = system_properties.chem_pot
    dos_grid, dos = output_dos(system_properties.evs, chem_pot, system_properties.k_weights, num_bins=500, energy_range=[chem_pot-4,chem_pot+4], sigma = sigma)

    dos_fermi = calculate_dos_fermi(system_properties.evs, system_properties.k_weights, chem_pot, sigma=sigma)
    print(f"DOS at Fermi level: {dos_fermi:.6f} states/eV")

    calculate_and_plot_jdos_q0_parallel(params, -1, 1, 500)
    calculate_and_plot_jdos_allq_parallel(params, -1, 1, 500)

    #friction_tensor = calculate_friction_tensor_serial(friction_aimsout, friction_dirname, n_spin, sigma, temperature, friction_max_energy, perturbing_energies)

    start_time = time.time()
    #friction_tensor = calculate_friction_tensor_parallel(params)
    friction_tensor = calculate_friction_tensor_parallel(params)
    end_time = time.time()
    print(f"Time taken to calculate friction tensor: {end_time-start_time:.2f} seconds")







    #  Processing the friction tensor
    friction_masses = system_properties.friction_masses
    relaxation_tensor = np.zeros_like(friction_tensor) # s-1
    for p, pe in enumerate(perturbing_energies):
        relaxation_tensor[:,:,p] = mass_weight_tensor(friction_tensor[:,:,p], friction_masses)
    #np.save("mass-weighted_friction.npy", relaxation_tensor/1e12) # convert to ps-1
    plot_relaxation_rates(np.real(relaxation_tensor),perturbing_energies, output_pdf="cartesian_tensor.pdf", output_txt="cartesian_tensor.txt")
    
    projected_tensor = np.zeros_like(relaxation_tensor) # s-1
    for p, pe in enumerate(perturbing_energies):
        projected_tensor[:,:,p] = project_tensor(relaxation_tensor[:,:,p], modes)

    plot_relaxation_rates(np.real(projected_tensor),perturbing_energies)

    #linewidths = output_linewidth_data(projected_tensor, perturbing_energies, mode="grid")#, mode="normal_modes")
    linewidths = output_linewidth_data(projected_tensor, perturbing_energies, mode="normal_modes")

    lambdas, lambda_total = calculate_epc_strength_mode(linewidths, dos_fermi, perturbing_energies, n_spin)
    print("Mode-resolved Electron-Phonon Coupling Strength (λ)", lambdas)
    print("Total Electron-Phonon Coupling Strength (λ)", lambda_total)

    
    omega_values = perturbing_energy_grid/hbar
    alpha2_F_values = calculate_alpha2_F_gaussian(omega_values, linewidths, perturbing_energies, dos_fermi, n_q_points, sigma/hbar)
    print(alpha2_F_values)
    plot_eliashberg_function(omega_values*hbar, alpha2_F_values, output_file="eliashberg_function.pdf")


    Tc = calculate_critical_temperature(np.average(lambdas), 750, mu_star=0.1)
    print(f"Superconducting Critical Temperature (Tc): {Tc:.2f} K")