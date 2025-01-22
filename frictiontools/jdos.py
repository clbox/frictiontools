from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from frictiontools.aims_parse import parse_kpoints_and_weights, parse_ev_data
from frictiontools.fermi_integral import evaluate_fermi_factor, find_max_occupied_state, find_min_unoccupied_state, find_bounded_states, delta_function
from frictiontools.utils import *


def process_jdos_q0_k_point(evs, chem_pot, k_weight, sigma, temperature, energy_cutoff, energy_bins, n_spin, fermi_mode):
    """
    Process a single k-point contribution to the JDOS.
    Args:
        args (tuple): Contains (k_index, evs, k_weights, bin_centers, sigma).
    Returns:
        ndarray: JDOS contribution for the given k-point.
    """

    max_occupied_state = find_max_occupied_state(evs, chem_pot, temperature)
    min_unoccupied_state = find_min_unoccupied_state(evs, chem_pot, temperature)

    minimum_state, maximum_state = find_bounded_states(evs, chem_pot, energy_cutoff)

    jdos_k = np.zeros(len(energy_bins))

    for i in range(minimum_state, max_occupied_state):
        for j in range(min_unoccupied_state,maximum_state):

            epsilon = evs[j] - evs[i]

            # FHI-aims friction has a bug where epsilon<tiny(epsilon) was used as the condition,
            # missing the abs, which therefore excludes negative excitations.
            if abs(epsilon)<1e-30:
                continue

            # Exclude negative excitations when using fi-fj , since they are included implicitly
            if epsilon<0:
                continue

            fermi_factor = evaluate_fermi_factor(evs[i], evs[j], chem_pot, temperature, energy_bins, fermi_mode)
            fermi_factor = fermi_factor * (2/n_spin)
        
            jdos_k[:] += fermi_factor * delta_function(epsilon, energy_bins, sigma, "gaussian")


    jdos_k[:] = jdos_k[:] * k_weight 

    return jdos_k

def process_jdos_allq_k_point(i_k_point,evs, chem_pot, k_weights, sigma, temperature, energy_cutoff, energy_bins, n_spin, fermi_mode):
    """
    Process a single k-point contribution to the JDOS.
    Args:
        args (tuple): Contains (k_index, evs, k_weights, bin_centers, sigma).
    Returns:
        ndarray: JDOS contribution for the given k-point.
    """
    n_k_points = len(k_weights)
    max_occupied_state_k = find_max_occupied_state(evs[i_k_point], chem_pot, temperature)
    min_unoccupied_state_k = find_min_unoccupied_state(evs[i_k_point], chem_pot, temperature)
    minimum_state_k, maximum_state_k = find_bounded_states(evs[i_k_point], chem_pot, energy_cutoff)

    jdos_k = np.zeros(len(energy_bins))
    for i_k_point2 in range(n_k_points):

        max_occupied_state_kq = find_max_occupied_state(evs[i_k_point2], chem_pot, temperature)
        min_unoccupied_state_kq = find_min_unoccupied_state(evs[i_k_point2], chem_pot, temperature) 

        minimum_state_kq, maximum_state_kq = find_bounded_states(evs[i_k_point2], chem_pot, energy_cutoff)
        q_weight = k_weights[i_k_point] * k_weights[i_k_point2]

        for i in range(minimum_state_k, max_occupied_state_k):
            for j in range(min_unoccupied_state_kq,maximum_state_kq):

                epsilon = evs[i_k_point2,j] - evs[i_k_point,i]

                # FHI-aims friction has a bug where epsilon<tiny(epsilon) was used as the condition,
                # missing the abs, which therefore excludes negative excitations.
                if abs(epsilon)<1e-30:
                    continue

                # Exclude negative excitations when using fi-fj , since they are included implicitly
                if epsilon<0:
                    continue

            
                fermi_factor = evaluate_fermi_factor(evs[i_k_point,i], evs[i_k_point2,j], chem_pot, temperature, energy_bins, fermi_mode)
                fermi_factor = fermi_factor * (2/n_spin)

            
                jdos_k[:] += (fermi_factor * delta_function(epsilon, energy_bins, sigma, "gaussian"))*q_weight



    return jdos_k

def calculate_and_plot_jdos_q0_parallel(aimsout, dirname, energy_min, energy_max, num_bins, sigma, n_spin, temperature, fermi_mode, output_filename="jdos.txt"):
    """
    Calculate and plot the q=0 Joint Density of States (JDOS) with Gaussian smearing, using parallelism over k-points.

    Args:
        evs (ndarray): Eigenvalues of shape (n_k_points, n_states).
        k_weights (ndarray): Weights for each k-point (shape: n_k_points).
        energy_min (float): Minimum energy for the JDOS range.
        energy_max (float): Maximum energy for the JDOS range.
        num_bins (int): Number of energy bins for the histogram.
        sigma (float): Gaussian smearing parameter (in eV).
        output_filename (str): Filename to save the JDOS data.
    """

    n_k_points, k_weights = parse_kpoints_and_weights(aimsout)
    chem_pot, evs = parse_ev_data(dirname)

    energy_bins = np.linspace(energy_min, energy_max, num_bins + 1)


    # Use multiprocessing to calculate JDOS contributions for each k-point

    for i_spin in range(n_spin):
        results = Parallel(n_jobs=-1)(
            delayed(process_jdos_q0_k_point)(
                evs[i_k_point, :], chem_pot, k_weights[i_k_point], sigma, temperature, energy_max, energy_bins, n_spin, fermi_mode
            )
            for i_k_point in range(n_k_points)
        )

        # Sum the results from all k-points
        jdos = np.sum(results, axis=0)

    # Save JDOS to a file
    with open(output_filename, 'w') as f:
        f.write("# Energy (eV)\tJDOS\n")
        for energy, value in zip(energy_bins, jdos):
            f.write(f"{energy:.6f}\t{value:.6f}\n")

    # Plot the JDOS
    plt.figure(figsize=(8, 6))
    plt.plot(energy_bins, jdos, label=f"JDOS (σ={sigma} eV)", color="blue")
    plt.xlabel("Energy (eV)")
    plt.ylabel("JDOS")
    plt.title("Joint Density of States (JDOS) at q=0 with Gaussian Smearing")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("jdos_q0.pdf")

def calculate_and_plot_jdos_allq_parallel(aimsout, dirname, energy_min, energy_max, num_bins, sigma, n_spin, temperature, fermi_mode, output_filename="jdos.txt"):
    """
    Calculate and plot the q=0 Joint Density of States (JDOS) with Gaussian smearing, using parallelism over k-points.

    Args:
        evs (ndarray): Eigenvalues of shape (n_k_points, n_states).
        k_weights (ndarray): Weights for each k-point (shape: n_k_points).
        energy_min (float): Minimum energy for the JDOS range.
        energy_max (float): Maximum energy for the JDOS range.
        num_bins (int): Number of energy bins for the histogram.
        sigma (float): Gaussian smearing parameter (in eV).
        output_filename (str): Filename to save the JDOS data.
    """

    n_k_points, k_weights = parse_kpoints_and_weights(aimsout)
    chem_pot, evs = parse_ev_data(dirname)

    energy_bins = np.linspace(energy_min, energy_max, num_bins + 1)

    # Use multiprocessing to calculate JDOS contributions for each k-point

    for i_spin in range(n_spin):
        results = Parallel(n_jobs=-1)(
            delayed(process_jdos_allq_k_point)(
                i_k_point, evs, chem_pot, k_weights, sigma, temperature, energy_max, energy_bins, n_spin, fermi_mode
            )
            for i_k_point in range(n_k_points)
        )

        # Sum the results from all k-points
        jdos = np.sum(results, axis=0)

    # Save JDOS to a file
    with open(output_filename, 'w') as f:
        f.write("# Energy (eV)\tJDOS\n")
        for energy, value in zip(energy_bins, jdos):
            f.write(f"{energy:.6f}\t{value:.6f}\n")

    # Plot the JDOS
    plt.figure(figsize=(8, 6))
    plt.plot(energy_bins, jdos, label=f"JDOS (σ={sigma} eV)", color="blue")
    plt.xlabel("Energy (eV)")
    plt.ylabel("JDOS")
    plt.title("Joint Density of States (JDOS) at all q with Gaussian Smearing")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("jdos_allq.pdf")