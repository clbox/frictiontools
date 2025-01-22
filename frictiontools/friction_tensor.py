from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from frictiontools.aims_parse import *
from frictiontools.constants import *
from frictiontools.utils import *
from frictiontools.fermi_integral import *

def calculate_friction(g, g_prime, evs, n_spin, chem_pot, k_weight, sigma, temperature, energy_cutoff, perturbing_energies, fermi_mode):
   
    n_states, _,  = g.shape

    max_occupied_state = find_max_occupied_state(evs, chem_pot, temperature)
    min_unoccupied_state = find_min_unoccupied_state(evs, chem_pot, temperature)

    minimum_state, maximum_state = find_bounded_states(evs, chem_pot, energy_cutoff)

    friction = np.zeros(len(perturbing_energies), dtype=np.complex128)

    for i in range(minimum_state, max_occupied_state):
        for j in range(min_unoccupied_state,maximum_state):

            epsilon = evs[j] - evs[i]


            if abs(epsilon)<1e-30:
                continue
            
            # Exclude negative excitations when using fi-fj , since they are included implicitly
            if epsilon<0:
                continue

            
            fermi_factor = evaluate_fermi_factor(evs[i], evs[j], chem_pot, temperature, perturbing_energies, fermi_mode)
            fermi_factor = fermi_factor * (2/n_spin)


            # if abs(fermi_factor)<0.0001:
            #     continue
            
            nac_cmplx = np.conj(g[i,j]) * g_prime[i,j] 

            friction_tmp = nac_cmplx * delta_function(epsilon, perturbing_energies, sigma, "gaussian") * fermi_factor 

            friction[:] += friction_tmp


    friction[:] = friction[:] * k_weight * np.pi * hbar


    return friction

def calculate_friction_tensor(friction_aimsout, friction_dirname, n_spin, sigma, temperature, friction_max_energy, perturbing_energies, fermi_mode):

    friction_indices = get_friction_indices(friction_aimsout)

    n_atoms = get_number_of_atoms(friction_aimsout)
    n_friction_atoms = len(friction_indices)
    ndims = 3 * n_friction_atoms

    n_k_points, k_weights = parse_kpoints_and_weights(friction_aimsout)
    chem_pot, evs = parse_ev_data(friction_dirname)

    friction_tensor = np.zeros((ndims,ndims,len(perturbing_energies)), dtype=np.complex128)
    friction = np.zeros(max(1, len(perturbing_energies)), dtype=np.complex128)

    for i_spin in range(n_spin): 

        i_coord = -1
        for i_atom in range(n_atoms):
            if i_atom not in friction_indices:
                continue
            print(f"      Processing atom {i_atom}...")

            for i_cart in range(3):
                i_coord +=1

                j_coord = -1
                for j_atom in range(n_atoms):

                    if j_atom not in friction_indices:
                        continue
                    
                    for j_cart in range(3):
                        j_coord += 1

                        if (j_coord<i_coord):
                            continue

                        # if (i_coord!=11 and j_coord !=1):
                        #     continue

                        for i_k_point in range(n_k_points):

                            g = parse_epc_data_kpoint(friction_dirname, i_k_point, i_atom, i_cart, i_spin)
                        
                            g_prime = parse_epc_data_kpoint(friction_dirname, i_k_point, j_atom, j_cart, i_spin)
                            friction[:] = calculate_friction(g, g_prime, evs[i_k_point,:], n_spin, chem_pot, k_weights[i_k_point], sigma, temperature, friction_max_energy, perturbing_energies, fermi_mode)
                            friction_tensor[i_coord,j_coord, :] += friction[:]

                            if i_coord>j_coord:
                                friction[j_coord,i_coord,:] += np.conj(friction[:])

    friction_tensor = friction_tensor * (eV_to_J / (Ang_to_m)**2)   #kg s-1
    return friction_tensor

def calculate_friction_tensor_for_k_point(i_k_point, i_spin, n_atoms, friction_indices, friction_dirname, evs, chem_pot, k_weights, sigma, temperature, friction_max_energy, perturbing_energies, n_spin, fermi_mode):
    # Initialize a local friction tensor for this k-point
    local_friction_tensor = np.zeros((len(friction_indices) * 3, len(friction_indices) * 3, len(perturbing_energies)), dtype=np.complex128)
    friction = np.zeros(max(1, len(perturbing_energies)), dtype=np.complex128)
    i_coord = -1
    for i_atom in range(n_atoms):
        if i_atom not in friction_indices:
            continue

        for i_cart in range(3):
            i_coord += 1

            j_coord = -1
            for j_atom in range(n_atoms):
                if j_atom not in friction_indices:
                    continue

                for j_cart in range(3):
                    j_coord += 1

                    if j_coord < i_coord:
                        continue

                    # if i_coord != 11 and j_coord != 1:
                    #     continue

                    g = parse_epc_data_kpoint(friction_dirname, i_k_point, i_atom, i_cart, i_spin)
                    g_prime = parse_epc_data_kpoint(friction_dirname, i_k_point, j_atom, j_cart, i_spin)
                    friction[:] = calculate_friction(
                        g, g_prime, evs[i_k_point, :], n_spin, chem_pot, k_weights[i_k_point], sigma, temperature, friction_max_energy, perturbing_energies, fermi_mode
                    )
                    local_friction_tensor[i_coord, j_coord, :] += friction[:]

                    if i_coord > j_coord:
                        local_friction_tensor[j_coord, i_coord, :] += np.conj(friction[:])

    return local_friction_tensor

def calculate_friction_tensor_parallel(friction_aimsout, friction_dirname, n_spin, sigma, temperature, friction_max_energy, perturbing_energies, fermi_mode):
    # Use parallelism to calculate the friction tensor for each k-point
    friction_indices = get_friction_indices(friction_aimsout)

    n_atoms = get_number_of_atoms(friction_aimsout)
    n_friction_atoms = len(friction_indices)
    ndims = 3 * n_friction_atoms

    n_k_points, k_weights = parse_kpoints_and_weights(friction_aimsout)
    chem_pot, evs = parse_ev_data(friction_dirname)

    for i_spin in range(n_spin):
        results = Parallel(n_jobs=-1)(
            delayed(calculate_friction_tensor_for_k_point)(
                i_k_point, i_spin, n_atoms, friction_indices, friction_dirname, evs, chem_pot, k_weights, sigma, temperature, friction_max_energy, perturbing_energies, n_spin, fermi_mode
            )
            for i_k_point in range(n_k_points)
        )

        # Sum the results from all k-points
        global_friction_tensor = np.sum(results, axis=0)

    global_friction_tensor = global_friction_tensor * (eV_to_J / (Ang_to_m)**2)   #kg s-1
    return global_friction_tensor

def project_tensor(cartesian_tensor, modes):

    normal_mode_tensor = np.dot(modes,np.dot(cartesian_tensor,modes.transpose()))


    return normal_mode_tensor

def mass_weight_tensor(tensor, masses):
    """
    Mass-weight a 3N x 3N tensor by dividing element (ij) by sqrt(mass_i) * sqrt(mass_j).
    
    Parameters:
        tensor (numpy.ndarray): A 3N x 3N numpy array (the input tensor).
        masses (numpy.ndarray): A 1D numpy array of length N containing atomic masses.
        
    Returns:
        numpy.ndarray: The mass-weighted tensor (3N x 3N).
    """
    # Ensure the input tensor is square
    if tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Tensor must be square (3N x 3N).")
    
    # Ensure the dimensions of the tensor and masses align
    if tensor.shape[0] % 3 != 0:
        raise ValueError("Tensor dimensions must be a multiple of 3.")
    
    N = tensor.shape[0] // 3
    if len(masses) != N:
        raise ValueError("Length of masses array must match the number of atoms (N).")
    
    # Create a 3N x 3N array of mass weights
    sqrt_masses = np.sqrt(np.repeat(masses, 3))  # Repeat each mass 3 times
    weight_matrix = np.outer(sqrt_masses, sqrt_masses)  # Compute sqrt(mass_i) * sqrt(mass_j)
    
    # Divide each element of the tensor by the corresponding weight
    weighted_tensor = tensor / weight_matrix
    
    return weighted_tensor

def output_linewidth_data(projected_tensor, energies, mode="grid"):

    ndims, _, n_energies = np.shape(projected_tensor)

    if mode=="normal_modes":
        linewidths = np.zeros((ndims)) #eV
        for m in range(len(linewidths)):
            linewidths[m] = np.real(hbar*projected_tensor[m,m,m]) 

        # Prepare the data
        indices = np.arange(1, len(linewidths) + 1)  # Mode indices
        linewidths_meV = linewidths * eV_to_meV
        linewidths_cm1 = linewidths * eV_to_cm1
        lifetimes_ps = hbar / linewidths * 1e12  # Lifetime in ps

        # Combine all data into a single array
        output_data = np.column_stack((indices, linewidths, linewidths_meV, linewidths_cm1, lifetimes_ps))

        # Save to a text file
        header = "Mode Index    Linewidth (eV)    Linewidth (meV)    Linewidth (cm^-1)    Lifetime (ps)"
        np.savetxt("linewidths.txt", output_data, fmt="%-15d %-18.6e %-18.6e %-20.6e %-20.6e", header=header)

        print("Data saved to 'linewidths.txt'")
        return linewidths
    

    elif mode =="grid":
        linewidths = np.zeros((ndims, n_energies))
        for m in range(len(linewidths)):
            linewidths[m, :] = np.real(hbar*projected_tensor[m,m,:])

        # Prepare the data
        indices = np.arange(1, len(linewidths) + 1)  # Mode indices
        linewidths_meV = linewidths * eV_to_meV
        linewidths_cm1 = linewidths * eV_to_cm1
        #lifetimes_ps = hbar / linewidths * 1e12  # Lifetime in ps

        with open("linewidths.txt", "w") as file:
            for mode in range(ndims):
                # Write the header for each mode
                file.write(f"# Mode {mode + 1}\n")
                file.write("# Energy (eV), Linewidth (eV), Linewidth (meV), Linewidth (cm^-1)\n")#, Lifetime (ps)\n")

                for i in range(n_energies):
                    # Write the data row
                    file.write(f"{energies[i]:.6f}, {linewidths[mode,i]:.6e}, {linewidths_meV[mode,i]:.6e}, {linewidths_cm1[mode,i]:.6e}\n")# , {lifetimes_ps[mode,i]:.6e}\n")
                
                file.write("\n")  # Add a blank line between modes
    else:
        print("Linewidth output runmode not recognised")
    
def plot_relaxation_rates(projected_tensor, perturbing_energies, output_pdf="relaxation_rates.pdf", output_txt="relaxation_rates.txt"):
    """
    Plot the diagonal elements of the relaxation rate tensor as a function of hbar omega and save the data.

    Args:
        projected_tensor (ndarray): Relaxation rate tensor of shape (n_modes, n_modes, len(perturbing_energies)).
        perturbing_energies (ndarray): Array of perturbing energies in eV.
        output_pdf (str): Filename for the PDF plot.
        output_txt (str): Filename for the output text file.
    """
    n_modes, _, n_energies = projected_tensor.shape
    
    # Convert relaxation rates to ps^-1 and perturbing energies to meV
    projected_tensor_ps = projected_tensor / 1e12  # Convert s^-1 to ps^-1
    #perturbing_energies_meV = perturbing_energies * 1000  # Convert eV to meV

    # Plot the diagonal elements
    fig, axes = plt.subplots(n_modes, 1, figsize=(8, n_modes * 2), sharex=True)
    fig.suptitle("Diagonal Relaxation Rates as a Function of hbar omega")

    if n_modes == 1:
        axes = [axes]  # Ensure axes is iterable for single mode

    for mode in range(n_modes):
        axes[mode].plot(
            perturbing_energies,
            projected_tensor_ps[mode, mode, :],
            label=f"Mode {mode + 1}",
            color="blue"
        )
        axes[mode].set_ylabel(r"Relaxation Rate / ps$^{-1}$ ")
        axes[mode].legend()
        axes[mode].grid(True)

    axes[-1].set_xlabel(r"$\hbar \omega$ / eV ")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_pdf)
    plt.close()
    
    # Save data to text file
    with open(output_txt, "w") as f:
        f.write("# Perturbing Energies (eV), Relaxation Rates (ps^-1) for Diagonal Elements\n")
        for i, energy in enumerate(perturbing_energies):
            f.write(f"{energy:.6e}")
            for mode in range(n_modes):
                f.write(f"\t{projected_tensor_ps[mode, mode, i]:.6e}")
            f.write("\n")