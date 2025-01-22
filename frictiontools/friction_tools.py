
import numpy as np
import scipy.sparse as sp
import struct
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

hbar = 6.582119569e-16  # eV·s
eV_to_J =  1.602e-19
Ang_to_m = 1e-10

# Constants for conversion
eV_to_meV = 1000
eV_to_cm1 = 8065.54  # 1 eV = 8065.54 cm⁻¹

# Todo check bounds on looping states (base 0)
# Todo:  FHI-aims friction EPC based restart
# Todo: Detailed information on what excitations are being included

# TODO: Density of States
# TODO: Joint - Density of States - q=0
# TODO: Joint - Density of States - all q

# 

def read_elsi_to_csc(filename):
    mat = open(filename,"rb")
    data = mat.read()
    mat.close()
    i8 = "l"
    i4 = "i"

    # Get header
    start = 0
    end = 128
    header = struct.unpack(i8*16,data[start:end])

    # Number of basis functions (matrix size)
    n_basis = header[3]

    # Total number of non-zero elements
    nnz = header[5]

    # Get column pointer
    start = end
    end = start+n_basis*8
    col_ptr = struct.unpack(i8*n_basis,data[start:end])
    col_ptr += (nnz+1,)
    col_ptr = np.array(col_ptr)

    # Get row index
    start = end
    end = start+nnz*4
    row_idx = struct.unpack(i4*nnz,data[start:end])
    row_idx = np.array(row_idx)

    # Get non-zero value
    start = end

    if header[2] == 0:
        # Real case
        end = start+nnz*8
        nnz_val = struct.unpack("d"*nnz,data[start:end])
    else:
        # Complex case
        end = start+nnz*16
        nnz_val = struct.unpack("d"*nnz*2,data[start:end])
        nnz_val_real = np.array(nnz_val[0::2])
        nnz_val_imag = np.array(nnz_val[1::2])
        nnz_val = nnz_val_real + 1j*nnz_val_imag

    nnz_val = np.array(nnz_val)

    # Change convention

    for i_val in range(nnz):
        row_idx[i_val] -= 1

    for i_col in range(n_basis+1):
        col_ptr[i_col] -= 1

    return sp.csc_matrix((nnz_val,row_idx,col_ptr),shape=(n_basis,n_basis))

def parse_chem_pot(aims_file):
    chem_pot = 0
    with open(aims_file, "r") as af:
        for line in af:
            if '**FRICTION**' in line:
                break
            if '| Chemical potential (Fermi level):' in line:
                chem_pot = float(line.split()[-2])
    return chem_pot # eV

def parse_evs(evs_file):
    with open(evs_file,"r") as f:
        for i in range(3):
            line = f.readline()
        line = f.readline()
        ncols = len(line.split())


    evs = np.loadtxt(evs_file,skiprows=3,usecols=range(1,ncols))
    # print(np.shape(evs))

    return evs

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

def gaussian_function(x,x0,s):
    return 1./np.sqrt(2*np.pi*s*s)*np.exp(-0.5*(((x-x0) / (s))**2))

def parse_timestep_data_kpoint(dirname,i_k_point,i_atom,i_cart,i_spin,parse_evecs=True):


    aims_filename = dirname+"aims.out"
    evs_filename = dirname+"friction_KS_eigenvalues.out"
    
    ham1_filename = dirname+"first_order_H_atom_ATOMID_cart_CARTID_k_KID.csc"
    ovlp1_filename = dirname+"first_order_S_atom_ATOMID_cart_CARTID_k_KID.csc"

    # Parse Fermi level (chemical potential)
    # start = time.time() 
    chem_pot = parse_chem_pot(aims_filename)
    # end = time.time()
    # print('        Time for 1 parse chempot/ s: '+str(end - start))
    # print('Chemical potential: '+str(chem_pot)+' / eV')

    evs_file = evs_filename
  
    ham1_file = ham1_filename.replace('ATOMID','{:06d}'.format(i_atom+1)).replace('CARTID','{:1d}'.format(i_cart+1)).replace('KID','{:06d}'.format(i_k_point+1))
    ovlp1_file = ovlp1_filename.replace('ATOMID','{:06d}'.format(i_atom+1)).replace('CARTID','{:1d}'.format(i_cart+1)).replace('KID','{:06d}'.format(i_k_point+1))

    # start = time.time() 
    evs = parse_evs(evs_file)
    # end = time.time()
    # print('        Time for 1 parse ev/ s: '+str(end - start))

    # start = time.time()
    ham1= read_elsi_to_csc(ham1_file)
    ovlp1 = read_elsi_to_csc(ovlp1_file)
    # end = time.time()
    # print('        Time for 1 parse ham1,ovlp1/ s: '+str(end - start))


    if parse_evecs:
        evecs_filename = dirname+"C_spin_SPINID_kpt_KID.csc"
        evecs_file = evecs_filename.replace('SPINID','{:02d}'.format(i_spin+1)).replace('KID','{:06d}'.format(i_k_point+1))
        evecs = read_elsi_to_csc(evecs_file)
    else:
        evecs = 0.

    return chem_pot, evs, evecs, ham1, ovlp1

def parse_epc_data_kpoint(dirname,i_k_point,i_atom,i_cart,i_spin):

    hartree = 27.2113845
    bohr = 0.52917721   

    # aims_filename = dirname+"aims.out"
    # evs_filename = dirname+"friction_KS_eigenvalues.out"
    
    epc_filename = dirname+"epc_atom_ATOMID_cart_CARTID_k_KID.csc"


    # Parse Fermi level (chemical potential)
    # start = time.time() 
    # chem_pot = parse_chem_pot(aims_filename)
    # end = time.time()
    # print('        Time for 1 parse chempot/ s: '+str(end - start))
    # print('Chemical potential: '+str(chem_pot)+' / eV')

    # evs_file = evs_filename
  
    epc_file = epc_filename.replace('ATOMID','{:06d}'.format(i_atom+1)).replace('CARTID','{:1d}'.format(i_cart+1)).replace('KID','{:06d}'.format(i_k_point+1))


    # start = time.time() 
    # evs = parse_evs(evs_file)
    # end = time.time()
    # print('        Time for 1 parse ev/ s: '+str(end - start))

    # start = time.time()
    epc= read_elsi_to_csc(epc_file)

    # end = time.time()
    # print('        Time for 1 parse epc/ s: '+str(end - start))


   

    return epc*hartree/bohr

def parse_ev_data(dirname):


    aims_filename = dirname+"aims.out"
    chem_pot = parse_chem_pot(aims_filename)

    evs_filename = dirname+"friction_KS_eigenvalues.out"
    evs_file = evs_filename
    evs = parse_evs(evs_file)


    return chem_pot, evs

def parse_normal_modes(filename, ndim):
    
    lines = open(filename).readlines() #ase vib.py output

    modes = np.zeros([ndim,ndim])
    for i, line in enumerate(lines):
        if 'Zero-point energy' in line:
            j = i+1
            for a in range(ndim):
                for b in range(ndim):
                    modes[a,b] = \
                            float(lines[j].split()[b])
                j += 1

    for i in range(ndim):
        modes[i,:]/=np.linalg.norm(modes[i,:])

    return modes

def parse_perturbing_energies(file_path):
    """
    Parses the perturbing energies of the modes from a file and returns them in eV.

    Parameters:
        file_path (str): Path to the file containing the mode data.

    Returns:
        numpy.ndarray: Array of perturbing energies in eV.
    """
    energies_meV = []

    with open(file_path, 'r') as file:
        mode_section = False  # Toggle to detect the mode energy section
        for line in file:
            if "eV" in line:  # Identify the header and separator
                mode_section = not mode_section
                continue

            if mode_section:
                parts = line.split()
                if len(parts) >= 2:  # Ensure the line has enough columns
                    try:
                        energy_meV = float(parts[1].replace('i', ''))  # Ignore 'i' for imaginary values
                        
                        energies_meV.append(energy_meV)
                    except ValueError:
                        continue  # Skip lines that don't parse correctly

    # Convert from meV to eV (1 eV = 1000 meV)
    energies_eV = np.array(energies_meV) / 1000
    return energies_eV

def parse_kpoints_and_weights(file_path):
    """
    Parses the number of k-points and their weights from an `aims.out` file.

    Parameters:
        file_path (str): Path to the `aims.out` file.

    Returns:
        int: Number of k-points.
        numpy.ndarray: Array of k-point weights.
    """
    num_kpoints = 0
    k_weights = []

    with open(file_path, 'r') as file:
        for line in file:
            # Parse the number of k-points
            if "Number of k-points" in line:
                num_kpoints = int(line.split(":")[1].strip())
            
            # Parse k-point weights
            elif "| k-point:" in line and "weight:" in line:
                weight_str = line.split("weight:")[1].strip()
                k_weights.append(float(weight_str))

            elif "Begin self-consistency loop" in line:
                break

    # Convert the weights list to a numpy array
    k_weights = np.array(k_weights)

    return num_kpoints, k_weights

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
    boltzmann_kB = 8.617333262145e-5  # Boltzmann constant in eV/K

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
    
    return (np.exp(-0.5 * ((x - x0) ** 2) / (s ** 2)) / (s * sqrt_pi)) * one_over_sqrt2

def delta_function(epsilon, x0, sigma, type="gaussian"):

    if type=="gaussian":
        result = delta_function_gaussian(epsilon, x0, sigma)
    else:
        print("unknown type of delta function")
        

    return result

def find_max_occupied_state(evs, chem_pot, temperature, tolerance=1e-3):
    n_states = len(evs)
    for i in range(n_states):
        if (fermi_pop(evs[i], chem_pot, temperature) > tolerance):
            max_occupied_state = i
    return max_occupied_state

def find_min_unoccupied_state(evs, chem_pot, temperature,  tolerance=1e-3):
    n_states = len(evs)
    for i in range(n_states):
        if (fermi_pop(evs[i], chem_pot, temperature)<(1-tolerance)):
            min_unoccupied_state = i
            break
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
    # Define the energy window bounds
    lower_bound = chem_pot - 2 * energy_cutoff
    upper_bound = chem_pot + 2 * energy_cutoff

    # Initialize the minimum and maximum state indices to None
    minimum_state = None
    maximum_state = None

    # Iterate over the energy eigenvalues to find the states within the window
    for i, ev in enumerate(evs):
        if ev >= lower_bound and ev <= upper_bound:
            if minimum_state is None or ev < evs[minimum_state]:
                minimum_state = i
            if maximum_state is None or ev > evs[maximum_state]:
                maximum_state = i

    return minimum_state, maximum_state

def project_tensor(cartesian_tensor, modes):

    normal_mode_tensor = np.dot(modes,np.dot(cartesian_tensor,modes.transpose()))


    return normal_mode_tensor

def extract_atomic_symbols(filename):
    atomic_symbols = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(1, len(lines)):  # Start from 1 to avoid index issues
            if "calculate_friction .true." in lines[i]:
                atomic_symbol = lines[i - 1].split()[-1]  # Get last string on previous line
                atomic_symbols.append(atomic_symbol)
    return atomic_symbols

def get_atomic_masses(symbols):
    # Periodic table in atomic mass units (amu)
    periodic_table = {
        "H": 1.00784, "He": 4.002602, "Li": 6.94, "Be": 9.0122, "B": 10.81, 
        "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
        "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
        "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
        "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
        "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
        "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
        "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
        "Nb": 92.906, "Mo": 95.95, "Tc": 98, "Ru": 101.07, "Rh": 102.91,
        "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
        "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
        "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
        "Pm": 145, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
        "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.04,
        "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
        "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
        "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209, "At": 210,
        "Rn": 222, "Fr": 223, "Ra": 226, "Ac": 227, "Th": 232.04, "Pa": 231.04,
        "U": 238.03, "Np": 237, "Pu": 244, "Am": 243, "Cm": 247, "Bk": 247,
        "Cf": 251, "Es": 252, "Fm": 257, "Md": 258, "No": 259, "Lr": 262,
        "Rf": 267, "Db": 270, "Sg": 271, "Bh": 270, "Hs": 277, "Mt": 276,
        "Ds": 281, "Rg": 282, "Cn": 285, "Nh": 286, "Fl": 289, "Mc": 290,
        "Lv": 293, "Ts": 294, "Og": 294
    }
    amu_to_kg = 1.66053906660e-27  # Conversion factor from amu to kg
    masses = []
    for symbol in symbols:
        if symbol in periodic_table:
            mass_kg = periodic_table[symbol] * amu_to_kg
            masses.append(mass_kg)
        else:
            raise ValueError(f"Unknown atomic symbol: {symbol}")
    return masses

def get_friction_masses(friction_aimsout):

    atomic_symbols = extract_atomic_symbols(friction_aimsout)
    masses = get_atomic_masses(atomic_symbols)

    return masses

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

def get_friction_indices(friction_aimsout):
    """
    Search the file for lines containing "Found friction request for atom",
    extract the integer at the end of the line, subtract 1, and return as a Python array.
    """
    search_string = "Found friction request for atom"
    atom_indices = []
    
    with open(friction_aimsout, 'r') as file:
        for line in file:
            if search_string in line:
                # Extract the integer at the end of the line
                atom_index = int(line.split()[-1]) - 1
                atom_indices.append(atom_index)
    
    return atom_indices

def get_number_of_atoms(filename):
    """
    Search the file for the string "| Number of atoms                   : " 
    and return the integer at the end of the line.
    """
    search_string = "| Number of atoms                   : "
    with open(filename, 'r') as file:
        for line in file:
            if search_string in line:
                # Extract the integer from the line
                number_of_atoms = int(line.split(":")[-1].strip())
                return number_of_atoms
    return None  # Return None if the string is not found

def print_system_parameters(n_k_points, k_weights, perturbing_energies, friction_masses, friction_indices, friction_symbols, n_atoms,
                            ndims, sigma, temperature, friction_max_energy, friction_aimsout, normal_mode_filename):
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
    
def calculate_dos(eigenvalues, chemical_potential, k_weights, num_bins=500, energy_range=None, sigma=0.01):
    """
    Calculate the Density of States (DOS) from eigenvalues with k-weights.

    Args:
        eigenvalues (ndarray): Array of shape (n_states, n_k_points) with eigenvalues in eV.
        k_weights (ndarray): Array of shape (n_k_points,) with weights for each k-point.
        num_bins (int): Number of bins for DOS calculation.
        energy_range (tuple): Energy range (min, max) in eV for the DOS.
        sigma (float): Broadening parameter (Gaussian smearing) in eV.

    Returns:
        energies (ndarray): Energy values for DOS.
        dos (ndarray): Calculated DOS values.
    """
    n_k_points, n_states = eigenvalues.shape
    eigenvalues = eigenvalues.flatten()  # Flatten states and k-points into one array
    k_weights = np.repeat(k_weights, n_states)  # Match k-weights to eigenvalues

    if energy_range is None:
        energy_min, energy_max = eigenvalues.min(), eigenvalues.max()
    else:
        energy_min, energy_max = energy_range

    energies = np.linspace(energy_min, energy_max, num_bins)
    dos = np.zeros_like(energies)

    # Calculate DOS with Gaussian smearing and k-weights
    for e, w in zip(eigenvalues, k_weights):
        dos += w * np.exp(-((energies - e) ** 2) / (2 * sigma ** 2))
    dos /= np.sqrt(2 * np.pi) * sigma

    return energies, dos

def save_dos(filename, energies, dos):
    """
    Save the DOS to a file.

    Args:
        filename (str): Output file name.
        energies (ndarray): Energy values.
        dos (ndarray): DOS values.
    """
    with open(filename, "w") as file:
        file.write("# Energy (eV)   DOS\n")
        for e, d in zip(energies, dos):
            file.write(f"{e:.6f}   {d:.6e}\n")

def plot_dos(energies, dos, output_image="dos.pdf"):
    """
    Plot the DOS.

    Args:
        energies (ndarray): Energy values.
        dos (ndarray): DOS values.
        output_image (str): Filename to save the plot (optional).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(energies, dos, label="DOS", color="blue")
    plt.xlabel("Energy (eV)", fontsize=14)
    plt.ylabel("DOS", fontsize=14)
    plt.title("Density of States", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)

def output_dos(eigenvalues, chemical_potential, k_weights, num_bins=500, energy_range=None, sigma=0.01 ):

    energies, dos = calculate_dos(eigenvalues, chemical_potential, k_weights, num_bins, energy_range, sigma)

    save_dos("dos.txt", energies, dos)

    plot_dos(energies, dos, "dos.pdf")

    return energies, dos

def process_jdos_q0_k_point(evs, chem_pot, k_weight, sigma, temperature, energy_cutoff, energy_bins, n_spin,fermi_mode):
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

            fermi_factor = evaluate_fermi_factor(evs[i], evs[j], chem_pot, temperature, perturbing_energies, fermi_mode)
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

            
                fermi_factor = evaluate_fermi_factor(evs[i_k_point,i], evs[i_k_point2,j], chem_pot, temperature, perturbing_energies, fermi_mode)
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
                            ndims, sigma, temperature, friction_max_energy, friction_aimsout, normal_mode_filename)

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


