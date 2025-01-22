import numpy as np
import scipy.sparse as sp
import struct



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

def parse_ev_data(dirname):


    aims_filename = dirname+"aims.out"
    chem_pot = parse_chem_pot(aims_filename)

    evs_filename = dirname+"friction_KS_eigenvalues.out"
    evs_file = evs_filename
    evs = parse_evs(evs_file)


    return chem_pot, evs

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