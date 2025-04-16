import numpy as np
from ase.io import read
from ase.db import connect
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

def parse_vib_aims(aimsout):

    energies_list = []
    with open(aimsout, "r") as file:
        for line in file:
            if "  | Total energy corrected      " in line:
                total_energy_corrected = float(line.split()[5])
                energies_list.append(total_energy_corrected)

    atoms_list = []
    with open(aimsout, "r") as file:
        initial_cell = []
        parse_cell = False
        parse_structure = False
        structure_count = 0
        for line in file:
            line = line.strip("\n")

            if "| Unit cell:" in line:
                parse_cell = True
                continue
            if parse_cell:
                if "| Atomic structure:" in line:
                    parse_cell = False
                    continue
                else:
                    vector = np.array([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])          
                    initial_cell.append(vector)
                    continue

            if "x [A]" in line and "y [A]" in line:
                parse_structure = True
                cell = []
                symbols = []
                positions = []
                continue
                 
            if parse_structure:
                if structure_count == 0:
                    if "Species" in line:
                        symbols.append(line.split()[3])
                        position = np.array([float(line.split()[4]), float(line.split()[5]), float(line.split()[6])])
                        positions.append(position)
                        continue
                    else:
                        atoms = Atoms(symbols=symbols, positions=np.array(positions), cell=np.array(initial_cell), pbc=True)
                        atoms_list.append(atoms)
                        parse_structure = False
                        structure_count += 1
                        continue
                else:
                    if "lattice_vector" in line:
                        lattice_vector = np.array([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
                        cell.append(lattice_vector)
                        continue
                    if "atom" in line:
                        symbols.append(line.split()[-1])
                        position = np.array([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
                        positions.append(position)
                        continue
                    if line=="":
                        continue 
                    else:
                        atoms = Atoms(symbols=symbols, positions=np.array(positions), cell=np.array(cell), pbc=True)
                        atoms_list.append(atoms)
                        parse_structure = False
                        structure_count += 1
                        continue

    forces_list = []
    with open(aimsout, "r") as file:
        parse_forces = False
        forces = []
        for line in file:
            if "Total atomic forces (unitary forces cleaned) " in line:
                parse_forces = True
                continue

            if parse_forces:

                if "|" in line:
                    force = np.array([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                    forces.append(force)
                    continue
                else:
                    parse_forces = False
                    forces_list.append(forces)
                    forces = []
                    continue


        # Relaxation tensor
        with open(aimsout, "r") as file:
            parse_tensor = False
            relaxation_tensor = []
            for line in file:
                if "Printing Friction Tensor in 1/ps" in line:
                    parse_tensor = True
                    continue
                if "END Printing Friction Tensor" in line:
                    parse_tensor = False
                    break
                if parse_tensor:
                    tensor_row = np.array([float(a) for a in line.split()[1:]])
                    relaxation_tensor.append(tensor_row)
                    
    friction_indices = []
    with open(aimsout, "r") as file:
         for line in file:
            if "Found friction request for atom  " in line:
                atom_index = int(line.split()[-1])-1 # Convert to 0-based index
                friction_indices.append(atom_index)



    for i in range(len(atoms_list)):
        atoms_list[i].set_calculator(SinglePointCalculator(atoms_list[i], energy=energies_list[i], forces=forces_list[i]))

    #return atoms_list, energies_list, forces_list, relaxation_tensor
    return atoms_list, np.array(relaxation_tensor), friction_indices

def add_friction_database_entry(atoms,db,relaxation_tensor, friction_indices):

    # Iterate over the lists and add entries to the database
        # Create a new entry in the database
    db.write(atoms, data={"friction_tensor": relaxation_tensor, "friction_indices": friction_indices})
        # row = con.get(id=i)
        # if i==0:
        #     db.update(id=i,relaxation_tensor=str(relaxation_tensor))
        #     db.update(id=i,friction_indices=friction_indices)
        #db.update(i,energy=energy,forces=forces)
    return db

def add_database_entries(atoms_list,db):
    # Create a new ASE database
    # db = connect("vib_aims.db")

    # Iterate over the lists and add entries to the database
    for i in range(len(atoms_list)):
        atoms = atoms_list[i]
        # Create a new entry in the database
        db.write(atoms)
        # row = con.get(id=i)
        # if i==0:
        #     db.update(id=i,relaxation_tensor=str(relaxation_tensor))
        #     db.update(id=i,friction_indices=friction_indices)
        #db.update(i,energy=energy,forces=forces)
    return db


if __name__=="__main__":

    atoms_list, relaxation_tensor, friction_indices = parse_vib_aims("aims.out")

    db = connect("vib_aims.db")
    db = add_friction_database_entry(atoms_list[0], db, relaxation_tensor, friction_indices)
    db = add_database_entries(atoms_list[1:], db)