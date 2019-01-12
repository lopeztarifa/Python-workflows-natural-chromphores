
import argparse
from noodles import gather
from qmworks import (dftb, run, Settings, templates)
from qmworks.utils import (chunksOf, concatMap)
from qmworks.parsers.xyzParser import string_to_plams_Molecule


import numpy as np

def main(path_xyz):

    molecules = create_molecules(path_xyz)
    
    jobs = [create_dftb_job(m, i) for i, m in enumerate(molecules)]

    # Extract dipoles
    ds = gather(*[j.dipole for j in jobs])

    # Extract charges
    cs = gather(*[j.charges for j in jobs])

    # Run the workflow
    dipoles, charges = run(gather(ds, cs), folder='charges')

    # Print the dipoles
    np.savetxt('dipoles.txt', dipoles)
    save_coordinates_charges(molecules, charges)
    

def create_dftb_job(mol, i):
    """
    Create the settings for the DFTB job
    """
    s = templates.singlepoint
    s.specific.dftb.dftb.scc.iterations = 1000
    s.specific.dftb.dftb.scc.thirdorder = ""
    # s.specific.dftb.properties.dipolemoment = ''

    return dftb(s, mol, job_name='point_{}'.format(i))


def create_molecules(path_xyz):
    """
    Create a set of plams molecules from a xyz trajectory
    """
    xss = split_file_geometries(path_xyz)
    
    return list(map(string_to_plams_Molecule, xss))


def split_file_geometries(path_xyz):
    """
    Reads a set of molecular geometries in xyz format and returns
    a list of string, where is element a molecular geometry
    :returns: String list containing the molecular geometries.
    """
    # Read Cartesian Coordinates
    with open(path_xyz, 'r') as f:
        xss = f.readlines()

    numat = int(xss[0].split()[0])
    return list(map(''.join, chunksOf(xss, numat + 2)))


def save_coordinates_charges(molecules, charges):
    """
    write the charges and coordinates in XYZ format
    """
    xs = concatMap(lambda t: format_xyz(*t), zip(molecules, charges))

    with open('coordinates_charges.xyz', 'w') as f:
        f.write(''.join(xs))


def format_xyz(mol, charges):
    """
    Write the coordinates and charges in XYZ format
    """
    s = '{}\n\n'.format(len(charges))
    for at, cs in zip(mol, charges):
        s += '{} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}\n'.format(at.symbol, *at.coords, cs)

    return s


if __name__ == "__main__":
    msg = " script -xyz <path/to/trajectory/xyz> "

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-xyz', required=True, help='Path to trajectory')

    args = parser.parse_args()

    main(args.xyz)
