from functools import partial
from itertools import chain
from noodles import (gather, schedule)
from os.path import join
from plams import Molecule
from qmworks import (Settings, templates, run)
from qmworks import molkit
from qmworks.packages.SCM import adf
from rdkit import Chem

import argparse

smile = 'O.O'


template = Chem.MolFromSmiles(smile)


def main(pdb_path, splitAt, work_dir):

    monomers = create_monomers(pdb_path, splitAt, work_dir)

    settings = Settings()
    settings.basis = "DZP"
#    settings.functional = "CAMY-B3LYP"
    settings.specific.adf.basis.core = "None"
    settings.specific.adf.symmetry = "Nosym"
    settings.specific.adf.xc.hybrid = "CAMY-B3LYP"
    settings.specific.adf.xc.xcfun = ""

    fde_settings = Settings()
#    fde_settings.functional = "camy-b3lyp"
    fde_settings.specific.adf.xc.hybrid = "CAMY-B3LYP"
    fde_settings.specific.adf.xc.xcfun = ""
    fde_settings.basis = "DZP"
    fde_settings.specific.adf.symmetry = "Nosym"
    fde_settings.specific.adf.fde.PW91k = ""
    fde_settings.specific.adf.fde.fullgrid = ""
    fde_settings.specific.adf.allow = "Partialsuperfrags"
    fde_settings.specific.adf.fde.GGAPOTXFD = "Becke"
    fde_settings.specific.adf.fde.GGAPOTCFD = "LYP"

    # Read input from molfiles
    jobs = chain(*list(map(lambda xs: create_jobs(settings, fde_settings, *xs),
                           enumerate(monomers))))

    run(gather(*jobs), n_processes=2)


def create_jobs(settings, fde_settings, i, mols):
    """ Create all the jobs associated to the dimer/monomers"""
    mol1, mol2 = mols
    name1 = 'frag1_frame_{}'.format(i)
    name2 = 'frag2_frame_{}'.format(i)
    name3 = 'frame_{}'.format(i)

    # Prepare isolated fragments
    temp_overlay = templates.singlepoint.overlay
    iso_frag1 = adf(temp_overlay(settings), mol1, job_name='iso_' + name1)
    iso_frag2 = adf(temp_overlay(settings), mol2, job_name='iso_' + name2)

    # Prepare embedded fragments
    emb_frag1 = embed_job(fde_settings, iso_frag1, iso_frag2, 'emb_' + name1,
                          switch=False)
    emb_frag2 = embed_job(fde_settings, iso_frag2, iso_frag1, 'emb_' + name2,
                          switch=True)

    # Prepare excitation calculations
    exc_frag1 = excitations_job(fde_settings, iso_frag1, iso_frag2, emb_frag1, emb_frag2,
                                'exc_' + name1, switch=False)
    exc_frag2 = excitations_job(fde_settings, iso_frag1, iso_frag2, emb_frag2, emb_frag1,
                                'exc_' + name2, switch=True)

    # Run all dependencies and final subsystem calculation
    name = 'subexc_' + name3
    subexc = subexc_job(fde_settings, iso_frag1, iso_frag2, exc_frag1, exc_frag2, name, switch=False)

    return [iso_frag1, iso_frag2, emb_frag1, emb_frag2, exc_frag1, exc_frag2, subexc]


def create_monomers(pdb_path, splitAt, work_dir):
    """
    Read the dimers from the pDB, clean, add hydrogens and split into monomers.
    """
    # Clean PDB and read it with RDkit
    mols = split_and_clean_pdb(pdb_path)
    # Remove Water and Atoms from C2 to C20
    alkyl_chain = ['C{}'.format(i) for i in range(2, 21)]
    water = ['OW', 'HW1', 'HW2']
    names = alkyl_chain + water

    # Read alkyl chain and water from molecules
    clean_mols = map(partial(remove_atoms, names), mols)

    # Create fragments from the molecules
    fragments = map(partial(fragment_molecule, splitAt), clean_mols)

    # saturate the monomers
    fun = lambda ts: (rdkit_saturate(ts[0], ts[1][0], name='monomer1_Hs'),
                      rdkit_saturate(ts[0], ts[1][1], name='monomer2_Hs'))

    saturated_monomers = map(fun, enumerate(fragments))

    return list(saturated_monomers)


def fragment_molecule(splitAt, mol):
    """
    split a dimer in fragment using an integer index `splitAt`.
    Note: It assumes that the number of atoms in each monomer is half of
    the dimers.
    """
    emol1 = Chem.EditableMol(mol)
    emol2 = Chem.EditableMol(mol)

    for x in range(splitAt):
        emol1.RemoveAtom(0)
        emol2.RemoveAtom(splitAt)

    return emol1.GetMol(), emol2.GetMol()


def rdkit_saturate(i, mol, name='Hs'):
    bs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "Mg":
            atom.SetNoImplicit(True)
            for bond in atom.GetBonds():
                bs.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    emol = Chem.EditableMol(mol)
    for b in bs:
        emol.RemoveBond(b[0], b[1])
    newmol = emol.GetMol()

    name_pdb = '{}_{}.pdb'.format(name, i)
    Chem.MolToPDBFile(newmol, name_pdb)

    newmol = Chem.AllChem.AssignBondOrdersFromTemplate(template, newmol)
    molHs = molkit.add_prot_Hs(newmol)

    return molHs


def remove_atoms(names, mol):
    """
    Remove `names` from pdb
    """
    def index_to_remove(i, at):
        info = at.GetPDBResidueInfo()
        name = info.GetName()
        split_name = name.split()[0]

        return split_name in names

    indexes_atoms = enumerate(mol.GetAtoms())
    # filter only the atoms to remove
    index_atoms_to_remove = list(filter(lambda ts: index_to_remove(*ts),
                                        indexes_atoms))
    # extract only the index and discard the atoms
    index_to_delete = list(zip(*index_atoms_to_remove))[0]

    # Editable molecule
    emol = Chem.EditableMol(mol)

    # the accumulator is mandatory because the index of an atom
    # changed when other atoms in the molecule are deleted
    for acc, i in enumerate(index_to_delete):
        emol.RemoveAtom(i - acc)

    return emol.GetMol()


def split_and_clean_pdb(path_pdb):
    """
    Clean Pdb then split then in frames and read them using rdkit
    """
    with open(path_pdb) as f:
        xss = f.read()

    # Filter block containing more than 1 element
    pdbs = filter(lambda x: len(x) > 1, xss.split('END'))
    frames = map(partial(replace_element_name, ('MG', 'Mg')), pdbs)
    mols = map(lambda frame: Chem.rdmolfiles.MolFromPDBBlock(frame, sanitize=False), frames)
    return mols


def replace_element_name(names, st):
    """
    Change the name of the element in the PDB
    """
    data = ''
    old, new = names
    for l in st.splitlines():
        if old in l:
            head, tail = l.split(old)
            data += (head + new + tail + '\n')
        else:
            data += (l + '\n')

    return data


def add_fragments(job1, job2, switch=False):
    print("path1: ", job1.kf.path)
    print("path2: ", job2.kf.path)
    mol_1 = job1.molecule.copy()
    mol_2 = job2.molecule.copy()
    for a in mol_1:
        if not switch:
            a.fragment = 'frag1'
        else:
            a.fragment = 'frag2'
    for a in mol_2:
        if not switch:
            a.fragment = 'frag2'
        else:
            a.fragment = 'frag1'
    m_tot = Molecule()
    if not switch:
       m_tot += mol_1 + mol_2
    else:
       m_tot += mol_2 + mol_1
    return m_tot


@schedule
def embed_job(settings, emb_frag, frozen_frag, jobname, switch):
    """
    Define different jobs
    """
    frag_settings = Settings()
    m_tot = add_fragments(emb_frag, frozen_frag, switch) 
    if not switch:
        frag_settings.specific.adf.fragments['frag1'] = emb_frag.kf.path + ' subfrag=active'
        frag_settings.specific.adf.fragments['frag2'] = frozen_frag.kf.path + ' subfrag=active type=FDE'
    else:
        frag_settings.specific.adf.fragments['frag2'] = emb_frag.kf.path + ' subfrag=active'
        frag_settings.specific.adf.fragments['frag1'] = frozen_frag.kf.path + ' subfrag=active type=FDE'
    return adf(settings.overlay(frag_settings), m_tot, job_name=jobname)


@schedule
def excitations_job(settings, iso1, iso2, emb_frag, frozen_frag, jobname, switch):
    exc_settings = Settings()
    if not switch:
        m_tot = add_fragments(iso1, iso2, switch) 
    else:
        m_tot = add_fragments(iso2, iso1, switch) 
    s_frag = exc_settings.specific.adf.fragments
    if not switch:
        s_frag['frag1'] = emb_frag.kf.path + ' subfrag=active'
        s_frag['frag2'] = frozen_frag.kf.path + ' subfrag=active type=FDE'
    else:
        s_frag['frag2'] = emb_frag.kf.path + ' subfrag=active'
        s_frag['frag1'] = frozen_frag.kf.path + ' subfrag=active type=FDE'
    exc_settings.specific.adf.excitations.onlysing = ""
    exc_settings.specific.adf.excitations.lowest = "5"
    return adf(settings.overlay(exc_settings), m_tot, job_name=jobname)


@schedule
def subexc_job(settings, iso_frag1, iso_frag2, emb_frag, frozen_frag, jobname, switch):
    subexc_settings = Settings()
    m_tot = add_fragments(iso_frag1, iso_frag2, switch)
    s_frag = subexc_settings.specific.adf.fragments
    s_frag['frag1'] = emb_frag.kf.path + ' subfrag=active'
    s_frag['frag2'] = frozen_frag.kf.path + ' subfrag=active type=FDE'
    subexc_settings.specific.adf.excitations.onlysing = ""
    subexc_settings.specific.adf.excitations.lowest = "5"
    subexc_settings.specific.adf.subexci.cthres = "10000.00"
    subexc_settings.specific.adf.subexci.sfthres = "0.00010000"
    subexc_settings.specific.adf.subexci.couplblock = ""
    subexc_settings.specific.adf.subexci.cicoupl = ""
    subexc_settings.specific.adf.subexci.tda = ""
    return adf(settings.overlay(subexc_settings), m_tot, job_name=jobname)


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    work_dir = args.w if args.w is not None else '.'

    return args.pdb, args.s, work_dir


if __name__ == "__main__":
    msg = " fde_qmworks -pdb <path/to/pdb>"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-pdb', required=True, help='path to PDB')
    parser.add_argument('-w', help='path to PDB')
    parser.add_argument('-s', help='index to split dimer', required=True,
                        type=int)

    main(*read_cmd_line(parser))
