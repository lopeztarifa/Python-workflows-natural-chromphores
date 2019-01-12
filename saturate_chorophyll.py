#!/usr/bin/env python
from rdkit import Chem
from rdkit.Chem import AllChem
import os
 
template = Chem.MolFromSmiles('CCC1=C(C2=NC1=CC3=C(C4=C([N-]3)C(=C5[C@H]([C@@H](C(=N5)C=C6C(=C(C(=C2)[N-]6)C=C)C)C)CCC(=O)OC/C=C(\C)/CCCC(C)CCCC(C)CCCC(C)C)[C@H](C4=O)C(=O)OC)C)C.[Mg+2]')
 
for filename in os.listdir("."):
    if filename[-4:] == ".pdb":
        mol = Chem.MolFromPDBFile(filename,sanitize=False)
        bs=[]
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "Mg":
                atom.SetNoImplicit(True)
                for bond in atom.GetBonds():
                    bs.append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        emol = Chem.EditableMol(mol)
        for b in bs:
            emol.RemoveBond(b[0],b[1])
        newmol = emol.GetMol()
        print Chem.MolToSmiles(newmol)
        newmol=Chem.AllChem.AssignBondOrdersFromTemplate(template,newmol)
        molHs=Chem.AddHs(newmol,addCoords=True)
        open(filename[:-4]+'-addedHs.mol','w').write(Chem.MolToMolBlock(molHs))
