import os
import pickle
import torch
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import rdBase
from tqdm import tqdm
import glob
import torch
import torch.nn.functional as F
from io import StringIO
import sys
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select
import scipy
import scipy.spatial
from rdkit.Geometry import Point3D
import gvp
import gvp.data
def read_mol(sdf_fileName, mol2_fileName, verbose=False):
    rdBase.LogToPythonStderr()
    stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        sm = str(e)
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        problem = False
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
        except Exception as e:
            sm = str(e)
            problem = True

    if verbose:
        print(sio.getvalue())
    sys.stderr = stderr
    return mol, problem


def write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName):
    # read in mol
    mol, _ = read_mol(sdf_fileName, mol2_fileName)
    # reorder the mol atom number as in smiles.
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)
    w = Chem.SDWriter(toFile)
    w.write(mol)
    w.close()
def read_pdbbind_data(fileName):
    with open(fileName) as f:
        a = f.readlines()
    info = []
    for line in a:
        if line[0] == '#':
            continue
        lines, ligand = line.split('//')
        pdb, resolution, year, affinity, raw = lines.strip().split('  ')
        ligand = ligand.strip().split('(')[1].split(')')[0]
        # print(lines, ligand)
        info.append([pdb, resolution, year, affinity, raw, ligand])
    info = pd.DataFrame(info, columns=['pdb', 'resolution', 'year', 'affinity', 'raw', 'ligand'])
    info.year = info.year.astype(int)
    info.affinity = info.affinity.astype(float)
    return info
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_protein_feature(res_list, pro_res_list=None, plm=True):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    ca= []
    # print('get_protein_feature!')
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            if atom == res['CA']:
                ca.append(list(atom.coord))
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = gvp.data.ProteinGraphDataset([structure])
    protein = dataset[0]
    if plm and pro_res_list:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval()#.to(device)
        # res_list = res_list.to(device)
        # pro_res_list = pro_res_list.to(device)
        batch_convert = alphabet.get_batch_converter()
        protein_ca, index = pocket_in_protein(res_list, pro_res_list)
        token_reps = get_plm_reps(protein_ca, model, batch_convert)
        pocket_token_reps = token_reps[index].to("cpu")
        protein.seq = pocket_token_reps
        # print('pass!')
    return protein.x, protein.seq, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v

def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list

def remove_hetero_and_extract_ligand(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    # get all regular protein residues. and ligand.
    clean_res_list = []
    ligand_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if (not ensure_ca_exist) or ('CA' in res):
                # in rare case, CA is not exists.
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        elif hetero == 'W':
            # is water, skipped.
            continue
        else:
            ligand_list.append(res)
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list, ligand_list

def get_res_unique_id(residue):
    pdb, _, chain, (_, resid, insertion) = residue.full_id
    unique_id = f"{chain}_{resid}_{insertion}"
    return unique_id

def save_cleaned_protein(c, proteinFile):
    res_list = list(c.get_residues())
    clean_res_list, ligand_list = remove_hetero_and_extract_ligand(res_list)
    res_id_list = set([get_res_unique_id(residue) for residue in clean_res_list])

    io=PDBIO()
    class MySelect(Select):
        def accept_residue(self, residue, res_id_list=res_id_list):
            if get_res_unique_id(residue) in res_id_list:
                return True
            else:
                return False
    io.set_structure(c)
    io.save(proteinFile, MySelect())
    return clean_res_list, ligand_list

def split_protein_and_ligand(c, pdb, ligand_seq_id, proteinFile, ligandFile):
    clean_res_list, ligand_list = save_cleaned_protein(c, proteinFile)
    chain = c.id
    # should take a look of this ligand_list to ensure we choose the right ligand.
    seq_id = ligand_seq_id
    # download the ligand in sdf format from rcsb.org. because we pdb format doesn't contain bond information.
    # you could also use openbabel to do this.
    url = f"https://models.rcsb.org/v1/{pdb}/ligand?auth_asym_id={chain}&auth_seq_id={seq_id}&encoding=sdf&filename=ligand.sdf"
    r = requests.get(url)
    open(ligandFile , 'wb').write(r.content)
    return clean_res_list, ligand_list




def generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=30, fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit = generate_conformation(mol_from_rdkit)
    coords = mol_from_rdkit.GetConformer().GetPositions()
    new_coords = coords + np.array([shift_dis, shift_dis, shift_dis])
    write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)


    # save protein chains that belong to chains_in_contact
    class MySelect(Select):
        def accept_residue(self, residue, chains_in_contact=chains_in_contact):
            pdb, _, chain, (_, resid, insertion) = residue.full_id
            if chain in chains_in_contact:
                return True
            else:
                return False

    io=PDBIO()
    io.set_structure(s)
    io.save(toFile, MySelect())

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj



class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg
