# %%
import os
import pandas as pd
import numpy as np

import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from utils import *
import warnings
from threading import Lock
my_lock = Lock()
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# %%
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))


def get_edge_index(mol, graph):
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feats = bond_features(bond)
        graph.add_edge(i, j)
        edge_features.append(feats)
        edge_features.append(feats)

    return torch.stack(edge_features)

def bond_features(bond):
    # 返回一个[1,6]的张量，表示一键的各种信息是否存在
    bt = bond.GetBondType() # 获取键的类型
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(),bond.GetIsConjugated()]
    return torch.Tensor(fbond)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    edge_features = get_edge_index(mol, graph)
    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index, edge_features


def inter_graph(ligand, pocket, dis_threshold=5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformer().GetPositions()
    pos_p = pocket.GetConformer().GetPositions()

    # 添加配体-配体和口袋-口袋之间的边
    dis_matrix_l = distance_matrix(pos_l, pos_l)
    dis_matrix_p = distance_matrix(pos_p, pos_p)
    dis_matrix_lp = distance_matrix(pos_l, pos_p)

    node_idx_l = np.where(dis_matrix_l < dis_threshold)
    for i, j in zip(node_idx_l[0], node_idx_l[1]):
        graph_inter.add_edge(i, j, feats=torch.tensor([1, 0, 0, dis_matrix_l[i, j]]))

    node_idx_p = np.where(dis_matrix_p < dis_threshold)
    for i, j in zip(node_idx_p[0], node_idx_p[1]):
        graph_inter.add_edge(i + atom_num_l, j + atom_num_l, feats=torch.tensor([0, 1, 0, dis_matrix_p[i, j]]))

    node_idx_lp = np.where(dis_matrix_lp < dis_threshold)
    for i, j in zip(node_idx_lp[0], node_idx_lp[1]):
        graph_inter.add_edge(i, j + atom_num_l, feats=torch.tensor([0, 0, 1, dis_matrix_lp[i, j]]))

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v, _ in graph_inter.edges(data=True)]).T
    edge_attrs_inter = torch.stack([feats['feats'] for _, _, feats in graph_inter.edges(data=True)]).float()

    return edge_index_inter, edge_attrs_inter



# %%
def mols2graphs(complex_path, pdbid, label, save_path_l,save_path_p,save_path_aa,save_path_complex, pocket_dis = 5):
    print(pdbid)
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    x_l, edge_index_l,edge_features_l = mol2graph(ligand)
    x_p, edge_index_p,edge_features_p = mol2graph(pocket) 
    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_inter, edge_attrs_inter = inter_graph(ligand, pocket, dis_threshold=5)
    pos = torch.concat([pos_l, pos_p], dim=0)
    parser = PDBParser(QUIET=True)
    pocket_pdb = os.path.join(os.path.dirname(complex_path), f'Pocket_{pocket_dis}A.pdb')
    # protein_pdb = os.path.join(os.path.dirname(complex_path).split('graph_data')[0], "protein", f'{pdbid}_protein.pdb')
    protein_pdb = os.path.join(os.path.dirname(complex_path).split(f'{pdbid}')[0], "protein", f'{pdbid}_protein.pdb')
    s = parser.get_structure(pdbid, pocket_pdb)
    protein = parser.get_structure(pdbid, protein_pdb)    
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    pro_res_list = get_clean_res_list(protein.get_residues(), verbose=False, ensure_ca_exist=True)
    x_aa, seq, node_s, node_v, edge_index, edge_s, edge_v = get_protein_feature(res_list,pro_res_list, plm=False)
    


    c_size_l = atom_num_l
    c_size_p = atom_num_p
    c_size_aa = len(seq)
    c_size_complex = c_size_l + c_size_p
    y = torch.FloatTensor([label])
    split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
    data_l = Data(x=x_l, edge_index=edge_index_l,edge_attr=edge_features_l,y=y)
    data_l.__setitem__('c_size', torch.LongTensor([c_size_l]))
    data_p = Data(x=x_p, edge_index=edge_index_p,edge_attr=edge_features_p,y=y)
    data_p.__setitem__('c_size', torch.LongTensor([c_size_p]))
    data_aa = Data(x_aa=x_aa, seq=seq,
                node_s=node_s, node_v=node_v,
                edge_index=edge_index,
                edge_s=edge_s, edge_v=edge_v,
                y=y,
                )
    data_aa.__setitem__('c_size', torch.LongTensor([c_size_aa]))
    data_complex =Data(x=x, edge_attr=edge_attrs_inter, edge_index=edge_index_inter, y=y, pos=pos)
    data_complex.__setitem__('c_size', torch.LongTensor([c_size_complex]))  
    # if len(interaction_label.reshape([-1])) != data_aa.num_nodes*data_l.num_nodes:
    #     print("interaction_label",interaction_label.shape)
    #     print("data_aa.num_nodes",data_aa.num_nodes)
    #     raise ValueError("interaction_label.shape != data_aa.num_nodes*data_aa.num_nodes")
    torch.save(data_l, save_path_l)
    torch.save(data_p, save_path_p)
    torch.save(data_aa, save_path_aa)
    # torch.save(interaction_label, save_path_interaction_label)
    torch.save(data_complex, save_path_complex)
    # return data

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data,  collate_fn=data.collate_fn, **kwargs)#

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=8, graph_type='ConBAP', num_process=8, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))
        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_l_list = []
        graph_path_p_list = []
        graph_path_aa_list = []
        graph_path_complex_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdb'], float(row['affinity'])
            if type(cid) != str:
                cid = str(int(cid))
            complex_dir = os.path.join(data_dir, cid)
            graph_path_l = os.path.join(complex_dir, f"{graph_type}-{cid}_l_{self.dis_threshold}A.pyg")
            graph_path_p = os.path.join(complex_dir, f"{graph_type}-{cid}_p_{self.dis_threshold}A.pyg")
            graph_path_aa = os.path.join(complex_dir, f"{graph_type}-{cid}_aa_{self.dis_threshold}A.pyg")
            graph_path_complex = os.path.join(complex_dir, f"{graph_type}-{cid}_complex_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_l_list.append(graph_path_l)
            graph_path_p_list.append(graph_path_p)
            graph_path_aa_list.append(graph_path_aa)
            graph_path_complex_list.append(graph_path_complex)

        if self.create:
            print('Generate complex graph...')
            #multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, complex_id_list, pKa_list, graph_path_l_list, graph_path_p_list,graph_path_aa_list,graph_path_complex_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths_l = graph_path_l_list
        self.graph_paths_p = graph_path_p_list
        self.graph_paths_aa = graph_path_aa_list
        self.complex_ids = complex_id_list
        self.graph_paths_complex = graph_path_complex_list

    def __getitem__(self, idx):
        
        
        return torch.load(self.graph_paths_l[idx]), torch.load(self.graph_paths_p[idx]), torch.load(self.graph_paths_aa[idx]) \
        ,torch.load(self.graph_paths_complex[idx])
            
    

    def collate_fn(self, data_list):
        batchA = Batch.from_data_list([data[0] for data in data_list])
        batchB = Batch.from_data_list([data[1] for data in data_list])
        batchC = Batch.from_data_list([data[2] for data in data_list])
        batch_complex =  Batch.from_data_list([data[3] for data in data_list])


        
        lig_scope = []
        atom_pocket_scope = []
        amino_acid_scope = []
        complex_scope = []
        start_atom = 0
        start_atom_pocket = 0
        start_amino_acid = 0
        start_complex = 0

        for i in range(len(batchA)):
            graphA = batchA[i]
            graphB = batchB[i]
            graphC = batchC[i]
            graph_complex = batch_complex[i]
            # print(data_list[0][2])
            atom_count_A = graphA.num_nodes
            atom_count_B = graphB.num_nodes
            atom_count_C = graphC.num_nodes
            atom_count_complex = graph_complex.num_nodes

            
                
            lig_scope.append((start_atom, atom_count_A))
            atom_pocket_scope.append((start_atom_pocket, atom_count_B))
            amino_acid_scope.append((start_amino_acid, atom_count_C))
            complex_scope.append((start_complex, atom_count_complex))

            start_atom += atom_count_A
            start_atom_pocket += atom_count_B
            start_amino_acid += atom_count_C
            start_complex += atom_count_complex

        batch = {'ligand_features': batchA, 'atom_pocket_features': batchB, 'amino_acid_features': batchC, 
                    'lig_scope': lig_scope, 'atom_pocket_scope': atom_pocket_scope, 'amino_acid_scope': amino_acid_scope,
                    'complex_scope': complex_scope, 'complex_features': batch_complex}

        return batch

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':
    data_root = './data'
    
    data_dir = os.path.join(data_root, 'toy_set')
    data_df = pd.read_csv(os.path.join(data_root, 'toy_set/toy_set.csv'))
    
    # # three hours
    toy_set = GraphDataset(data_dir, data_df, graph_type='ConBAP', dis_threshold=8, create=True)
    print('finish!')

    


# %%
