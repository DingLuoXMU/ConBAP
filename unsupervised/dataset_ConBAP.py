# %%
import os
import pandas as pd
import numpy as np

import pickle
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
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
from feature_utils import *
import warnings
from threading import Lock
import gvp
my_lock = Lock()
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')
atom_radius = {"N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1, "F": 1.5, "Cl": 1.8,
               "Br": 2.0, "I": 2.2, "C": 1.9, "H": 0.0, "Zn": 0.5, "B": 1.8,"metal": 1.2}#, "Na": 2.2, "K": 2.8, "Mg": 1.7, "Fe": 2.23,"V":2.3}
dir_lig_pqr = "./data/pdbbind/pqr_lig"
dir_pdb_pqr = "./data/pdbbind/pdb_pqr"
# %%
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]
def is_matal(x):
    if x in ['Zn', 'Na', 'K', 'Mg', 'Fe', 'V','Se','As','Sb','Te','Po','At','Rn',"Si"
             ,'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md',
             'No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og',"Pt"]:
        return True
    else:
        return False

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def check_common_elements(list1, list2):
    if len(list1) != len(list2):
        return False
    count_almost_equal = 0
    count_almost_equal1 = 0
    for x, y in zip(list1, list2):
        if x[:-2] == y[:-2]:
            count_almost_equal += 1
        if x == y:
            count_almost_equal1 += 1
        if count_almost_equal >= 3 and count_almost_equal1 >= 2:
            return True
    return False

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
def mols2graphs(complex_path_native,complex_path_redocked, pdbid, native_pose_name, graph_path_l, graph_path_aa, graph_path_complex_native, graph_path_complex_redocked, native_pose_label, redocked_pose_label, dis_threshold=5.):
    print(pdbid)

    # for native pose
    with open(complex_path_native, 'rb') as f:
        ligand_native, pocket_native = pickle.load(f)
        if ligand_native is None or pocket_native is None:
            print("ligand_native is None or pocket_native is None:",complex_path_native)
            return

    atom_num_l_native = ligand_native.GetNumAtoms()
    atom_num_p_native = pocket_native.GetNumAtoms()
    

    x_l_native, edge_index_l_native,edge_features_l_native = mol2graph(ligand_native)
    x_p_native, edge_index_p_native,edge_features_p_native = mol2graph(pocket_native) 
    pos_l_native = torch.FloatTensor(ligand_native.GetConformers()[0].GetPositions())
    pos_p_native = torch.FloatTensor(pocket_native.GetConformers()[0].GetPositions())
    x_native = torch.cat([x_l_native, x_p_native], dim=0)
    edge_index_inter_native, edge_attrs_inter_native = inter_graph(ligand_native, pocket_native, 5) # dis_threshold=5
    pos_native = torch.concat([pos_l_native, pos_p_native], dim=0)
    y_native = torch.FloatTensor([native_pose_label])
    # for redocked pose
    with open(complex_path_redocked, 'rb') as f:
        ligand_redocked, pocket_redocked = pickle.load(f)
        if ligand_redocked is None or pocket_redocked is None:
            print("ligand_redocked is None or pocket_redocked is None:",complex_path_redocked)
            return
    
    atom_num_l_redocked = ligand_redocked.GetNumAtoms()
    atom_num_p_redocked = pocket_redocked.GetNumAtoms()
    
    x_l_redocked, edge_index_l_redocked,edge_features_l_redocked = mol2graph(ligand_redocked)
    x_p_redocked, edge_index_p_redocked,edge_features_p_redocked = mol2graph(pocket_redocked)
    pos_l_redocked = torch.FloatTensor(ligand_redocked.GetConformers()[0].GetPositions())
    pos_p_redocked = torch.FloatTensor(pocket_redocked.GetConformers()[0].GetPositions())
    x_redocked = torch.cat([x_l_redocked, x_p_redocked], dim=0)
    edge_index_inter_redocked, edge_attrs_inter_redocked = inter_graph(ligand_redocked, pocket_redocked, 5)  # dis_threshold=5
    pos_redocked = torch.concat([pos_l_redocked, pos_p_redocked], dim=0)
    y_redocked = torch.FloatTensor([redocked_pose_label])

    # for aa

    # split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
    parser = PDBParser(QUIET=True)
    if len(native_pose_name.split("_")) < 5:
        pocket_pdb = f"../redocked_complex/{pdbid}/Pocket_{native_pose_name}_native_{dis_threshold}A.pdb"
    else:
        pocket_pdb = complex_path_native.replace(f"complex_{dis_threshold}A.rdkit",f"Pocket_redocked_{dis_threshold}A.pdb")
    
    protein_pdb = f"../redocked_complex/{pdbid}/Protein_{native_pose_name}_native_{dis_threshold}A.pdb"
        



    s = parser.get_structure(pdbid, pocket_pdb)
    # protein = parser.get_structure(pdbid, protein_pdb)    
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    # pro_res_list = get_clean_res_list(protein.get_residues(), verbose=False, ensure_ca_exist=True)
    # print("res_list",len(res_list))
    # interaction_label,len_coords_alpha_c = get_interaction_label(ligand,res_list, dis_threshold=5.)
    # 不使用pro_res_list，处理数据时则不会使用esm模型
    x_aa, seq, node_s, node_v, edge_index, edge_s, edge_v = get_protein_feature(res_list, plm=False)
    
    # if atom_num_l_native != atom_num_l_redocked:
    #     print("atom_num_l_native != atom_num_l_redocked:",complex_path_redocked)
    #     raise ValueError("atom_num_l_native != atom_num_l_redocked")

    c_size_l = atom_num_l_native

    c_size_aa = len(seq)
    c_size_complex_native = atom_num_l_native + atom_num_p_native
    c_size_complex_redocked = atom_num_l_redocked + atom_num_p_redocked


    
# save data_l
# if not os.path.exists(graph_path_l):            
    data_l = Data(x=x_l_native, edge_index=edge_index_l_native,edge_attr=edge_features_l_native)
    data_l.__setitem__('c_size', torch.LongTensor([c_size_l]))
    torch.save(data_l, graph_path_l)
# save data_native
# if not os.path.exists(graph_path_complex_native):            
    data_native = Data(x=x_native, edge_attr=edge_attrs_inter_native, edge_index=edge_index_inter_native, y=y_native, pos=pos_native)
    data_native.__setitem__('c_size', torch.LongTensor([c_size_complex_native]))
    torch.save(data_native, graph_path_complex_native)
# save data_redocked
# if not os.path.exists(graph_path_complex_redocked):
    data_redocked = Data(x=x_redocked, edge_attr=edge_attrs_inter_redocked, edge_index=edge_index_inter_redocked, y=y_redocked, pos=pos_redocked)
    data_redocked.__setitem__('c_size', torch.LongTensor([c_size_complex_redocked]))
    torch.save(data_redocked, graph_path_complex_redocked)

# save data_aa
# if not os.path.exists(graph_path_aa):
    data_aa = Data(x_aa=x_aa, seq=seq,
                node_s=node_s, node_v=node_v,
                edge_index=edge_index,
                edge_s=edge_s, edge_v=edge_v,                    
                )
    data_aa.__setitem__('c_size', torch.LongTensor([c_size_aa]))
    torch.save(data_aa, graph_path_aa)

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data,  collate_fn=data.collate_fn, **kwargs)#

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, data_pair, dis_threshold=5, graph_type='ConBAP', num_process=32, read_csv=True, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.read_csv = read_csv
        self.data_pair = data_pair
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()
        
        

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        data_pair = self.data_pair
        # read_csv = self.read_csv
        dis_thresholds = repeat(self.dis_threshold, len(data_df))
        complex_path_native_list = []
        complex_path_redocked_list = []
        native_pose_name_list = []
        complex_id_list = []        
        graph_path_l_list = []        
        graph_path_aa_list = []

        graph_path_complex_native_list = []
        graph_path_complex_redocked_list = []

        native_pose_label_list = []
        redocked_pose_label_list = []
        
        if self.read_csv:
            complex_path_native_list = data_pair['complex_path_native'].tolist()
            complex_path_redocked_list = data_pair['complex_path_redocked'].tolist()

            native_pose_name_list = data_pair['native_pose_name'].tolist()
            complex_id_list = data_pair['complex_id'].tolist()
            graph_path_l_list = data_pair['graph_path_l'].tolist()
            graph_path_aa_list = data_pair['graph_path_aa'].tolist()

            graph_path_complex_native_list = data_pair['graph_path_complex_native'].tolist()
            graph_path_complex_redocked_list = data_pair['graph_path_complex_redocked'].tolist()

            native_pose_label_list = data_pair['native_vina_label'].tolist()
            redocked_pose_label_list = data_pair['redocked_vina_label'].tolist()            
        else:
            for i, row in data_df.iterrows():
                
                label = row['label']
                pose_name = (row['Ligand'].split('/')[1]).split('.')[0]          # 5f74_A_rec_5f74_amp_lig_tt_docked_0
                native_pose_name = extract_parts(pose_name)                    # 5f74_amp
                receptor_name = ((row['Receptor'].split('/')[1]).split('.')[0])[:-2]  # 5f74_A_rec

                complex_dir_redocked = os.path.join(data_dir, row['Receptor'].split('/')[0])    # redock_dataset/1433B_HUMAN_1_240_pep_0
                complex_dir_native = os.path.join("../redocked_complex", f"{receptor_name}_{native_pose_name}")    # ../redocked_complex/5f74_A_rec_5f74_amp

                graph_path_l = os.path.join(complex_dir_native,f"{graph_type}-{native_pose_name}_l_{self.dis_threshold}A.pyg")            
                graph_path_aa = os.path.join(complex_dir_native, f"{graph_type}-{receptor_name}_aa_{self.dis_threshold}A.pyg")

                complex_id = f"{receptor_name}_{native_pose_name}"

                if label == 1:
                    native_pose_name = pose_name
                    graph_path_complex_native = os.path.join(complex_dir_redocked,pose_name ,f"{graph_type}-{pose_name}_complex_redocked_{self.dis_threshold}A.pyg")
                    # print(pose_name)
                    pose_names_rows = data_df[(data_df['label'] == 0) & ((data_df['Ligand'].apply(lambda x: (x.split('/')[1]).split('.')[0].split('_lig')[0])) == pose_name.split('_lig')[0]) \
                    & ((data_df['Receptor'].apply(lambda x: (x.split('/')[0]))) == row['Receptor'].split('/')[0])]
                    if len(pose_names_rows) == 0:
                        print('no negative pose for pose', pose_name)
                        continue
                    # random_pose_name = (pose_names_rows.sample(1)['Ligand']).values[0].split('/')[1].split('.')[0]
                    random_pose = pose_names_rows.sample(1)
                    random_pose_name = (random_pose['Ligand']).values[0].split('/')[1].split('.')[0]
                    
                    native_pose_label = row['pK']
                    


                    # native_pose_label = row['pK']
                    redocked_vina_label = random_pose['vina_label'].values[0]

                    graph_path_complex_redocked = os.path.join(complex_dir_redocked, random_pose_name, f"{graph_type}-{random_pose_name}_complex_redocked_{self.dis_threshold}A.pyg")


                    complex_path_native = os.path.join(complex_dir_redocked,pose_name, f"complex_{self.dis_threshold}A.rdkit")
                    complex_path_redocked = os.path.join(complex_dir_redocked, random_pose_name, f"complex_{self.dis_threshold}A.rdkit")


                else:
                    graph_path_complex_native = os.path.join(complex_dir_native, f"{graph_type}-{native_pose_name}_complex_native_{self.dis_threshold}A.pyg")
                    graph_path_complex_redocked = os.path.join(complex_dir_redocked,pose_name ,f"{graph_type}-{pose_name}_complex_redocked_{self.dis_threshold}A.pyg")
                    
                    complex_path_native = os.path.join(complex_dir_native, f"{native_pose_name}_{self.dis_threshold}A.rdkit")
                    complex_path_redocked = os.path.join(complex_dir_redocked, pose_name, f"complex_{self.dis_threshold}A.rdkit")

                    native_pose_label = -row['pK']
                    redocked_pose_label = row['vina_label']

                complex_path_native_list.append(complex_path_native)
                complex_path_redocked_list.append(complex_path_redocked)

                complex_id_list.append(complex_id)
                native_pose_name_list.append(native_pose_name)

                graph_path_l_list.append(graph_path_l)        
                graph_path_aa_list.append(graph_path_aa)

                graph_path_complex_native_list.append(graph_path_complex_native)
                graph_path_complex_redocked_list.append(graph_path_complex_redocked)

                native_pose_label_list.append(native_pose_label)
                redocked_pose_label_list.append(redocked_pose_label)
        if self.create:
            print('Generate complex graph...')
            #multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_native_list,complex_path_redocked_list, complex_id_list, native_pose_name_list, graph_path_l_list, graph_path_aa_list, graph_path_complex_native_list, graph_path_complex_redocked_list, native_pose_label_list, redocked_pose_label_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths_l = graph_path_l_list
        self.graph_paths_aa = graph_path_aa_list
        self.graph_paths_native = graph_path_complex_native_list
        self.graph_paths_redocked = graph_path_complex_redocked_list




    def __getitem__(self, idx):
        # print(len(self.graph_paths_l),len(self.graph_paths_aa),len(self.graph_paths_native),len(self.graph_paths_redocked))
        return torch.load(self.graph_paths_l[idx]), torch.load(self.graph_paths_aa[idx]), torch.load(self.graph_paths_native[idx]) ,torch.load(self.graph_paths_redocked[idx])
        
            
    

    def collate_fn(self, data_list):
        batch_lig = Batch.from_data_list([data[0] for data in data_list])
        batch_aa = Batch.from_data_list([data[1] for data in data_list])
        batch_native = Batch.from_data_list([data[2] for data in data_list])
        batch_redocked =  Batch.from_data_list([data[3] for data in data_list])


        
        lig_scope = []
        amino_acid_scope = []
        complex_scope = []
        start_atom = 0
        start_amino_acid = 0


        for i in range(len(batch_lig)):
            graphA = batch_lig[i]
            graphB = batch_aa[i]
            # print(data_list[0][2])
            atom_count_A = graphA.num_nodes
            atom_count_B = graphB.num_nodes

            
                
            lig_scope.append((start_atom, atom_count_A))
            amino_acid_scope.append((start_amino_acid, atom_count_B))
            

            start_atom += atom_count_A
            start_amino_acid += atom_count_B
        

        batch = {'ligand_features': batch_lig, 'amino_acid_features': batch_aa, 
                    'lig_scope': lig_scope, 'amino_acid_scope': amino_acid_scope,
                    'native_complex_features': batch_native, 'redocked_complex_features': batch_redocked}

        return batch

    def __len__(self):
        return len(self.graph_paths_l)

            
if __name__ == '__main__':
    data_root = './'
    # toy_dir = os.path.join(data_root, 'pdbbind')
    data_dir = os.path.join(data_root, './redock_dataset')
    data_df = pd.read_csv(os.path.join(data_root, 'data_test_finall_label.csv'))
    data_pair = pd.read_csv(os.path.join(data_root, 'data_index_8A_test.csv'))
    # three hours
    toy_set = GraphDataset(data_dir, data_df, data_pair, graph_type='ConBAP', dis_threshold=8, read_csv=True, create=True)
    print('finish!')
    # for casf_docking

    # pdbids = os.listdir("./data/CASF-2016/graph_data")
    # # pdbids =["1a30"]
 
    # for pdb in pdbids:
    #     data_root = f'./data/CASF-2016/graph_data/{pdb}'
    #     data_df = pd.read_csv(os.path.join(data_root, f'{pdb}.csv'))
    #     valid_set = GraphDataset(data_root, data_df, graph_type='Graph_GIGN', dis_threshold=8, create=True)
    # print('finish!')
    # for casf_screening
    # pdbids = os.listdir("./data/CASF-2016/data_screening")
    # pdbids = ["3dd0"]
    # # pdbids =["1a30"]
    # for pdb in pdbids:
    #     data_root = f'./data/CASF-2016/data_screening/{pdb}'
    #     data_df = pd.read_csv(os.path.join(data_root, f'{pdb}.csv'))
    #     valid_set = GraphDataset(data_root, data_df, graph_type='Graph_GIGN', dis_threshold=8, create=True)

    


# %%
