import os
import pandas as pd
import torch
from ConBAP import downstream_docking, ConBAP
from dataset_ConBAP import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        with torch.no_grad():
            data['ligand_features'] = data['ligand_features'].to(device)
            data['atom_pocket_features'] = data['atom_pocket_features'].to(device)
            data['amino_acid_features'] = data['amino_acid_features'].to(device)
            data['complex_features'] = data['complex_features'].to(device)
            pred = model(data)
            label = data["ligand_features"].y
            
            # label = data.y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    # pred = np.concatenate(pred_list, axis=0)
    # pred = np.squeeze(pred) # (N, 1) -> (N, )
    # label = np.concatenate(label_list, axis=0)
    # # print(pred.shape)
    # # print(label.shape)


    # coff = np.corrcoef(pred, label)[0, 1]
    # rmse = np.sqrt(mean_squared_error(label, pred))

    # model.train()

    # return rmse, coff
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return pred 


data_root = './data/CASF-2016/data_screening'
graph_type = 'ConBAP'
batch_size = 1
# data_dir = os.path.join('../redock_dataset')
# valid_df = pd.read_csv(os.path.join(data_root, 'data_index_8A_test_vina-score.csv'))
# valid_data_pair = pd.read_csv(os.path.join(data_root, 'data_index_8A_test_vina-score.csv'))
# valid_set = GraphDataset(data_dir, valid_df, valid_data_pair, graph_type='Graph_GIGN', dis_threshold=8, create=False, read_saved_datapair=True)
# valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device('cuda:0')
model = ConBAP(35, 256).to(device)

load_model_dict(model, f'../unsupervised/model/20230817_110144_ConBAP_repeat3/model/contrastive_1.pt')

model = downstream_docking(model, 256).to(device)
model = model.to(device)


# test for screening power
os.makedirs(f"./data/CASF-2016/power_screening/examples/test", exist_ok=True)
pdbids = os.listdir("./data/CASF-2016/data_screening")




for pdbid in pdbids:
    print("processing", pdbid)
    data_dir = os.path.join('./data/CASF-2016/data_screening', pdbid)
    valid_df = pd.read_csv(os.path.join(data_root, pdbid, f'{pdbid}.csv'))
    valid_set = GraphDataset(data_dir, valid_df, graph_type='ConBAP', dis_threshold=8, create=False)
    valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=28)




    pred = val(model, valid_loader, device)
    pred = pred.reshape(-1, 1)
    with open(f"./data/CASF-2016/power_screening/examples/test/{pdbid}_score.dat", 'w') as f:
        f.write("#code\tscore\n")
        poses = valid_df['pdb'].values.tolist()
        for pose, score in zip(poses, pred):
            f.write(f"{pose}\t{score[0]}\n") 
            
print("done")


