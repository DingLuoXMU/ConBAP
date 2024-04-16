import os
import pandas as pd
import torch
from ConBAP import downstream_docking, ConBAP, downstream_affinity
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
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return pred 

def predict(data_root, graph_type, batch_size):

    data_graph = os.path.join(data_root)
    valid_df = os.listdir(data_root)
    valid_df = [os.path.join(data_root, i) for i in valid_df if 'csv' in i][0]
    valid_df = pd.read_csv(valid_df)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("used device:", device)
    
    model = ConBAP(35, 256).to(device)

    load_model_dict(model, f'../unsupervised/model/20240111_193210_ConBAP_repeat3/model/contrastive_no_filtered.pt')
    model_pose = downstream_docking(model, 256).to(device)
    model_pose = model_pose.to(device)
    valid_set = GraphDataset(data_dir=data_graph, data_df=valid_df, graph_type=graph_type, dis_threshold=8, create=False)
    valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=28)
    pred_pose = val(model_pose, valid_loader, device)


    model_affinity = downstream_affinity(model, 256).to(device)
    load_model_dict(model_affinity, f"./model/20231007_111336_ConBAP_repeat0/model/epoch-292, train_loss-0.1220, train_rmse-0.3493, valid_rmse-1.1663, valid_pr-0.7788.pt")
    model_affinity = model_affinity.to(device)
    
    pred_affinity = val(model_affinity, valid_loader, device)
    native_pose_list = valid_df['pdb'].values.tolist()
    print("pdbid","pose_score", "affinity")
    for i in range(len(native_pose_list)):
        
        print(native_pose_list[i], pred_pose[i], pred_affinity[i])


if __name__ == '__main__':

    data_root = './data/toy_set/' # data_dir for the toy set
    graph_type = 'ConBAP'
    batch_size = 1
    predict_type = 'pose'  # 'pose' or 'affinity'
    predict(data_root, graph_type, batch_size)
    









