
# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pandas as pd
import torch
from ConBAP import downstream_affinity,ConBAP, downstream_docking
from dataset_ConBAP import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error

# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        
        with torch.no_grad():
            data['ligand_features'] = data['ligand_features'].to(device)
            data['complex_features'] = data['complex_features'].to(device)
            pred = model(data)
            label = data["ligand_features"].y
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, coff, pred, label
    
# %%

graph_type = 'ConBAP'
batch_size = 1



# data_dir = os.path.join('./data/pdbbind/v2020-other-PL')
# valid_df = pd.read_csv(os.path.join('./data/pdbbind/val_new.csv'))
# test2016_df = pd.read_csv(os.path.join('./data/pdbbind/data_test.csv'))

data_dir = os.path.join('./data/toy_set/')
test_df = pd.read_csv(os.path.join('./data/toy_set/toy_set.csv'))








test_set = GraphDataset(data_dir, test_df, graph_type='ConBAP', dis_threshold=8, create=False)
test_loader = PLIDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)



device = torch.device('cuda:0')

model = ConBAP(35, 256).to(device)
model = downstream_affinity(model, 256).to(device)
load_model_dict(model, f"./model/20231009_182510_ConBAP_repeat2/model/epoch-538, train_loss-0.0763, train_rmse-0.2761, valid_rmse-1.1418, valid_pr-0.7890.pt")
model = model.cuda()



test_rmse, test_coff,pred_crystal, label = val(model, test_loader, device)




msg = "test_rmse-%.4f, test_r-%.4f" \
    % (test_rmse, test_coff)


    
print(msg)
# %%
