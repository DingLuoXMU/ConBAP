# %%
import os
# os.environ['CUDA_VISIBLE_DlrEVICES'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from ConBAP import ConBAP
from dataset_ConBAP import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from torch.nn import TripletMarginLoss
from margin import MarginScheduledLossFunction
# %%
from sklearn.metrics import accuracy_score


def val(model, dataloader, device):
    model.eval()

    loss_list = []
    pred_anchor_list = []
    pred_negative_list = []
    label_anchor_list = []
    label_negative_list = []
    # acc_list = []
    for data in dataloader:
        data['ligand_features'] = data['ligand_features'].to(device)
        data['amino_acid_features'] = data['amino_acid_features'].to(device)
        data["native_complex_features"] = data["native_complex_features"].to(device)
        data["redocked_complex_features"] = data["redocked_complex_features"].to(device)
        affinity_anchor = data["native_complex_features"].y
        affinity_positive = data["redocked_complex_features"].y
         
        with torch.no_grad():
            anchor, positive, negative, affinity_anchor_pred, affinity_negative_pred = model(data)
            

            mask = affinity_anchor != 0      #去掉0的值
            affinity_anchor = affinity_anchor[mask]
            affinity_anchor_pred = affinity_anchor_pred[mask]
            pred_anchor_list.append(affinity_anchor_pred.detach().cpu().numpy())
            pred_negative_list.append(affinity_negative_pred.detach().cpu().numpy())
            label_anchor_list.append(affinity_anchor.detach().cpu().numpy())
            label_negative_list.append(affinity_positive.detach().cpu().numpy())

            pred_anchor = np.concatenate(pred_anchor_list, axis=0)
            pred_negative = np.concatenate(pred_negative_list, axis=0)
            label_anchor = np.concatenate(label_anchor_list, axis=0)
            label_negative = np.concatenate(label_negative_list, axis=0)

            coff_anchor = np.corrcoef(pred_anchor, label_anchor)[0,1]
            coff_negative = np.corrcoef(pred_negative, label_negative)[0,1]
            rmse_anchor = np.sqrt(mean_squared_error(pred_anchor, label_anchor))
            rmse_negative = np.sqrt(mean_squared_error(pred_negative, label_negative))


    return coff_anchor, coff_negative, rmse_anchor, rmse_negative
        


# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_ConBAP'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    only_affinity = args.get("only_affinity")
    margin = args.get("margin")
    margin_t0 = args.get("margin_t0")
    margin_fn =args.get("margin_fn")
    print(data_root)
    # for repeat in range(repeats):
    #     args['repeat'] = repeat

    # data_dir = os.path.join(data_root, 'v2020-other-PL')
    # valid_dir = os.path.join(data_root, 'valid')
    # test2016_dir = os.path.join(data_root, 'test2016')
    data_dir = os.path.join('../redock_dataset')

    train_df = pd.read_csv(os.path.join(data_root, 'data_train_finall_label.csv'))
    valid_df = pd.read_csv(os.path.join(data_root, 'data_test_finall_label.csv'))

    train_data_pair = pd.read_csv(os.path.join(data_root, 'data_index_8A_train.csv'))
    valid_data_pair = pd.read_csv(os.path.join(data_root, 'data_index_8A_test.csv'))
    # test2016_df = pd.read_csv(os.path.join(data_root, 'data_test.csv'))

    train_set = GraphDataset(data_dir, train_df, data_pair=train_data_pair, graph_type='ConBAP', dis_threshold=8,read_csv=True, create=False)
    valid_set = GraphDataset(data_dir, valid_df, data_pair=valid_data_pair, graph_type='ConBAP', dis_threshold=8,read_csv=True, create=False)

    # test2016_set = GraphDataset(data_dir, test2016_df, dis_threshold=5,graph_type=graph_type, create=False)
    
    train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    
    
    valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)
    # test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=16)


    logger = TrainLogger(args, cfg, create=True)
    logger.info(__file__)
    logger.info(f"train data: {len(train_set)}")
    logger.info(f"valid data: {len(valid_set)}")

    
    device = torch.device('cuda:0')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ConBAP(35, 256).to(device)
    

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model = MGNN(35, 256)
    
    # if torch.cuda.device_count() > 1: # check if more than one GPU is available
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model) # use multiple GPUs
    model = model.to(device)
   
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)#  lr=5e-4, weight_decay=1e-6
    criterion_affinity = nn.MSELoss()
    
    criterion = MarginScheduledLossFunction(M_0=margin,N_epoch=epochs,N_restart=margin_t0,update_fn=margin_fn)
    running_loss = AverageMeter()
    running_acc = AverageMeter()
    running_best_mse = BestMeter("min")
    best_model_list = []
    
    model.train()
    for epoch in range(epochs):
        
        for data in train_loader:
            data['ligand_features'] = data['ligand_features'].to(device)
            data['amino_acid_features'] = data['amino_acid_features'].to(device)
            data["native_complex_features"] = data["native_complex_features"].to(device)
            data["redocked_complex_features"] = data["redocked_complex_features"].to(device)
            affinity_anchor = data["native_complex_features"].y
            # print(affinity_anchor)
            affinity_negative = data["redocked_complex_features"].y
            anchor, positive, negative, affinity_anchor_pred, affinity_negative_pred = model(data)
            
            # for i, j in zip(affinity_anchor_pred, affinity_anchor):
            #     if j == 0:
            #         affinity_anchor.remove(j)
            #         affinity_anchor_pred.remove(i)
            mask = affinity_anchor != -0.0000      #去掉0的值
            # print(mask)
            affinity_anchor = affinity_anchor[mask]
            affinity_anchor_pred = affinity_anchor_pred[mask]
            # print(affinity_anchor_pred)
            # print(affinity_anchor)
            # print(affinity_positive)
            loss_affinity_anchor = criterion_affinity(affinity_anchor_pred, affinity_anchor)        
            loss_affinity_negative = criterion_affinity(affinity_negative_pred, affinity_negative)
          
            loss_triple = criterion(anchor, positive, negative)
            loss = loss_triple + loss_affinity_anchor + loss_affinity_negative#  
            loss =loss.to(device)  #
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss_affinity_anchor.item(),affinity_anchor.size(0))   # batch_size

        epoch_loss = running_loss.get_average()
        epoch_rmse = np.sqrt(epoch_loss)   
        running_loss.reset()

        # start vali
        coff_anchor, coff_negative, rmse_anchor, rmse_negative = val(model, valid_loader, device)
        msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f, valid_rmse_decoy-%.4f, valid_pr_decoy-%.4f" \
                % (epoch, epoch_loss, epoch_rmse, rmse_anchor,coff_anchor, rmse_negative,coff_negative)
        logger.info(msg)
        # if rmse_anchor < running_best_mse.get_best():
        #     running_best_mse.update(rmse_anchor)
        if save_model:
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f, valid_rmse_decoy-%.4f, valid_pr_decoy-%.4f " \
            % (epoch, epoch_loss, epoch_rmse, rmse_anchor,coff_anchor, rmse_negative,coff_negative)
            model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
            best_model_list.append(model_path)
            save_model_dict(model, logger.get_model_dir(), msg)


        

# %%