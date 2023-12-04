
# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from ConBAP import ConBAP, downstream_affinity
from dataset_ConBAP import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
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
                # data['atom_pocket_features'] = data['atom_pocket_features'].to(device)
                # data['amino_acid_features'] = data['amino_acid_features'].to(device)
            data['complex_features'] = data['complex_features'].to(device)

            label = data["ligand_features"].y
            pred = model(data)
            

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, coff

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

    for repeat in range(repeats):
        args['repeat'] = repeat


        data_dir = os.path.join(data_root, 'v2020-other-PL')

        train_df = pd.read_csv(os.path.join(data_root, 'train_2016.csv'))#/  filter_train_2020 refined_2016 data_training filter_train_2020 train_2016 val_2016
        valid_df = pd.read_csv(os.path.join(data_root, 'val_2016.csv'))# /  filter_val_2020 val_2016 data_validation.
        test2016_df = pd.read_csv(os.path.join(data_root, 'data_test.csv'))


        train_set = GraphDataset(data_dir, train_df, graph_type=graph_type, dis_threshold=8, create=False)
        valid_set = GraphDataset(data_dir, valid_df, graph_type=graph_type, dis_threshold=8, create=False)
        test2016_set = GraphDataset(data_dir, test2016_df, graph_type=graph_type,dis_threshold=8, create=False)

        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
        valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)
        test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=8)

        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")


        device = torch.device('cuda:0')
        model = ConBAP(35, 256).to(device)
        load_model_dict(model, f'../unsupervised/model/20230817_110144_Con_BAP_repeat3/model/contrastive_1.pt')
        model = downstream_affinity(model, 256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6) #1e-4
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        
        model.train()
        for epoch in range(epochs):
            for data in train_loader:
                data['ligand_features'] = data['ligand_features'].to(device)
                # data['atom_pocket_features'] = data['atom_pocket_features'].to(device)
                # data['amino_acid_features'] = data['amino_acid_features'].to(device)
                data['complex_features'] = data['complex_features'].to(device)
                pred = model(data)
                label = data["ligand_features"].y
                
                

                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0)) 

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse, valid_pr = val(model, valid_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

        # final testing
        load_model_dict(model, best_model_list[-1])
        valid_rmse, valid_pr = val(model, valid_loader, device)
        test2016_rmse, test2016_pr = val(model, test2016_loader, device)


        msg = "valid_rmse-%.4f, valid_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f," \
                    % (valid_rmse, valid_pr, test2016_rmse, test2016_pr)

        logger.info(msg)
        

# %%