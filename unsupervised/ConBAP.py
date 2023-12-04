# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import global_mean_pool, max_pool_x
from torch_geometric.nn.conv import  GATv2Conv
from HIL import CrossAttentionBlock,GVP_embedding,InteractionBlock,MPNNL,EGNN_complex

import math
import numpy as np


class ConBAP(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        # print(node_dim)
        self.lin_node_lig = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        
        self.lin_node_complex= nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())

    
        self.gconv1_l = MPNNL(hidden_dim, hidden_dim)
        self.gconv2_l = MPNNL(hidden_dim, hidden_dim)
        self.gconv3_l = MPNNL(hidden_dim, hidden_dim)

        self.egnn = EGNN_complex(hidden_dim, edge_dim=4, n_layers=4,attention=False)
        # self.egnn = EGNN_complex(hidden_dim,4,3) # 4 edge_dim, 3 lyaer
        self.conv_protein = GVP_embedding((6, 3), (hidden_dim, 16), 
                                              (32, 1), (32, 1), seq_in=True, plm=False)
        self.interaction = InteractionBlock(hidden_dim, 0.1)
        self.g = nn.Sequential(
            nn.Linear(hidden_dim,2*hidden_dim, bias=False), 
            # nn.BatchNorm1d(2*hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True),
            )
        self.fc = FC(hidden_dim*2, hidden_dim, 3, 0.1, 1)
        # self.SiLU = nn.SiLU()
        # self.dropout = nn.Dropout(0.1)
        # self.pro_lig_interaction_loss_fuction = nn.BCELoss(reduction='sum')
        # self.affinity_loss = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction='sum')
    def forward(self, data):
        
        lig_scope = data['lig_scope']
        amino_acid_scope = data['amino_acid_scope']

        data_l = data['ligand_features']
        data_aa = data['amino_acid_features']
        data_complex_native = data['native_complex_features']
        data_complex_redocked = data['redocked_complex_features']

        x_l,  edge_index_l,  x_aa, seq, node_s, node_v, edge_index, edge_s, edges_v,edge_feat_l = \
        data_l.x, data_l.edge_index, data_aa.x_aa, data_aa.seq, data_aa.node_s, data_aa.node_v,data_aa.edge_index, data_aa.edge_s, data_aa.edge_v, data_l.edge_attr
        
        # x_c_n, edge_index_c_n,edge_feat_c_n = data_complex_native.x, data_complex_native.edge_index, data_complex_native.edge_attr
        # x_c_r, edge_index_c_r,edge_feat_c_r = data_complex_redocked.x, data_complex_redocked.edge_index, data_complex_redocked.edge_attr

        # print('x_l:', x_l.shape)
        x_l = self.lin_node_lig(x_l)
        x_l = self.gconv1_l(x_l, edge_index_l, edge_feat_l)
        x_l = self.gconv2_l(x_l, edge_index_l, edge_feat_l)
        x_l = self.gconv3_l(x_l, edge_index_l, edge_feat_l)


         
        
       
        data_complex_native.x = self.lin_node_complex(data_complex_native.x)
        data_complex_redocked.x = self.lin_node_complex(data_complex_redocked.x)
        x_c_n = self.egnn(data_complex_native) # x_c_n:feature of native complex
        x_c_r = self.egnn(data_complex_redocked) # x_c_r:feature of redocked complex
        # print(x_c.shape)
        x_c_n = global_mean_pool(x_c_n,data_complex_native.batch)  # [batchsize,hidim]
        x_c_r = global_mean_pool(x_c_r,data_complex_redocked.batch)  # [batchsize,hidim]
        # print(x_c.shape)

        
        
        # print(x_c.shape)     
        nodes = (node_s, node_v)
        edges = (edge_s, edges_v)
        protein_out = self.conv_protein(nodes, edge_index, edges, seq)
        predicted_vectors = self.interaction(x_l, protein_out, lig_scope,  amino_acid_scope)  # [batchsize,hidim]


        predicted_vectors = self.g(predicted_vectors)
        x_c_r = self.g(x_c_r)
        x_c_n = self.g(x_c_n)
        affinity_x_c_r = self.fc(x_c_r)
        affinity_x_c_r = affinity_x_c_r.view(-1)
        affinity_x_c_n = self.fc(x_c_n)
        affinity_x_c_n = affinity_x_c_n.view(-1)
        # print(predicted_vectors)

        return predicted_vectors, x_c_n, x_c_r, affinity_x_c_n, affinity_x_c_r 


 

class EGNN_Model(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        # print(node_dim)
        self.lin_node_lig = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        
        self.lin_node_complex = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.lin_hide = nn.Sequential(Linear(hidden_dim, hidden_dim), nn.SiLU())

        self.egnn = EGNN_complex(hidden_dim, edge_dim=4, n_layers=3)
        # self.egnn = EGNN_complex(hidden_dim,4,3) # 4 edge_dim, 3 lyaer
        self.conv_protein = GVP_embedding((6, 3), (hidden_dim, 16), 
                                              (32, 1), (32, 1), seq_in=True, plm=True)
        self.interaction = InteractionBlock(hidden_dim, 0.1)
        self.affinity = nn.Sequential(
            nn.Linear(hidden_dim,2*hidden_dim), 
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.dropout(0.1),
            nn.BatchNorm1d(2*hidden_dim),
            nn.Linear(2*hidden_dim,1),
            )
        # self.fc = FC(hidden_dim*2, hidden_dim, 3, 0.1, 1)
        # self.SiLU = nn.SiLU()
        # self.dropout = nn.Dropout(0.1)
        # self.pro_lig_interaction_loss_fuction = nn.BCELoss(reduction='sum')
        # self.affinity_loss = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction='sum')
    def forward(self, data):
        data_l = data['ligand_features']
        
        data_aa = data['amino_acid_features']
        data_complex = data['complex_features']
        lig_scope = data['lig_scope']
        amino_acid_scope = data['amino_acid_scope']
        x_l,  edge_index_l,  x_aa, seq, node_s, node_v, edge_index, edge_s, edges_v,edge_feat_l, data_complex = \
        data_l.x, data_l.edge_index, data_aa.x_aa, data_aa.seq, data_aa.node_s, data_aa.node_v,data_aa.edge_index, data_aa.edge_s, data_aa.edge_v, data_l.edge_attr, \
        data_complex   
        data_complex.x = self.lin_node_complex(data_complex.x)
        x_c = self.egnn(data_complex) # x_c:feature of nodes
        # print(x_c.shape)
        x_c = global_mean_pool(x_c,data_complex.batch)  # [batchsize,hidim]
        # print(x_c.shape)

        output = self.affinity(x_c)
        # print('x_c.shape', x_c.shape)
        # print('predicted_vectors', predicted_vectors.shape)
        # print('out-1:', torch.max(predicted_vectors),'shape:', predicted_vectors.shape)
        # print('out-2:', torch.max(x_c), 'shape:', x_c.shape)
        # print("--------------------")
        return  output

class GPV_Model(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        # print(node_dim)
        self.lin_node_lig = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        
        self.lin_node_complex = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.lin_hide = nn.Sequential(Linear(hidden_dim, hidden_dim), nn.SiLU())
    
        self.gconv1_l = MPNNL(hidden_dim, hidden_dim)
        self.gconv2_l = MPNNL(hidden_dim, hidden_dim)
        self.gconv3_l = MPNNL(hidden_dim, hidden_dim)

        self.egnn = EGNN_complex(hidden_dim, edge_dim=4, n_layers=3)
        # self.egnn = EGNN_complex(hidden_dim,4,3) # 4 edge_dim, 3 lyaer
        self.conv_protein = GVP_embedding((6, 3), (hidden_dim, 16), 
                                              (32, 1), (32, 1), seq_in=True, plm=True)
        self.interaction = InteractionBlock(hidden_dim, 0.1)
        self.affinity = nn.Sequential(
            nn.Linear(hidden_dim,2*hidden_dim), 
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.dropout(0.1),
            nn.BatchNorm1d(2*hidden_dim),
            nn.Linear(2*hidden_dim,1),
            )
        # self.fc = FC(hidden_dim*2, hidden_dim, 3, 0.1, 1)
        # self.SiLU = nn.SiLU()
        # self.dropout = nn.Dropout(0.1)
        # self.pro_lig_interaction_loss_fuction = nn.BCELoss(reduction='sum')
        # self.affinity_loss = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction='sum')
    def forward(self, data):
        data_l = data['ligand_features']
        
        data_aa = data['amino_acid_features']
        data_complex = data['complex_features']
        lig_scope = data['lig_scope']
        amino_acid_scope = data['amino_acid_scope']
        x_l,  edge_index_l,  x_aa, seq, node_s, node_v, edge_index, edge_s, edges_v,edge_feat_l, data_complex = \
        data_l.x, data_l.edge_index, data_aa.x_aa, data_aa.seq, data_aa.node_s, data_aa.node_v,data_aa.edge_index, data_aa.edge_s, data_aa.edge_v, data_l.edge_attr, \
         data_complex

        # print('x_l:', x_l.shape)
        x_l = self.lin_node_lig(x_l)
        x_l = self.gconv1_l(x_l, edge_index_l, edge_feat_l)
        x_l = self.gconv2_l(x_l, edge_index_l, edge_feat_l)
        x_l = self.gconv3_l(x_l, edge_index_l, edge_feat_l)


         
        
       
        data_complex.x = self.lin_node_complex(data_complex.x)
        x_c = self.egnn(data_complex) # x_c:feature of nodes
        # print(x_c.shape)
        x_c = global_mean_pool(x_c,data_complex.batch)  # [batchsize,hidim]
        # print(x_c.shape)

        
        
        # print(x_c.shape)     
        nodes = (node_s, node_v)
        edges = (edge_s, edges_v)
        protein_out = self.conv_protein(nodes, edge_index, edges, seq)
        predicted_vectors = self.interaction(x_l, protein_out, lig_scope,  amino_acid_scope)  # [batchsize,hidim]
        # predicted_interactions:[num_nodes_lig*num_nodes_aa]  predicted_affinities:[batch_size,1]
        # 需要修改的是：x_c与 predicted_interactions经过线性变换，让他俩很像，然后再做一个loss
        # print(predicted_interactions.shape)
        # print('x_c.shape', len(x_c))
        # print('predicted_interactions', len(predicted_interactions))
        # pred_list = []
        # x_c_list = []
        # for i in range(len(x_c)):
        #     pred_list.append(self.g(predicted_interactions[i]))
        #     x_c_list.append(self.g(x_c[i]))
        predicted_vectors = self.g(predicted_vectors)
        # print('pro-out-2:',torch.max(x_c), 'shape:', x_c.shape)
        # print("   ")
        x_c = self.g(x_c)
        # print('x_c.shape', x_c.shape)
        # print('predicted_vectors', predicted_vectors.shape)
        # print('out-1:', torch.max(predicted_vectors),'shape:', predicted_vectors.shape)
        # print('out-2:', torch.max(x_c), 'shape:', x_c.shape)
        # print("--------------------")
        return  predicted_vectors, x_c 

class downstream_docking(nn.Module):
    def __init__(self, model, hidden_dim):
        super(downstream_docking,self).__init__()
        self.model = model
        self.hidden_dim = hidden_dim


    def forward(self, data):
        data_complex = data['complex_features']
        data_complex.x = self.model.lin_node_complex(data_complex.x)  
        x_c = self.model.egnn(data_complex) # x_c:feature of nodes
        # print(x_c.shape)
        x_c = global_mean_pool(x_c,data_complex.batch)  # [batchsize,hidim]
        # print(x_c.shape)
        x_c = self.model.g(x_c)
        affinity_x_c = self.model.fc(x_c)
        
        affinity_x_c = affinity_x_c.view(-1)
        
        return affinity_x_c

class downstream_affinity(nn.Module):
    def __init__(self, model, hidden_dim):
        super(downstream_affinity,self).__init__()
        self.model = model
        self.hidden_dim = hidden_dim
        self.fc = FC(hidden_dim*2, hidden_dim, 1, 0.1, 1)  #3

    def forward(self, data):
        data_complex = data['complex_features']
        data_complex.x = self.model.lin_node_complex(data_complex.x)  
        x_c = self.model.egnn(data_complex) # x_c:feature of nodes
        # print(x_c.shape)
        x_c = global_mean_pool(x_c,data_complex.batch)  # [batchsize,hidim]
        # print(x_c.shape)
        x_c = self.model.g(x_c)
        affinity_x_c = self.fc(x_c)
        
        affinity_x_c = affinity_x_c.view(-1)
        
        return affinity_x_c
        

class complex_free(nn.Module):
    def __init__(self, model, hidden_dim):
        super(complex_free,self).__init__()
        self.model = model
        self.hidden_dim = hidden_dim  
        self.fc = FC(hidden_dim*2, hidden_dim, 1, 0.1, 1)  #3
    def forward(self, data):
        data_l = data['ligand_features']
        
        data_aa = data['amino_acid_features']
        lig_scope = data['lig_scope']
        amino_acid_scope = data['amino_acid_scope']
        x_l,  edge_index_l,  x_aa, seq, node_s, node_v, edge_index, edge_s, edges_v,edge_feat_l = \
        data_l.x, data_l.edge_index, data_aa.x_aa, data_aa.seq, data_aa.node_s, data_aa.node_v,data_aa.edge_index, data_aa.edge_s, data_aa.edge_v, data_l.edge_attr
        
        # x_c_n, edge_index_c_n,edge_feat_c_n = data_complex_native.x, data_complex_native.edge_index, data_complex_native.edge_attr
        # x_c_r, edge_index_c_r,edge_feat_c_r = data_complex_redocked.x, data_complex_redocked.edge_index, data_complex_redocked.edge_attr

        # print('x_l:', x_l.shape)
        x_l = self.model.lin_node_lig(x_l)
        x_l = self.model.gconv1_l(x_l, edge_index_l, edge_feat_l)
        x_l = self.model.gconv2_l(x_l, edge_index_l, edge_feat_l)
        x_l = self.model.gconv3_l(x_l, edge_index_l, edge_feat_l)

        nodes = (node_s, node_v)
        edges = (edge_s, edges_v)
        protein_out = self.model.conv_protein(nodes, edge_index, edges, seq)
        predicted_vectors = self.model.interaction(x_l, protein_out, lig_scope, amino_acid_scope)
        predicted_vectors = self.model.g(predicted_vectors)
        affinity = self.fc(predicted_vectors)
        affinity = affinity.view(-1)
        return affinity


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h

# %%