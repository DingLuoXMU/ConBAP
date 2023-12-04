
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from gvp import GVP, GVPConvLayer, LayerNorm
from egnn_clean import EGNN

# heterogeneous interaction layer
class MPNNL(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int, 
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MPNNL, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.edge_mlp = nn.Sequential(
            nn.Linear(6, out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.updateNN = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.messageNN = nn.Sequential(
            nn.Linear(in_channels*3, out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

    def forward(self, x, edge_index, edge_features,
                size=None):

        out_node_intra = self.propagate(edge_index=edge_index, x=x,edge_features=edge_features,messageNN=self.messageNN, updateNN=self.updateNN , size=size)#radial=radial_cov,
        out_node = self.mlp_node_cov(x + out_node_intra)  

        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, messageNN, edge_features,
                index: Tensor):
        edge_feat = self.edge_mlp(edge_features)
        return messageNN(torch.cat([x_j, x_i,edge_feat], dim=-1)) 

    def update(self, aggr_out: Tensor, x: Tensor, updateNN):
        return updateNN(torch.cat([aggr_out, x], dim=-1))



class MPNNP(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MPNNP, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.edge_mlp = nn.Sequential(
            nn.Linear(6, out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.updateNN = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.messageNN = nn.Sequential(
            nn.Linear(in_channels * 2 + 6, out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

    def forward(self, x, edge_index, edge_features,
                size=None):


        out_node_intra = self.propagate(edge_index=edge_index, x=x,messageNN=self.messageNN, updateNN=self.updateNN , size=size, edge_features=edge_features)#radial=radial_cov,

        out_node = self.mlp_node_cov(x + out_node_intra)

        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, edge_features: Tensor,messageNN,
                index: Tensor):
        return messageNN(torch.cat([x_j, x_i, edge_features], dim=-1))

    def update(self, aggr_out: Tensor, x: Tensor, updateNN):
        return updateNN(torch.cat([aggr_out, x], dim=-1))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
class GVP_embedding(nn.Module):
    '''
    Modified based on https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1, plm=True):

        super(GVP_embedding, self).__init__()
        
        # nn.Embedding(20,20)
        if seq_in and plm:
            self.W_s = nn.Embedding(20, 1280)
            node_in_dim = (node_in_dim[0] + 1280, node_in_dim[1])
        elif seq_in and not plm:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        self.plm = plm

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, h_V, edge_index, h_E, seq):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if self.plm:
            seq = seq
            # print('plm')
            # print('seq',np.array(seq).shape)
        else:
            seq = self.W_s(seq)
            # print('without plm')

        h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        # print('pass!')

        return out


class AttentionBlock(nn.Module):
    """ A class for attention mechanism with QKV attention """

    def __init__(self, hid_dim,  dropout):
        super().__init__()

        self.hid_dim = hid_dim

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):
        """ 
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Query and Value should always come from the same source (Aiming to focus on), Key comes from the other source
        Self-Att : Both three Query, Key, Value come form the same source (For refining purpose)
        """
          # Assuming key and value have the same sequence length
        
        q = self.f_q(query)
        k = self.f_k(key)
        v = self.f_v(value)
        q = torch.reshape(q,shape=[-1,self.hid_dim])  #[b,f]
        k = torch.reshape(k,shape=[-1,self.hid_dim]).permute([1,0])  #[f,a]
        v = torch.reshape(v,shape=[-1,self.hid_dim])  #[a,f]
        attention_matrix = torch.matmul(q,k)   #[b,a]
        attention_matrix = torch.sigmoid(attention_matrix)
        update_message = torch.matmul(attention_matrix,v) #[b,f]
        return update_message



class CrossAttentionBlock(nn.Module):
    def __init__(self, hid_dim,  dropout):
        super(CrossAttentionBlock, self).__init__()
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.att = AttentionBlock(hid_dim=hid_dim,  dropout=dropout)
        
        self.linear_res = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
        )

        self.linear_pocket = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
        )

        self.linear_lig = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
        )


    def forward(self, ligand_features, aa_features, lig_scope, amino_acid_scope):
        # Initialize lists to hold output features
        lig_features_list = []
        aa_features_list = []
        # Loop over the molecules, pockets, and amino acids in the batch
        for ((start_l, end_l), (start_a, end_a)) in zip(lig_scope, amino_acid_scope):
            lig_feature = (ligand_features[start_l:start_l+end_l])            
            # pocket_feature = (pocket_features[start_p:start_p+end_p])
            aa_feature = (aa_features[start_a:start_a+ end_a])
            # cross attention for compound information enrichment
            # aa_feature = self.linear_pocket(self.att(aa_feature,pocket_feature, pocket_feature))/end_p + aa_feature
           
            # cross-attention for interaction
            aa_feature = self.linear_res(self.att(aa_feature, lig_feature, lig_feature))/end_l + aa_feature            
            lig_feature = self.linear_lig(self.att(lig_feature, aa_feature, aa_feature))/end_a + lig_feature
        

            lig_features_list.append(lig_feature)
            aa_features_list.append(aa_feature)

        lig_features_batch = torch.cat(lig_features_list, dim=0)
        aa_features_batch = torch.cat(aa_features_list, dim=0)
            # Compute the element-wise product between ligand_features and pocket_features        
        return lig_features_batch, aa_features_batch

class InteractionBlock(nn.Module):
    def __init__(self, hid_dim,  dropout):
        super(InteractionBlock, self).__init__()
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.prot_lig_attention = CrossAttentionBlock(hid_dim=hid_dim,  dropout=dropout)
        
        self.pro_lig_interaction = nn.Sequential(
            nn.Linear(hid_dim,1),
            nn.Dropout(self.dropout),
            nn.Sigmoid(),
        )

        self.lig_trans_inteact = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
        )

        self.aa_trans_inteact = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
        )

        self.aa_trans_affinity = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
        )
        self.lig_trans__affinity = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),
        )
        self.linear_affinity = nn.Sequential(
            nn.Linear(hid_dim,hid_dim,bias=False),
            nn.LeakyReLU(0.1),
        )


    def forward(self, ligand_features,  aa_features, lig_scope,  amino_acid_scope):
        ligand_features, aa_features = self.prot_lig_attention(ligand_features,  aa_features, lig_scope, amino_acid_scope)
        # print(ligand_features.shape, aa_features.shape)
        predicted_interactions,predicted_vectors = [],[]
        for (start_l, end_l),  (start_a, end_a) in zip(lig_scope,  amino_acid_scope):
            lig_feature =  torch.unsqueeze((ligand_features[start_l: start_l+end_l]),0)  # [1, num_atoms, hid_dim]
            aa_feature = torch.unsqueeze((aa_features[start_a:start_a+ end_a]),1)  # [num_residues, 1, hid_dim]
            # print(lig_feature.shape, aa_feature.shape)
            predicted_interaction = self.pro_lig_interaction(torch.multiply(self.aa_trans_inteact(aa_feature),self.lig_trans_inteact(lig_feature))) # [num_residues, num_atoms, 1]
            # predicted_interaction = self.node_feature(predicted_interaction.reshape(-1,1))
            predicted_interactions.append(predicted_interaction) # [num_residues * num_atoms] 
            

            complex_features = torch.multiply(
                self.lig_trans__affinity(lig_feature),
                self.aa_trans_affinity(aa_feature)
            ) # [num_residues, num_atoms, hid_dim]
            # predict_vector = torch.sum(complex_features,dim=(0, 1))
            predict_vector = torch.sum(self.linear_affinity(complex_features) * predicted_interaction, dim=(0, 1))
            # print(predict_vector.shape)
            # max_pooling, _ = torch.max(complex_features.view(complex_features.shape[0] * complex_features.shape[1], hid_dim), dim=0)

            
            predicted_vectors.append(torch.unsqueeze(predict_vector,0))

        # predicted_interactions = torch.cat(predicted_interactions, dim=0) 
        predicted_vectors = torch.cat(predicted_vectors, dim=0)
        # print(predicted_vectors.shape)

        # print('predicted_interactions',len(predicted_interactions))

        return predicted_vectors# , predicted_affinities
# EGNN
class EGNN_complex(nn.Module):
    def __init__(self, hid_dim,  edge_dim, n_layers,attention=False, normalize=False, tanh=False):
        super(EGNN_complex, self).__init__()
        self.hid_dim = hid_dim
        self.edge_dim = edge_dim
        self.n_layers = n_layers
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh

        
        self.egnn=EGNN(hid_dim, hid_dim, hid_dim, in_edge_nf=edge_dim, n_layers=n_layers, residual=True, attention=attention, normalize=normalize, tanh=tanh)
    

        

       
    
    def forward(self, data_complex):
        
        complex_x_list = []
        for i in range(len(data_complex)):
            complex_x =data_complex[i].x
            complex_edge_attr = data_complex[i].edge_attr
            complex_edge_index =data_complex[i].edge_index
            complex_pos = data_complex[i].pos
                       
            complex_x, complex_pos = self.egnn(complex_x, complex_pos,complex_edge_index, complex_edge_attr)
            

            complex_x_list.append(complex_x)
        complex_x = torch.cat(complex_x_list, dim=0)  # [num_atoms, hid_dim]
 
        return complex_x

            




            
        


# %%