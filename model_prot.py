# noinspection PyInterpreter
import torch
from torch import nn
from e3nn import o3
from torch_cluster import radius_graph
from torch_scatter import scatter
from torch.nn import functional as F
from e3nn.nn import BatchNorm
import torch_geometric.nn as pyg_nn
from utils import rec_residue_feature_dims

class Protmod(torch.nn.Module):
    def __init__(self, sh_lmax=2, ns=64, nv=16, num_conv_layers=2, rec_max_radius=15, c_alpha_max_neighbors=24,
                 distance_embed_dim=32, use_second_order_repr=False, batch_norm=True, dropout=0.0, device='cuda'):
        """
        @param sh_lmax: spherical_harmonics
        @param ns: Number of hidden features per node of order 0
        @param nv: Number of hidden features per node of order >0
        @param num_conv_layers: num_conv_layers
        @param rec_max_radius: radius of residue
        @param c_alpha_max_neighbors: Maximum number of neighbors for each residue
        @param distance_embed_dim: Embedding size for the distance
        @param use_second_order_repr: Whether to use only up to first order representations or also second
        @param batch_norm: batch_norm
        @param dropout: dropout
        """
        super(Protmod, self).__init__()
        self.rec_max_radius = rec_max_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.distance_embed_dim = distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.num_conv_layers = num_conv_layers
        self.device = device

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        self.rec_conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            if i==0:
               in_irreps = in_irreps + ' + 2x1o'
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            rec_layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                hidden_features=3 * ns,
                residual=False,
                batch_norm=batch_norm,
                dropout=dropout
            )
            self.rec_conv_layers.append(rec_layer)
            self.rec_conv_layers = nn.ModuleList(self.rec_conv_layers)


    def forward(self, data, eps=1e-12):
        # builds the receptor initial node and edge embeddings
        node_attr = torch.cat([data['receptor'].x, data['receptor'].chis.sin() * data['receptor'].chi_masks,
                               data['receptor'].chis.cos() * data['receptor'].chi_masks], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = radius_graph(data['receptor'].pos, self.rec_max_radius, data['receptor'].batch,
                                  max_num_neighbors=self.c_alpha_max_neighbors)
        
        # create edge vec
        edge_index = edge_index[[1, 0]]
        src, dst = edge_index
        edge_vec = (data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]).to(self.device)

        # edge length (encoded with gussian smearing)
        edge_length_embedded = self.rec_distance_expansion(edge_vec.norm(dim=-1))

        # sphirical harmonics for a richer describtor of the relative positions
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component').to(self.device)

        # embd
        node_attr = node_attr.to(self.device)
        node_attr = self.rec_node_embedding(node_attr)
        edge_attr = self.rec_edge_embedding(edge_length_embedded)

        for l in range(len(self.rec_conv_layers)):

            if l == 0:
                n_vec = (data['receptor'].lf_3pts[:, 0] - data['receptor'].lf_3pts[:, 1]).to(self.device)
                n_norm_vec = n_vec / (n_vec.norm(dim=-1, keepdim=True) + eps)
                c_vec = (data['receptor'].lf_3pts[:, 2] - data['receptor'].lf_3pts[:, 1]).to(self.device)
                c_norm_vec = c_vec / (c_vec.norm(dim=-1, keepdim=True) + eps)

            edge_attr_ = torch.cat(
                [edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1).to(self.device)
            
            if l == 0:
                node_attr = self.rec_conv_layers[l](torch.cat([node_attr, n_norm_vec, c_norm_vec], dim=-1),
                                                        edge_index.to(self.device), edge_attr_, edge_sh)
            else:
                node_attr = self.rec_conv_layers[l](node_attr, edge_index, edge_attr_, edge_sh)

        emb = pyg_nn.global_mean_pool(node_attr, data['receptor'].batch.to(self.device))
        
        return emb



class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
            self.lm_embedding_dim = 1280
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(
                x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], axis=1))
        return x_embedding

class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src.to('cuda'), dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded
        if self.batch_norm:
            out = self.batch_norm(out)

        return out

class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader, DataListLoader
    from graph_prep.dataset import Myme
    train_dataset = Myme('./graph_prep/data/prot_emb_train_chain.h5', './graph_prep/pdbs_train/')
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_loader = loader_class(dataset=train_dataset, batch_size=2, num_workers=0, pin_memory=True)
    from model_prot import Protmod
    m = Protmod().to('mps')
    for g in train_loader:
        x = m(g)
        print(1)