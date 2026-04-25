import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np
import math

from torch_geometric.nn import Sequential, GATConv, TransformerConv
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn.models import InnerProductDecoder, GAE, VGAE
from torch_geometric.nn import GATConv, GAE


class PhenotypeAttention(torch.nn.Module):
    def __init__(self, in_channels, phenotype_dim, max_weight=0.1, dropout=0.2):
        super(PhenotypeAttention, self).__init__()
        self.query = Linear(in_channels, in_channels, bias=False)
        self.key = Linear(phenotype_dim, in_channels, bias=False)
        self.value = Linear(phenotype_dim, in_channels, bias=False)
        self.dropout = Dropout(dropout)
        self.max_weight = max_weight

    def forward(self, x, phenotype, phenotype_mask=None):
        if phenotype is None or phenotype.numel() == 0:
            return x

        phenotype = phenotype.to(device=x.device, dtype=x.dtype)
        if phenotype_mask is None:
            phenotype_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        else:
            phenotype_mask = phenotype_mask.to(device=x.device, dtype=torch.bool)

        attention_logits = (self.query(x) * self.key(phenotype)).sum(dim=-1, keepdim=True)
        attention_gate = torch.sigmoid(attention_logits / math.sqrt(x.size(-1)))
        phenotype_delta = torch.tanh(self.value(phenotype))
        phenotype_delta = self.dropout(phenotype_delta)
        phenotype_delta = phenotype_delta * phenotype_mask.unsqueeze(-1)

        return x + self.max_weight * attention_gate * phenotype_delta


# Define GAT-based encoder for GAE
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, phenotype_dim=0,
                 phenotype_attention_weight=0.1,
                 phenotype_attention_dropout=0.2):
        super(GATEncoder, self).__init__()
        self.phenotype_attention = None
        if phenotype_dim and phenotype_dim > 0 and phenotype_attention_weight > 0:
            self.phenotype_attention = PhenotypeAttention(
                in_channels,
                phenotype_dim,
                max_weight=phenotype_attention_weight,
                dropout=phenotype_attention_dropout)
        self.conv1 = GATConv(in_channels, 32, heads=1, dropout=0.6)
        self.conv2 = GATConv(32 * 1, out_channels, heads=1, concat=True, dropout=0.6)

    def apply_phenotype_attention(self, x, phenotype=None, phenotype_mask=None):
        phenotype_attention = getattr(self, "phenotype_attention", None)
        if phenotype_attention is None:
            return x
        return phenotype_attention(x, phenotype, phenotype_mask)

    def forward(self, x, edge_index, phenotype=None, phenotype_mask=None):
        x = self.apply_phenotype_attention(x, phenotype, phenotype_mask)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize GAE model with GAT encoder and move it to the GPU
class GAEModel(GAE):
    def __init__(self, in_channels, out_channels, phenotype_dim=0,
                 phenotype_attention_weight=0.1,
                 phenotype_attention_dropout=0.2):
        encoder = GATEncoder(in_channels, out_channels, phenotype_dim,
                             phenotype_attention_weight,
                             phenotype_attention_dropout)
        super(GAEModel, self).__init__(encoder)

    def get_attention_scores(self, data):
        x, edge_index = data.x, data.edge_index
        phenotype = getattr(data, "phenotype", None)
        phenotype_mask = getattr(data, "phenotype_mask", None)
        x = self.encoder.apply_phenotype_attention(x, phenotype, phenotype_mask)
        # Pass data through the first GAT layer to get attention scores
        _, (edge_index_selfloop, alpha) = self.encoder.conv1(x, edge_index, return_attention_weights=True)
        # matrix shape: number of edges x number of heads
        return edge_index_selfloop,alpha
    

class Encoder(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear1 = Linear(dim, dim)
        self.linear2 = Linear(dim, dim)

        # self loop is default，so also include the attention of self-loop
        # delete self loop
        self.conv1 = GATConv(dim, dim, add_self_loops=False)
        self.conv2 = GATConv(dim, dim, add_self_loops=False)
        
        # self.conv1 = TransformerConv(dim, dim, heads=1)
        # self.conv2 = TransformerConv(dim, dim, heads=1)

        self.act = torch.nn.CELU()

    def cat(self, x_gene, x_cell, y):
        result = []
        count_gene = 0
        count_cell = 0

        for i in y:
            if i:
                result.append(x_gene[count_gene].view(1, -1))
                count_gene += 1
            else:
                result.append(x_cell[count_cell].view(1, -1))
                count_cell += 1

        result = torch.cat(result)
        return result

    def forward(self, graph):
        x, edge_index, y = graph.x, graph.edge_index, graph.y

        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = self.act(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        return x

    def get_att(self, graph):
        x, edge_index,  y = graph.x, graph.edge_index, graph.y
        print(x.shape,y.shape)
        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x, att = self.conv2(x, edge_index, return_attention_weights=True)

        return x, att


class SenGAE(GAE):
    def __init__(self):
        super(SenGAE, self).__init__(encoder=Encoder(),
                                     decoder=InnerProductDecoder())

    def forward(self, graph, split=10):
        z = self.encode(graph)
        # adj_pred = self.decoder(z)
        return z
