import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as Data
# from linetimer import CodeTimer
from torch import nn, optim
from torch.nn import Dropout, Linear, ReLU
from torch.nn import functional as F
from tqdm import tqdm


def get_cluster_cell_dict(sencell_dict, nonsencell_dict):
    # Calculate the mapping table of cells contained in the cluster
    # {cluster: [cell_indexs]}
    cluster_sencell = defaultdict(list)
    cluster_nonsencell = defaultdict(list)

    for key, value in sencell_dict.items():
        cluster_sencell[value[1]].append(key)

    for key, value in nonsencell_dict.items():
        cluster_nonsencell[value[1]].append(key)

    return cluster_sencell, cluster_nonsencell


def getPrototypeEmb(sencell_dict, cluster_sencell, emb_pos=2):
    # Calculate the prototype embedding, only the prototype of sen cells is calculated
    prototype_emb = {}
    for key, value in cluster_sencell.items():
        embs = []
        for i in value:
            embs.append(sencell_dict[i][emb_pos].view(1, -1))
        embs = torch.cat(embs)
        prototype_emb[key] = torch.mean(embs, 0)

    return prototype_emb


class Sencell(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.hidden_size=128
        self.linear1 = Linear(dim, self.hidden_size)
        self.linear2 = Linear(self.hidden_size, self.hidden_size)
        self.linear21 = Linear(self.hidden_size, self.hidden_size)
        self.linear22 = Linear(self.hidden_size, self.hidden_size)
        self.linear3 = Linear(self.hidden_size, self.hidden_size)
        self.linear4 = Linear(self.hidden_size, dim)
        # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=8)
        # self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer,
        #                                                        num_layers=2)
        self.act = torch.nn.CELU()
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.levels = torch.nn.Parameter(torch.tensor([0, 0, 4., 4]),
                                         requires_grad=True)
        # self.levels = torch.tensor([-3.0, -1, 1, 3.0])
        self.device = None

    def catEmbeddings(self, sencell_dict, nonsencell_dict):
        embeddings = []
        for key, value in sencell_dict.items():
            embeddings.append(value[0].view(1, -1))
        for key, value in nonsencell_dict.items():
            embeddings.append(value[0].view(1, -1))
        return torch.cat(embeddings)

    def updateDict(self, x, sencell_dict, nonsencell_dict):
        count = 0
        for key, value in sencell_dict.items():
            sencell_dict[key][2] = x[count]
            count += 1
        for key, value in nonsencell_dict.items():
            nonsencell_dict[key][2] = x[count]
            count += 1

        return sencell_dict, nonsencell_dict

    def forward(self, sencell_dict, nonsencell_dict, device):
        x = self.catEmbeddings(sencell_dict, nonsencell_dict).to(device)
        self.device = device

        # model1
        # x = self.linear2(self.act(self.linear1(x)))
        # x = self.linear4(self.act(self.linear3(x)))
        
        # model 2
        x=self.act(self.linear1(x))
        x=x+self.act(self.linear2(x))
        x=self.layer_norm(x)
        x=x+self.act(self.linear22(self.act(self.linear21(x))))
        x=self.layer_norm(x)
        x = self.linear4(self.act(self.linear3(x)))

        result = self.updateDict(x, sencell_dict, nonsencell_dict)
        return result
    
    def get_embeddings(self,embs,device):
        x = embs.to(device)
        self.device = device

        # model1
        # x = self.linear2(self.act(self.linear1(x)))
        # x = self.linear4(self.act(self.linear3(x)))
        
        # model 2
        x=self.act(self.linear1(x))
        x=x+self.act(self.linear2(x))
        x=self.layer_norm(x)
        x=x+self.act(self.linear22(self.act(self.linear21(x))))
        x=self.layer_norm(x)
        x = self.linear4(self.act(self.linear3(x)))
        
        return x
    

    def prototypeLoss(self, distances):
        results = 0
        for cluster_distance in distances:
            results += cluster_distance[0].square().sum().sqrt()

        return results/len(distances)

    def eucliDistance(self, v1, v2):
        # Calculate Euclidean distance
        return F.pairwise_distance(v1.view(1, -1), v2.view(1, -1), p=2)

    def get_d1(self, sencell_dict, cluster_sencell, prototype_emb, emb_pos=2):
        d1 = []
        for cluster, prototype in prototype_emb.items():
            # For the case where there is only one sen cell, the distance is 0, but it is displayed as 1.1314e-05
            distance_ls = []
            for cell_index in cluster_sencell[cluster]:
                cell_emb = sencell_dict[cell_index][emb_pos]
                distance = self.eucliDistance(cell_emb, prototype)
                distance_ls.append(distance)
            d1.append(distance_ls)

        return d1

    def get_d2(self, sencell_dict, cluster_sencell, prototype_emb, emb_pos=2):
        # d2 represents the distance between sen cells in different clusters
        d2 = []
        for cluster, prototype in prototype_emb.items():
            distance_ls = []
            for another_cluster, cell_indexs in cluster_sencell.items():
                if another_cluster != cluster:
                    for cell_index in cell_indexs:
                        cell_emb = sencell_dict[cell_index][emb_pos]
                        distance = self.eucliDistance(cell_emb, prototype)
                        distance_ls.append(distance)
            d2.append(distance_ls)
        return d2

    def get_d3(self, nonsencell_dict, cluster_nonsencell, prototype_emb, emb_pos=2):
        # d3 represents the distance between senescent cells and non-senescent cells within the same cell type
        d3 = []
        for cluster, prototype in prototype_emb.items():
            distance_ls = []
            if cluster not in cluster_nonsencell:
                # This is the situation we want to avoid
                # This is a special case that needs to be handled separately. If there are no non-sen cells, there is no need to optimize d3
                # There are two solutions here. One is to directly d3.append([]), but this requires subsequent logic to take this situation into account
                # Another way is to avoid the situation where a cluster has no non-sen cells when sampling in the front
                # Let's put this problem aside for now
                print("There is no snc in this cluster：", cluster)
                distance_ls.append(torch.tensor([0.75]).to(self.device))
            else:
                for nonsencell_index in cluster_nonsencell[cluster]:
                    nonsencell_emb = nonsencell_dict[nonsencell_index][emb_pos]
                    distance = self.eucliDistance(nonsencell_emb, prototype)
                    distance_ls.append(distance)
            d3.append(distance_ls)
        return d3

    def get_d4(self, nonsencell_dict, cluster_nonsencell, prototype_emb, emb_pos=2):
        # d4 represents the distance between different cell types, between snc and non-snc
        d4 = []
        for cluster, prototype in prototype_emb.items():
            distance_ls = []
            for another_cluster, nonsencell_indexs in cluster_nonsencell.items():
                if another_cluster != cluster:
                    for nonsencell_index in nonsencell_indexs:
                        nonsencell_emb = nonsencell_dict[nonsencell_index][emb_pos]
                        distance = self.eucliDistance(
                            nonsencell_emb, prototype)
                        distance_ls.append(distance)
            d4.append(distance_ls)
        return d4

    def caculateDistance(self, sencell_dict, nonsencell_dict,
                         cluster_sencell, cluster_nonsencell,
                         prototype_emb, emb_pos=2):
        # d1: The distance between the prototype and each cell in the sen cell cluster
        # For each cluster with sen cells, there will be d1
        d1 = self.get_d1(sencell_dict, cluster_sencell, prototype_emb, emb_pos)
        # d2 represents the distance between sen cells in different clusters
        d2 = self.get_d2(sencell_dict, cluster_sencell, prototype_emb, emb_pos)
        # d3 represents the distance between sen cells and non-sen cells within the same cell type
        d3 = self.get_d3(nonsencell_dict, cluster_nonsencell,
                         prototype_emb, emb_pos)
        # d4 represents the distance between different cell types, between sen and non-sen
        # d4 = self.get_d4(nonsencell_dict, cluster_nonsencell,
        #                  prototype_emb, emb_pos)
        return d1, d2, d3

    def getMultiLevelDistanceLoss(self, distances):
        d1, d2, d3 = distances
        result = 0

        def distanceDiff(cluster_d, level):
            # cluster_d is a list of distances of the same type in the same cluster
            count = 0
            result = 0
            for d in cluster_d:
                result += (d-level).abs()
                count += 1
            if count==0:
                return 0
            return result/count

        for cluster_d_1, cluster_d_2, cluster_d_3 in zip(d1, d2, d3):
            result += distanceDiff(cluster_d_1, self.levels[0])
            result += distanceDiff(cluster_d_2, self.levels[1])
            result += distanceDiff(cluster_d_3, self.levels[2])
            # result += distanceDiff(cluster_d_4, self.levels[3])

        return result

    def loss(self, sencell_dict, nonsencell_dict):
        # step 1: Calculate the mapping table of cluster and cell
        cluster_sencell, cluster_nonsencell = get_cluster_cell_dict(
            sencell_dict, nonsencell_dict)
        # step 2: Calculate prototype embedding of sen cell clusters
        prototype_emb = getPrototypeEmb(sencell_dict, cluster_sencell)
        # step 3: Calculate distance
        distances = self.caculateDistance(sencell_dict, nonsencell_dict,
                                          cluster_sencell, cluster_nonsencell,
                                          prototype_emb)
        # step 4: Calculate multi-level distance
        loss = self.getMultiLevelDistanceLoss(distances)

        return loss


def process_dict(cell_dict,dgl_graph,args):
    if dgl_graph is None:
        for key in cell_dict:
            cell_index=cell_dict[key][-1]-args.gene_num
    else:
        for key in cell_dict:
            cell_index=cell_dict[key][-1]-args.gene_num
            cell_dict[key][0]=torch.cat([cell_dict[key][0], 
                                        dgl_graph.nodes[cell_index].data['pos_enc'].reshape(-1,)
                                        ])
    return cell_dict


def cell_optim(cellmodel, optimizer, sencell_dict, nonsencell_dict,dgl_graph, args, train=False):
    # optimizer = torch.optim.RMSprop(cellmodel.parameters(), lr=0.1, alpha=0.5,
    #                                 weight_decay=1e-4)
    final_loss = None
    if train:
        cellmodel.train()
        sencell_dict=process_dict(sencell_dict,dgl_graph,args)
        nonsencell_dict=process_dict(nonsencell_dict,dgl_graph,args)
        
        for epoch in range(args.cell_optim_epoch):
            optimizer.zero_grad()
            sencell_dict, nonsencell_dict = cellmodel(
                sencell_dict, nonsencell_dict, args.device)
            loss = cellmodel.loss(sencell_dict, nonsencell_dict)
            print(loss.item())
            loss.backward()
            optimizer.step()
        final_loss = loss.item()

        torch.save(cellmodel, os.path.join(
            args.output_dir, f'{args.exp_name}_cellmodel.pt'))
    else:
        cellmodel = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_cellmodel.pt'))
        sencell_dict, nonsencell_dict = cellmodel(
            sencell_dict, nonsencell_dict, args.device)

    return cellmodel, sencell_dict, nonsencell_dict, final_loss


def old_cell_optim(sencell_dict, nonsencell_dict, device, retrain=False):
    cellmodel = Sencell().to(device)
    optimizer = torch.optim.Adam(cellmodel.parameters(), lr=0.001,
                                 weight_decay=1e-3)
    # optimizer = torch.optim.RMSprop(cellmodel.parameters(), lr=0.1, alpha=0.5,
    #                                 weight_decay=1e-4)
    if retrain:
        cellmodel.train()
        for epoch in range(20):
            optimizer.zero_grad()
            # print(cellmodel.levels)
            sencell_dict, nonsencell_dict = cellmodel(
                sencell_dict, nonsencell_dict, device)
            loss = cellmodel.loss(sencell_dict, nonsencell_dict)
            print(loss.item())
            loss.backward()
            optimizer.step()

        torch.save(cellmodel, './cellmodel1.pt')
    else:
        cellmodel = torch.load('./cellmodel1.pt')
        cellmodel.eval()
        sencell_dict, nonsencell_dict = cellmodel(
            sencell_dict, nonsencell_dict, device)
    return sencell_dict, nonsencell_dict


def update_cell_embeddings(sampled_graph, sencell_dict, nonsencell_dict):
    feature = sampled_graph.x
    for key, value in sencell_dict.items():
        feature[key] = value[2].detach()
    for key, value in nonsencell_dict.items():
        feature[key] = value[2].detach()

    sampled_graph.x = feature
    return sampled_graph
