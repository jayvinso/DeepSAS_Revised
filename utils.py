import scanpy as sp
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
from torch_geometric.data import Data as Graphdata
from torch_geometric.utils import to_undirected
import networkx as nx
from tabulate import tabulate
from scipy import sparse as scsp

import argparse


def parse_args():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description='DeepSAS main program for senescent cells identification')

    # Input/output arguments
    parser.add_argument('--input_data_count', type=str, 
                       default="/bmbl_data/huchen/deepSAS_data/fixed_data_0525.h5ad", 
                       help='Path to input data (h5ad format)')
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                       help='Base output directory')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name (used for output directory naming)')
    parser.add_argument('--device_index', type=int, default=0, 
                       help='CUDA device index to use')
    parser.add_argument('--retrain', action='store_true', default=False, 
                       help='Whether to retrain models or use saved ones')
    parser.add_argument('--timestamp', type=str, default="", 
                       help='Timestamp for the experiment, used for output directory naming')

    # Model configuration arguments
    parser.add_argument('--seed', type=int, default=40, 
                       help='Random seed for reproducibility')
    parser.add_argument('--n_genes', type=str, default='full', 
                       help='Number of genes to use (3000, 8000 or full)')
    parser.add_argument('--ccc', type=str, default='type1', 
                       help='Cell-cell edge type: type1 (binary), type2 (continuous), type3 (none)')
    parser.add_argument('--gene_set', type=str, default='full', 
                       help='Gene set to use (senmayo, fridman, etc.)')
    parser.add_argument('--emb_size', type=int, default=12, 
                       help='Embedding dimension size')
    parser.add_argument('--phenotype_key', type=str, default='', 
                       help='Optional adata.obs column for phenotype-aware attention')
    parser.add_argument('--phenotype_attention_weight', type=float, default=0.1, 
                       help='Maximum weight for phenotype-aware attention residual')
    parser.add_argument('--phenotype_attention_dropout', type=float, default=0.2, 
                       help='Dropout applied inside phenotype-aware attention')

    # Training parameters
    parser.add_argument('--gat_epoch', type=int, default=30, 
                       help='Number of epochs to train the GAT model')
    parser.add_argument('--sencell_num', type=int, default=600, 
                       help='Number of senescent cells to use')
    parser.add_argument('--sengene_num', type=int, default=200, 
                       help='Number of senescence-associated genes to use')
    parser.add_argument('--sencell_epoch', type=int, default=40, 
                       help='Number of epochs to train the Sencell model')
    parser.add_argument('--cell_optim_epoch', type=int, default=50, 
                       help='Number of epochs for cell embedding optimization')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                       help='Initial learning rate')
    parser.add_argument('--batch_id', type=int, default=0, 
                       help='Batch ID for processing')

    args = parser.parse_args()
    
    # Validate arguments
    if args.emb_size <= 0:
        parser.error("Embedding size must be positive")
    
    if args.gat_epoch <= 0 or args.sencell_epoch <= 0 or args.cell_optim_epoch <= 0:
        parser.error("Number of epochs must be positive")

    if args.phenotype_attention_weight < 0 or args.phenotype_attention_weight > 0.5:
        parser.error("Phenotype attention weight must be between 0 and 0.5")

    if args.phenotype_attention_dropout < 0 or args.phenotype_attention_dropout >= 1:
        parser.error("Phenotype attention dropout must be in [0, 1)")
    
    return args




def load_example_data(path="example_data/example_data.h5ad",
                      ct_name='clusters'):
    print("Load example data ...")
    adata = sp.read_h5ad(path)
    
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=10)

    print(f'\tNumber of cells: {adata.shape[0]}\n\tNumber of genes: {adata.shape[1]}')
    
    celltype_names=list(adata.obs[ct_name].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("Cell type names:", celltype_names)
    # 2d-list, include the cell index in each cluster
    cluster_cell_ls = []
    # 1d-array，include the cluster index of each cell
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # all cell index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs[ct_name]
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names



def load_data_rep(exp_name):
    file_name=f"/bmbl_data/chenghao/sencell/data4_robust_test/{exp_name}.h5ad"
    print("load_data ", file_name, "...")
    adata = sp.read_h5ad(file_name)
    ct_name='clusters'
    
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=10)

    print(f'Number of cells: {adata.shape[0]}\nNumber of genes: {adata.shape[1]}')
    
    celltype_names=list(adata.obs[ct_name].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, include the cell index in each cluster
    cluster_cell_ls = []
    # 1d-array，include the cluster index of each cell
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # all cell index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs[ct_name]
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names



def load_data1(path="/bmbl_data/huchen/deepSAS_data/new_anno_data1.h5ad"):
    # "/bmbl_data/chenghao/sencell/fixed_data_0520.h5ad", 7w cells
    print("load_data1 ...")
    adata = sp.read_h5ad(path)
    ct_name='clusters'
    
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=10)

    print(f'Number of cells: {adata.shape[0]}\nNumber of genes: {adata.shape[1]}')

    celltype_names=list(adata.obs[ct_name].value_counts().index)

    print("The number of cell types", len(celltype_names))
    print("celltype names:", celltype_names)
    # 2d-list, include the cell index in each cluster
    cluster_cell_ls = []
    # 1d-array，include the cluster index of each cell
    cell_cluster_arr = np.array([0]*adata.shape[0])
    # all cell index
    all_indexs = np.arange(adata.shape[0])
    for i, celltype_name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs[ct_name]
                                 == celltype_name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)]
               for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def get_ccc_markers():
    ligand_receptor_dict = {
        "IL6": ["IL6ST", "IL6R", "HRH1", "F3"],
        "CXCL10": ["DPP4", "CCR3", "GRM7", "ADRA2A", "SDC4", "CXCR3", "MTNR1A"],
        "IL1B": ["ADRB2", "SIGIRR", "IL1RAP", "IL1R2", "IL1R1"],
        "CCL2": ["ACKR1", "CCR2", "ACKR4", "CCR10", "CCR3", "CCR5", "CCR1", "CCR4", "ACKR2"],
        "CCL5": ["SDC1", "GPR75", "ADRA2A", "CCR3", "CCRL2", "CCR5", "GRM7", "SDC4", "ACKR1", "MTNR1A", "CCR1", "CCR4", "CXCR3", "ACKR4", "ACKR2"],
        "HMGB1": ["TLR4", "TLR9", "AGER", "CXCR4", "SDC1", "THBD", "TLR2", "CD163"],
        "TNF": ["PTPRS", "CELSR2", "RIPK1", "FLT4", "TRAF2", "FAS", "ICOS", "NOTCH1", "TNFRSF1B", "TRADD", "TRPM2", "FFAR2", "VSIR", "TNFRSF21", "TNFRSF1A"],
        "SERPINE1": ["LRP1", "ITGAV", "PLAUR", "ITGB5", "LRP2"],
    }
    
    new_gene_set=set()
    for key,value in ligand_receptor_dict.items():
        new_gene_set.add(key)
        new_gene_set.update(value)
    return ligand_receptor_dict,new_gene_set

def get_cellcyle_markers():
    return ["CDKN1A","CDKN2A","TP53","GADD45A","IGFBP7","SERPINE1","GLB1","IL6","IL8","MMP1","MMP3"]


def load_markers(args):
    markers = pd.read_csv("senescence_marker_list.csv")
    # Series
    markers_ls=[]
    for col_name, data in markers.items():
        markers_ls.append(list(data[data.notnull()]))
    
    markers5 = list(get_ccc_markers()[1])
    markers_ls.append(markers5)
    
    markers6 = get_cellcyle_markers()
    markers_ls.append(markers6)

    print('The number of genes in marker list：')
    # print(tabulate([["SenMayo", "FRIDMAN", "CellAge", "GO","L-R Markers","Cell Cycle Markers"],
    #                 [len(markers_ls[0]), len(markers_ls[1]), len(markers_ls[2]), len(markers_ls[3]),len(markers_ls[4]),len(markers_ls[5])]],
    #                headers="firstrow"))
    
    # remove go list and L-R markers
    print(tabulate([["SenMayo", "FRIDMAN", "CellAge", "Cell Cycle Markers"],
                    [len(markers_ls[0]), len(markers_ls[1]), len(markers_ls[2]),len(markers_ls[5])]],
                headers="firstrow"))
    
    markers_ls=[markers_ls[0],markers_ls[1],markers_ls[2],markers_ls[5]]
    
    # senmayo or fridman or cellage or senmayo+cellage 
    # or senmayo+fridman or senmayo+fridman+cellage or full
    if args.gene_set == 'full':
        return markers_ls
    elif args.gene_set == 'senmayo':
        return [markers_ls[0]]
    elif args.gene_set == 'fridman':
        return [markers_ls[1]]
    elif args.gene_set == 'cellage':
        return [markers_ls[2]]
    elif args.gene_set == 'senmayo+cellage':
        return [markers_ls[0],markers_ls[2]]
    elif args.gene_set == 'senmayo+fridman':
        return [markers_ls[0],markers_ls[1]]
    elif args.gene_set == 'senmayo+fridman+cellage':
        return [markers_ls[0],markers_ls[1],markers_ls[2]]
    # markers_ls
    return markers_ls


def load_nonsenmarkers(adata):
    nonsen_markers=["CCNB1","CDK1","CDC25C","WEE1","CHK1","CCNA2","PCNA","MCM","RPA",
                    "DHFR","CCNB1","AURKA","AURKB","PLK1","H3S10ph","BUB1"]
    nonsen_markers=[gene for gene in nonsen_markers if gene in adata.var_names]
    return nonsen_markers

def get_highly_genes_old(adata):
    return list(adata.var[adata.var['vst.variable'] == True].index)


def get_highly_genes(adata,n_genes):
    new_data=adata.copy()
    sp.pp.normalize_total(new_data, target_sum=1e4)
    sp.pp.log1p(new_data)
    
    sp.pp.highly_variable_genes(new_data, n_top_genes=n_genes)
    highly_genes = list(new_data.var[new_data.var['highly_variable'] == True].index)
    return highly_genes


def combine_genes(adata, markers_ls, args):
    markers_set = set([gene for markers in markers_ls for gene in markers])
    print('Total marker genes: ', len(markers_set))

    # use highly_genes
    # adata.X = scipy.sparse.csr_matrix(adata.X)
    if args.n_genes=='full':
        print("use all genes!")
        highly_genes= list(adata.var.index)
    else:
        print(f"use {args.n_genes} highly genes!")
        highly_genes = get_highly_genes(adata,int(args.n_genes))
        print('Highly genes num: ', len(highly_genes))
    
    # use all genes
    # highly_genes=list(adata.var.index)

    # remove senescent genes and append them to the end
    highly_genes = sorted(list(set(highly_genes)-markers_set))
    print('After genes dropped duplicated sengenes: ', len(highly_genes))

    # Note: It is possible that senescent genes are not expressed in any cells
    # Therefore, add a step to remove genes with zero expression
    sen_gene_ls = []
    # cell_gene = adata.X.toarray()
    if scsp.issparse(adata.X):
        cell_gene = adata.X.toarray()
    else:
        cell_gene = adata.X
    gene_names = list(adata.var.index)

    for gene in markers_set:
        if gene in gene_names:
            gene_index = gene_names.index(gene)
            if max(cell_gene[:, gene_index]) != 0:
                sen_gene_ls.append(gene)

    # highly gene may have zero expression in all cells
    filtered_highly_genes = []
    for gene in highly_genes:
        if gene in gene_names:
            gene_index = gene_names.index(gene)
            if max(cell_gene[:, gene_index]) != 0:
                filtered_highly_genes.append(gene)

    if len(filtered_highly_genes) != len(highly_genes):
        print("Highly genes have zero expression in all cells！！")

    # combine them all
    sen_gene_ls = sorted(sen_gene_ls)
    gene_names = filtered_highly_genes+sen_gene_ls
    print('Total gene num:', len(gene_names))

    # subset adata
    adata_gene_names = list(adata.var.index)
    gene_indexs = [adata_gene_names.index(name) for name in gene_names]
    new_data = adata[:, gene_indexs]

    assert gene_names[100] == new_data.var.index[100], "Bug!!!"
    # Get the index of each marker in the marker list
    # markers_index has overlap
    markers_index = []
    for markers in markers_ls:
        indexs = []
        for marker in markers:
            if marker in gene_names:
                indexs.append(gene_names.index(marker))
        markers_index.append(indexs)
    # Get markers_index, which contains 4 lists, which are the indexes of markers in genes
    # the index of all markers
    sen_gene_ls = [gene_names.index(i) for i in sen_gene_ls]
    nonsen_gene_ls = [gene_names.index(i) for i in filtered_highly_genes]
    return new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names


def process_data(adata, cluster_cell_ls, cell_cluster_arr,args):
    # step 1: load marker
    markers_ls = load_markers(args)
    # step 2: append markers to highly genes
    new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names = combine_genes(
        adata, markers_ls,args)

    return new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names


def build_graph_nx(adata,gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names,args):
    # build nx graph and pyg graph
    # step 1: Calculate edge index
    g_index, c_index = np.nonzero(gene_cell)
    print('Cell-gene graph, the number of edges:', len(g_index))
    gene_num = gene_cell.shape[0]
    c_index += gene_num
    if args.ccc=='type1':
        print("add cell-cell edges with weights in 0 and 1...")
        adj_matrix,_=build_ccc_graph(gene_cell,gene_names)
        index1, index2 = np.nonzero(adj_matrix)
        print('CCC graph, the number of edges:', len(index1))
        index1+=gene_num
        index2+=gene_num
        new_g_index=np.concatenate([g_index,index1])
        new_c_index=np.concatenate([c_index,index2])
        edge_index = torch.tensor(np.array([new_g_index, new_c_index]), dtype=torch.long)
        ccc_matrix=None
    elif args.ccc=='type2':
        print("add cell-cell edges with weights in 0 to 1...")
        adj_matrix,ccc_matrix=build_ccc_graph(gene_cell,gene_names)
        index1, index2 = np.nonzero(adj_matrix)
        print('CCC graph, the number of edges:', len(index1))
        index1+=gene_num
        index2+=gene_num
        new_g_index=np.concatenate([g_index,index1])
        new_c_index=np.concatenate([c_index,index2])
        edge_index = torch.tensor(np.array([new_g_index, new_c_index]), dtype=torch.long)
    else:
        print("no cell-cell edges ...")
        edge_index = torch.tensor(np.array([g_index, c_index]), dtype=torch.long)
        ccc_matrix=None
        
    # step 2: build nx graph, add attributes
    graph_nx = nx.Graph(edge_index.T.tolist())

    # Add another attribute, which is the index of each node in the big graph
    for i in range(gene_num):
        graph_nx.nodes[i]['type'] = 'g'
        graph_nx.nodes[i]['index'] = i
        graph_nx.nodes[i]['name'] = gene_names[i]
        graph_nx.nodes[i]['is_sen'] = i in sen_gene_ls

    cell_names=list(adata.obs.index)
    for i in range(gene_cell.shape[1]):
        graph_nx.nodes[i+gene_num]['type'] = 'c'
        graph_nx.nodes[i+gene_num]['cluster'] = cell_cluster_arr[i]
        graph_nx.nodes[i+gene_num]['index'] = i+gene_num
        graph_nx.nodes[i+gene_num]['name'] = cell_names[i]

    return graph_nx,edge_index,ccc_matrix

def add_nx_embedding(graph_nx, gene_embed, cell_embed):
    for i in range(gene_embed.shape[0]):
        graph_nx.nodes[i]['emb'] = gene_embed[i].detach().cpu()

    for i in range(cell_embed.shape[0]):
        graph_nx.nodes[i+gene_embed.shape[0]
                       ]['emb'] = cell_embed[i].detach().cpu()

    return graph_nx


def build_phenotype_node_features(adata, gene_num, phenotype_key):
    if phenotype_key is None or phenotype_key == '':
        return None, None, []

    if phenotype_key not in adata.obs:
        raise ValueError(f'Phenotype key "{phenotype_key}" was not found in adata.obs')

    phenotype_series = adata.obs[phenotype_key]
    valid_cells = ~phenotype_series.isna().to_numpy()
    if valid_cells.sum() == 0:
        raise ValueError(f'Phenotype key "{phenotype_key}" contains only missing values')

    if pd.api.types.is_numeric_dtype(phenotype_series):
        values = phenotype_series.astype(float).to_numpy().reshape(-1, 1)
        median_value = np.nanmedian(values)
        values = np.where(np.isnan(values), median_value, values)
        std_value = np.std(values)
        if std_value == 0:
            std_value = 1.0
        cell_phenotype = (values - np.mean(values)) / std_value
        phenotype_names = [phenotype_key]
    else:
        values = phenotype_series.astype('object').where(phenotype_series.notna(), '__missing__')
        phenotype_df = pd.get_dummies(values, prefix=phenotype_key)
        if f'{phenotype_key}___missing__' in phenotype_df.columns:
            phenotype_df = phenotype_df.drop(columns=[f'{phenotype_key}___missing__'])
        cell_phenotype = phenotype_df.to_numpy(dtype=np.float32)
        phenotype_names = list(phenotype_df.columns)

    if cell_phenotype.shape[1] == 0:
        raise ValueError(f'Phenotype key "{phenotype_key}" has no usable non-missing categories')

    phenotype = np.zeros((gene_num + adata.n_obs, cell_phenotype.shape[1]), dtype=np.float32)
    phenotype[gene_num:] = cell_phenotype.astype(np.float32)

    phenotype_mask = np.zeros(gene_num + adata.n_obs, dtype=bool)
    phenotype_mask[gene_num:] = valid_cells

    print(f'Phenotype-aware attention enabled with key "{phenotype_key}" and {cell_phenotype.shape[1]} feature(s)')
    return torch.tensor(phenotype, dtype=torch.float32), torch.tensor(phenotype_mask), phenotype_names


def build_graph_pyg(gene_cell, gene_embed, cell_embed,edge_indexs,ccc_matrix=None,
                    phenotype=None, phenotype_mask=None, phenotype_names=None):
    print("build graph pyg")
    # Represents the node type
    y = [True]*gene_cell.shape[0]+[False]*gene_cell.shape[1]
    y = torch.tensor(y)

    print('edge index: ', edge_indexs.shape)
    x = torch.cat([gene_embed, cell_embed]).detach()
    print('node feature: ', x.shape)

    if ccc_matrix is None:
        print('build graph pyg without edge weights ... ')
        edge_index = to_undirected(edge_indexs)
        graph_pyg = Graphdata(x=x, edge_index=edge_index, y=y)
    else:
        print('build graph pyg with edge weights ... ')
        flatten_edge_features=ccc_matrix[ccc_matrix!=0]
        min_val = np.min(flatten_edge_features)
        max_val = np.max(flatten_edge_features)
        normalized_array = (flatten_edge_features - min_val) / (max_val - min_val)
        
        edge_attr=np.concatenate([np.ones(edge_indexs.shape[1]-len(normalized_array)),normalized_array])
        undirected_edge_index, undirected_edge_attr = to_undirected(edge_indexs, 
                                                            edge_attr=torch.tensor(edge_attr),reduce='mean')
        
        graph_pyg = Graphdata(x=x, edge_index=undirected_edge_index,edge_attr=undirected_edge_attr, y=y)

    if phenotype is not None:
        graph_pyg.phenotype = phenotype
        graph_pyg.phenotype_mask = phenotype_mask
        graph_pyg.phenotype_names = phenotype_names or []

    print('Pyg graph:', graph_pyg)
    print('graph.is_directed():', graph_pyg.is_directed())

    return graph_pyg


def build_ccc_matrix(expression_matrix,gene_names):
    # cell x gene
    ccc_matrix=None
    ligand_receptor_dict=get_ccc_markers()[0]
    
    for ligand in ligand_receptor_dict:
        if ligand not in gene_names:
            continue
        ligand_exp=expression_matrix[:,gene_names.index(ligand)].reshape(-1,1)
        receptor_indexs=[]
        for receptor in ligand_receptor_dict[ligand]:
            if receptor in gene_names:
                receptor_indexs.append(gene_names.index(receptor))
        receptor_exp=expression_matrix[:,receptor_indexs]
        receptor_exp=np.sum(receptor_exp, axis=1).reshape(1,-1)
        result=ligand_exp*receptor_exp
        if ccc_matrix is None:
            ccc_matrix=result
        else:
            ccc_matrix=ccc_matrix+result
            
    return ccc_matrix

def convert_to_adj(ccc_matrix):
    # p=exp(-1/x)
    masked_result = ma.masked_where(ccc_matrix == 0, ccc_matrix)
    result_transformed = np.exp(-1 / masked_result)
    result_transformed = result_transformed.filled(0)
    
    symmetric_result = 0.5 * (result_transformed + result_transformed.T)
    
    mask = symmetric_result >= 0.8
    result_transformed_masked = np.where(mask, symmetric_result, 0)

    return result_transformed_masked

def convert_to_adj_v1(ccc_matrix,t=0.8):
    print("convert_to_adj_v1")
    n=ccc_matrix.shape[0]
    # Parameters for your equation
    w = 1.0  # replace with the actual value
    b = 0.0  # replace with the actual value

    # Initialize an empty adjacency matrix
    adj_matrix = np.zeros((n, n))

    # Iterate over each pair of nodes
    for i in range(n):
        for j in range(n):
            # Calculate the norm of the difference between the embeddings
            diff_norm = np.linalg.norm(ccc_matrix[i] - ccc_matrix[j])

            # Use your equation to calculate the edge weight
            edge_weight = 1 / (1 + np.exp(w * diff_norm**2 + b))

            # Store the result in the adjacency matrix
            adj_matrix[i, j] = edge_weight
    
    mask = adj_matrix >= t
    result_transformed_masked = np.where(mask, adj_matrix, 0)
    return result_transformed_masked
    
    
import numpy as np
from numba import jit

@jit(nopython=True)
def compute_adj_matrix(ccc_matrix, w=1.0, b=0.0):
    n = ccc_matrix.shape[0]
    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            diff_norm = np.linalg.norm(ccc_matrix[i] - ccc_matrix[j])
            adj_matrix[i, j] = 1 / (1 + np.exp(w * diff_norm**2 + b))
    
    return adj_matrix

def convert_to_adj_v2(ccc_matrix, t=0.8):
    print("convert_to_adj_v2")
    adj_matrix = compute_adj_matrix(ccc_matrix)
    result_transformed_masked = np.where(adj_matrix >= t, adj_matrix, 0)
    return result_transformed_masked


def build_ccc_graph(gene_cell,gene_names):
    ccc_matrix=build_ccc_matrix(gene_cell.T,gene_names)
    adj_matrix=convert_to_adj(ccc_matrix)

    ccc_matrix=ccc_matrix*adj_matrix
    return adj_matrix,ccc_matrix


def get_sencell_cover(old_sencell_dict, sencell_dict):
    set1 = set(list(old_sencell_dict.keys()))
    set2 = set(list(sencell_dict.keys()))
    set3 = set1.intersection(set2)
    if len(set2) == 0:
        print('sencell cover: 0.0 (sencell_dict is empty)')
        return 0.0
    print('sencell cover:', len(set3)/len(set2))
    return len(set3)/len(set2)

def get_sencell_intersection(old_sencell_dict, sencell_dict):
    set1 = set(list(old_sencell_dict.keys()))
    set2 = set(list(sencell_dict.keys()))
    set3 = set1.intersection(set2)
    new_sencell_dict={}
    for i in set3:
        new_sencell_dict[i]=sencell_dict[i]
    print('length of new sencell_dict: ', len(new_sencell_dict))
    return new_sencell_dict

def get_sengene_cover(old_sengene_ls, sengene_ls):
    if isinstance(old_sengene_ls, torch.Tensor):
        old_sengene_ls = old_sengene_ls.tolist()
    if isinstance(sengene_ls, torch.Tensor):
        sengene_ls= sengene_ls.tolist()
    
    set1 = set(old_sengene_ls)
    set2 = set(sengene_ls)
    set3 = set1.intersection(set2)
    print('sengene cover:', len(set3)/len(set2))

    return len(set3)/len(set2)


def save_objs(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print("obj saved", path)


def load_objs(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
    
    
def caculate_GSEA(adata,args,use_onemarker=False,one_marker=None):
    # p16: CDKN2A, use_onemarker= True, one_marker="CDKN2A"
    # p21: CDKN1A
    import pandas as pd
    import gseapy as gp
    
    gene_expression_df = pd.DataFrame(adata.X.T, index=adata.var.index, columns=adata.obs.index)
    if use_onemarker:
        all_marker_genes=[one_marker]
    else:
        all_marker_genes=load_markers(args)
        all_marker_genes=list(set([j for i in all_marker_genes for j in i]))

    # Prepare gene sets as a dictionary
    gene_sets = {'GeneSet1': all_marker_genes}

    # Run ssGSEA
    ssgsea_results = gp.ssgsea(data=gene_expression_df, gene_sets=gene_sets, sample_norm_method='rank', outdir=None)

    # Extract enrichment scores
    enrichment_scores = ssgsea_results.res2d

    return enrichment_scores
