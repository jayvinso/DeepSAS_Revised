import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd

import utils
from model_AE import reduction_AE
from model_GAT import GAEModel, PhenotypeEmbedding, PhenotypeGAEModel
from model_Sencell import Sencell
from model_Sencell import cell_optim, update_cell_embeddings

import logging
import os
import random
import datetime
import scanpy as sp

args = utils.parse_args()
print(vars(args))

args.output_dir=f"./outputs/{args.exp_name}"
print("Outputs dir:",args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# set random seed
seed=args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='# %Y-%m-%d %H:%M:%S')

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

# Part 1: load and process data
# cell_cluster_arr used in umap ploting
logger.info("====== Part 1: load and process data ======")
if 'data1' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data1()
elif 'rep' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_rep(args.exp_name)
elif 'example' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_example_data(path=args.input_data_count, ct_name='clusters')
else:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data1(args.input_data_count)
        
new_data, markers_index,\
    sen_gene_ls, nonsen_gene_ls, gene_names = utils.process_data(
        adata, cluster_cell_ls, cell_cluster_arr,args)

new_data.write_h5ad(os.path.join(args.output_dir, f'{args.exp_name}_new_data.h5ad'))

gene_cell = new_data.X.toarray().T
args.gene_num = gene_cell.shape[0]
args.cell_num = gene_cell.shape[1]

print(f'cell num: {new_data.shape[0]}, gene num: {new_data.shape[1]}')

def resolve_age_phenotype_inputs(obs):
    age_bucket_to_value = {
        'young': 30.0,
        'middle-aged': 50.0,
        'old': 70.0,
    }

    if 'Age' in obs.columns:
        numeric_age = pd.to_numeric(obs['Age'], errors='coerce')
        if numeric_age.notna().all():
            age_bucket = pd.cut(
                numeric_age,
                bins=[-np.inf, 39, 59, np.inf],
                labels=['young', 'middle-aged', 'old'],
            ).astype(str)
            phenotype_values = age_bucket.map(age_bucket_to_value).to_numpy(dtype=np.float32)
            return phenotype_values, age_bucket, 'Age'

    for column_name in ['Age_Status', 'Old.Young']:
        if column_name not in obs.columns:
            continue

        normalized = obs[column_name].astype(str).str.strip().str.lower()
        category_map = {
            'young': 'young',
            'middle': 'middle-aged',
            'middle-aged': 'middle-aged',
            'middle aged': 'middle-aged',
            'old': 'old',
        }
        mapped = normalized.map(category_map)
        if mapped.isna().any():
            missing_labels = sorted(set(normalized[mapped.isna()]))
            raise ValueError(
                f'Unsupported age phenotype labels in {column_name}: {missing_labels}'
            )

        phenotype_values = mapped.map(age_bucket_to_value).to_numpy(dtype=np.float32)
        return phenotype_values, mapped, column_name

    return None, None, None


def resolve_disease_status_inputs(obs):
    healthy_labels = {
        'healthy',
        'control',
        'controls',
        'healthy control',
        'healthy controls',
        'normal',
        'non-ipf',
        'non ipf',
    }
    disease_columns = [
        'Status',
        'Disease_Status',
        'Disease Status',
        'Disease',
        'Condition',
        'condition',
        'Diagnosis',
        'diagnosis',
    ]

    for column_name in disease_columns:
        if column_name not in obs.columns:
            continue

        normalized = obs[column_name].astype(str).str.strip().str.lower()
        if normalized.isin(['', 'nan', 'none']).any():
            missing_labels = sorted(set(normalized[normalized.isin(['', 'nan', 'none'])]))
            raise ValueError(
                f'Unsupported missing disease status labels in {column_name}: {missing_labels}'
            )

        disease_values = (~normalized.isin(healthy_labels)).astype(np.int64).to_numpy()
        display_labels = normalized.map(
            lambda value: 'healthy' if value in healthy_labels else value.upper()
        )
        binary_labels = pd.Series(
            np.where(disease_values == 0, 'healthy', 'disease'),
            index=obs.index,
        )
        return disease_values, display_labels, binary_labels, column_name

    return None, None, None, None


#NEW START: phenotype metadata resolution and runtime gating
# Resolve phenotype metadata from processed cell observations before graph construction.
# Age metadata is the required signal; disease status is optional auxiliary context.
if args.phenotype_attention:
    cell_age_values, cell_age_buckets, phenotype_source = resolve_age_phenotype_inputs(new_data.obs)
    disease_status_values, disease_status_labels, disease_binary_labels, disease_source = resolve_disease_status_inputs(new_data.obs)
    use_phenotype_attention = cell_age_values is not None

    if cell_age_values is None:
        raise ValueError(
            'Phenotype attention is enabled, but no supported phenotype metadata was detected. '
            'Expected one of: Age, Age_Status, or Old.Young. '
            'Use --no-phenotype_attention to run without phenotype attention.'
        )
else:
    cell_age_values = None
    cell_age_buckets = None
    phenotype_source = None
    disease_status_values = None
    disease_status_labels = None
    disease_binary_labels = None
    disease_source = None
    use_phenotype_attention = False

# Persist the resolved phenotype mode on args so later helpers can branch on
# the actual runtime behavior rather than only the CLI request.
args.use_phenotype_attention = use_phenotype_attention
if not use_phenotype_attention:
    args.mixed_precision = False

if use_phenotype_attention:
    # Expand cell-level phenotype values onto the combined gene+cell graph so the
    # phenotype-aware encoder can consume them without changing the graph layout.
    if disease_status_values is None:
        disease_status_values = np.full(args.cell_num, -1, dtype=np.int64)
        disease_status_labels = pd.Series(['unknown'] * args.cell_num, index=new_data.obs.index)
        disease_binary_labels = pd.Series(['unknown'] * args.cell_num, index=new_data.obs.index)
        disease_source = 'none'

    node_ages, node_disease_status, is_gene_mask = PhenotypeEmbedding.build_node_phenotype_inputs(
        cell_age_values,
        disease_status_values,
        num_genes=args.gene_num,
        age_dtype=torch.float32,
    )
    age_bucket_counts = cell_age_buckets.value_counts().to_dict()
    disease_counts = disease_status_labels.value_counts().to_dict()
    disease_binary_counts = disease_binary_labels.value_counts().to_dict()
    logger.info(
        'Phenotype attention enabled using %s with age buckets young=18-39, middle-aged=40-59, old=60+. Distribution: %s',
        phenotype_source,
        age_bucket_counts,
    )
    logger.info(
        'Disease status phenotype enabled using %s. Labels: %s; binary distribution: %s',
        disease_source,
        disease_counts,
        disease_binary_counts,
    )
else:
    node_ages = None
    node_disease_status = None
    is_gene_mask = None
    logger.info('Phenotype attention disabled by configuration; using standard GAT encoder.')
#NEW END: phenotype metadata resolution and runtime gating

if args.retrain:
    graph_nx,edge_indexs,ccc_matrix = utils.build_graph_nx(
        new_data,gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names,args)


logger.info("Part 1, data loading and processing end!")


# Part 2: generate init embedding
logger.info("====== Part 2: generate init embedding ======")
use_autoencoder=False

device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
print('device:', device)
args.device = device

def run_scanpy(adata,batch_remove=False,batch_name='Sample'):
    adata=adata.copy()
    sp.pp.normalize_total(adata, target_sum=1e4)
    sp.pp.log1p(adata)
    sp.pp.scale(adata, max_value=10)
    if batch_remove:
        print(f'Start remove batch effect, the default batch annotation in adata.obs["{batch_name}"] ...')
        sp.pp.combat(adata, key=batch_name)
    else:
        print('Do not remove batch effect ...')
    sp.tl.pca(adata, svd_solver='arpack')
    sp.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sp.tl.umap(adata,n_components=args.emb_size)
    
    return adata.obsm['X_umap']

if use_autoencoder:
    if args.retrain:
        gene_embed, cell_embed = reduction_AE(gene_cell, device, use_amp=args.mixed_precision)
        print(gene_embed.shape, cell_embed.shape)
        torch.save(gene_embed, os.path.join(
            args.output_dir, f'{args.exp_name}_gene.emb'))
        torch.save(cell_embed, os.path.join(
            args.output_dir, f'{args.exp_name}_cell.emb'))
    else:
        print('skip training!')
        gene_embed = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_gene.emb'))
        cell_embed = torch.load(os.path.join(
            args.output_dir, f'{args.exp_name}_cell.emb'))

    if args.retrain:
        graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed,edge_indexs)
        torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        torch.save(graph_pyg, os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))

    else:
        graph_nx = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed)
        
else:
    if args.retrain:
        #NEW START: example-specific batch metadata selection
        if 'example' in args.exp_name:
            batch_name = 'Sample'
        else:
            batch_name = 'Sample.ID'
        #NEW END: example-specific batch metadata selection
        cell_embed=run_scanpy(new_data.copy(),batch_remove=True,batch_name=batch_name)
        print('cell embedding generated!')
        gene_embed=run_scanpy(new_data.copy().T)
        print('gene embedding generated!')
        cell_embed=torch.tensor(cell_embed)
        gene_embed=torch.tensor(gene_embed)
        
        graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed,edge_indexs,ccc_matrix)
        torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        torch.save(graph_pyg, os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
        print('graph nx and pyg saved!')
    else:
        print('Load graph nx and pyg ...')
        graph_nx=torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        graph_pyg=torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))

#NEW START: attach phenotype tensors to graph
if use_phenotype_attention:
    graph_pyg.node_ages = node_ages
    graph_pyg.node_disease_status = node_disease_status
    graph_pyg.is_gene_mask = is_gene_mask
#NEW END: attach phenotype tensors to graph

logger.info("Part 2, AE end!")
logger.info("====== Part 3: GAT training ======")

data = graph_pyg
data=data.to(device)
torch.cuda.empty_cache() 


#NEW START: phenotype-aware graph wrapper helpers
def encode_graph(model, graph_data):
    # Keep the downstream training loop agnostic to whether the standard or
    # phenotype-aware encoder is active.
    if use_phenotype_attention:
        return model.encode(
            graph_data.x,
            graph_data.edge_index,
            graph_data.node_ages,
            graph_data.is_gene_mask,
            getattr(graph_data, 'node_disease_status', None),
        )
    return model.encode(graph_data.x, graph_data.edge_index)


def forward_graph(model, graph_data):
    # Mirror encode_graph so later contrastive-learning code can reuse one path.
    if use_phenotype_attention:
        return model(
            graph_data.x,
            graph_data.edge_index,
            graph_data.node_ages,
            graph_data.is_gene_mask,
            getattr(graph_data, 'node_disease_status', None),
        )
    return model(graph_data.x, graph_data.edge_index)


def get_graph_attention_scores(model, graph_data):
    return model.get_attention_scores(graph_data)
#NEW END: phenotype-aware graph wrapper helpers


if args.retrain:
    # Initialize model and optimizer
    if use_phenotype_attention:
        # Swap in the phenotype-aware dual-head encoder only when age metadata
        # was resolved successfully from the processed AnnData object.
        model = PhenotypeGAEModel(args.emb_size, args.emb_size, age_boundaries=(40.0, 60.0)).to(device)
        logger.info('Phenotype-aware DualHeadPhenotypeGAT is active for this run.')
    else:
        model = GAEModel(args.emb_size, args.emb_size).to(device)
        logger.info('Using standard GAT encoder because phenotype attention is unavailable for this dataset.')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    use_amp = args.mixed_precision and device.type == 'cuda'
    def train():
        model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            z = encode_graph(model, data)
            loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # Training the model
    for epoch in range(args.gat_epoch):
        loss = train()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

    GAT_path=os.path.join(args.output_dir, f'{args.exp_name}_GAT.pt')
    torch.save(model, GAT_path)
    print(f'GAT model saved! {GAT_path}')
else:
    GAT_path=os.path.join(args.output_dir, f'{args.exp_name}_GAT.pt')
    print(f'Load GAT from {GAT_path}')
    model=torch.load(GAT_path)
    model=model.to(device)
    if use_phenotype_attention:
        logger.info('Loaded phenotype-aware GAT model and phenotype attention metadata.')
    
torch.cuda.empty_cache() 


logger.info("Part 2, train GAT end!")

logger.info("====== Part 4: Contrastive learning ======")

def check_celltypes(predicted_cell_indexs,graph_nx,celltype_names):
    cell_types=[]
    for i in predicted_cell_indexs:
        cluster=graph_nx.nodes[int(i)]['cluster']
        cell_types.append(celltype_names[cluster])
    from collections import Counter
    print("snc in different cell types: ",Counter(cell_types))
    

def build_cell_dict(gene_cell,predicted_cell_indexs,GAT_embeddings,graph_nx):
    sencell_dict={}
    nonsencell_dict={}

    for i in range(gene_cell.shape[0],gene_cell.shape[0]+gene_cell.shape[1]):
        if i in predicted_cell_indexs:
            sencell_dict[i]=[
                GAT_embeddings[i],
                graph_nx.nodes[int(i)]['cluster'],
                0,
                i]
        else:
            nonsencell_dict[i]=[
                GAT_embeddings[i],
                graph_nx.nodes[int(i)]['cluster'],
                0,
                i]
    return sencell_dict,nonsencell_dict


        
def identify_sengene_v1(sencell_dict, gene_cell, edge_index_selfloop, attention_scores, sen_gene_ls):
    print("identify_sengene_v1 ... ")
    if use_phenotype_attention and len(sencell_dict) == 0:
        print('No senescent cells identified in this epoch; keeping previous senescence gene list.')
        return sen_gene_ls

    cell_index = torch.tensor(list(sencell_dict.keys()))
    cell_mask = torch.zeros(gene_cell.shape[0] + gene_cell.shape[1], dtype=torch.bool)
    cell_mask[cell_index] = True

    res = []
    score_sengene_ls = []

    for gene_index in range(gene_cell.shape[0]):
        connected_cells = edge_index_selfloop[0][edge_index_selfloop[1] == gene_index]
        masked_connected_cells = connected_cells[cell_mask[connected_cells]]

        if masked_connected_cells.numel() == 0:
            res.append(0)  # Store as integer, less memory
        else:
            tmp = attention_scores[edge_index_selfloop[1] == gene_index]
            attention_edge = torch.sum(tmp[cell_mask[connected_cells]], dim=1)
            attention_s = torch.mean(attention_edge)
            res.append(attention_s.item())  # Convert to Python scalar

        if gene_index in sen_gene_ls:
            score_sengene_ls.append(res[-1])
           
    # number of updated genes
    num=10
    res1=torch.tensor(res)
    new_genes=torch.argsort(res1)[-num:]
    score_sengene_ls=torch.tensor(score_sengene_ls)
    if isinstance(sen_gene_ls, torch.Tensor):
        new_sen_gene_ls=sen_gene_ls[torch.argsort(score_sengene_ls)[num:].tolist()]
    else:
        new_sen_gene_ls=torch.tensor(sen_gene_ls)[torch.argsort(score_sengene_ls)[num:].tolist()]
    new_sen_gene_ls=torch.cat((new_sen_gene_ls,new_genes))
    #print(sen_gene_ls)
    #print(new_sen_gene_ls)
    # threshold = torch.mean(res1) + 2*torch.std(res1)  # Example threshold
    # print("the number of identified sen genes:", res1[res1>threshold].shape)
    return new_sen_gene_ls


def get_sorted_sengene(sencell_dict,gene_cell,edge_index_selfloop,attention_scores,sen_gene_ls):
    attention_scores=attention_scores.to('cpu')
    edge_index_selfloop=edge_index_selfloop.to('cpu')
    
    cell_index=torch.tensor(list(sencell_dict.keys()))
    cell_mask = torch.zeros(gene_cell.shape[0]+gene_cell.shape[1], dtype=torch.bool)
    cell_mask[cell_index] = True

    gene_index=0
    res=[]
    score_sengene_ls=[]
    while gene_index<gene_cell.shape[0]:
        connected_cells=edge_index_selfloop[0][edge_index_selfloop[1] == gene_index]
        if len(connected_cells[cell_mask[connected_cells]])==0:
            res.append(torch.tensor(0))
            # print('no sencell in this gene')
        else:
            attention_edge=torch.sum(attention_scores[edge_index_selfloop[1] == gene_index][cell_mask[connected_cells]],axis=1)
            attention_s=torch.mean(attention_edge)
            res.append(attention_s)
        if gene_index in sen_gene_ls:
            score_sengene_ls.append(res[-1])
        gene_index+=1
        
    res1=torch.tensor(res)
    score_sengene_ls=torch.tensor(score_sengene_ls)
    sorted_sengene_ls=torch.tensor(sen_gene_ls)[torch.argsort(score_sengene_ls).tolist()]
    return sorted_sengene_ls



def calculate_outliers(scores):
    counts=0
    snc_index=[]
    
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    for i,score in enumerate(scores): 
        if score > upper_bound:
            counts+=1
            snc_index.append(i)
    
    
    return counts,snc_index



def generate_ct_specific_scores(sen_gene_ls,gene_cell,edge_index_selfloop,
                                attention_scores,graph_nx,celltype_names):
    print('generate_ct_specific_scores ...')
    # attention_scores=attention_scores.to('cpu')
    # edge_index_selfloop=edge_index_selfloop.to('cpu')
    
    gene_index=torch.tensor(sen_gene_ls)

    gene_mask = torch.zeros(gene_cell.shape[0]+gene_cell.shape[1], dtype=torch.bool)
    gene_mask[gene_index] = True

    # res=[]
    
    # key is cluster index, value is a 2d list, each row: [score, cell index]
    ct_specific_scores={}

    for cell_index in range(gene_cell.shape[0],gene_cell.shape[0]+gene_cell.shape[1]):
        connected_genes=edge_index_selfloop[0][edge_index_selfloop[1] == cell_index]
        # Consider that if the cell does not express any senescence-associated genes, 
        # set the score to 0, otherwise PyTorch will calculate it as NaN, which requires additional handling
        if len(connected_genes[gene_mask[connected_genes]])==0:
            print('no sengene in this cell!')
            # res.append(torch.tensor(0))
        else:
            attention_edge=torch.sum(attention_scores[edge_index_selfloop[1] == cell_index][gene_mask[connected_genes]],axis=1)
            attention_s=torch.mean(attention_edge)
            # res.append(attention_s)
            
            cluster=graph_nx.nodes[int(cell_index)]['cluster']
            if cluster in ct_specific_scores:
                ct_specific_scores[cluster].append([float(attention_s),int(cell_index)])
            else:
                ct_specific_scores[cluster]=[[float(attention_s),int(cell_index)]]
            
    
    return ct_specific_scores


def calculate_outliers_v1(scores_index):
    scores_index=np.array(scores_index)
    
    counts=0
    snc_index=[]
    
    outliers_ls=[]
    
    scores=scores_index[:,0]
    indexs=scores_index[:,1]
    
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    for i,score in enumerate(scores): 
        if score > upper_bound:
            counts+=1
            snc_index.append(indexs[i])
            outliers_ls.append([score,indexs[i]])
    
    
    return counts,snc_index,outliers_ls


def calculate_outlier_bounds(scores):
    if len(scores) == 0:
        return np.nan, np.nan, np.nan

    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound, IQR


def summarize_snc_score_group(epoch, group_type, group_name, scores_index,
                              selected_cell_indexs, threshold_info=None):
    if len(scores_index) == 0:
        return None

    scores_index = np.array(scores_index)
    scores = scores_index[:, 0].astype(float)
    cell_indexs = scores_index[:, 1].astype(int)

    if threshold_info is None:
        lower_bound, upper_bound, iqr = calculate_outlier_bounds(scores)
        threshold_source = 'mixed_cell_type'
        baseline_count = len(scores)
        threshold_center = float(np.percentile(scores, 50))
        mad = np.nan
        mad_multiplier = np.nan
        candidate_count = None
        min_hub_cells = np.nan
        hub_pruned = False
    else:
        lower_bound = threshold_info['lower_bound']
        upper_bound = threshold_info['upper_bound']
        iqr = threshold_info.get('iqr', np.nan)
        threshold_source = threshold_info['threshold_source']
        baseline_count = threshold_info['baseline_count']
        threshold_center = threshold_info.get('threshold_center', np.nan)
        mad = threshold_info.get('mad', np.nan)
        mad_multiplier = threshold_info.get('mad_multiplier', np.nan)
        candidate_count = threshold_info.get('candidate_count')
        min_hub_cells = threshold_info.get('min_hub_cells', np.nan)
        hub_pruned = threshold_info.get('hub_pruned', False)

    outlier_mask = scores > upper_bound
    if candidate_count is None:
        candidate_count = int(outlier_mask.sum())
    selected_mask = np.array([cell_index in selected_cell_indexs
                              for cell_index in cell_indexs])

    row = {
        'epoch': epoch,
        'group_type': group_type,
        'group': group_name,
        'n_scored_cells': len(scores),
        'n_outliers': int(outlier_mask.sum()),
        'n_selected_snc': int(selected_mask.sum()),
        'selected_fraction': float(selected_mask.mean()),
        'min': float(np.min(scores)),
        'p01': float(np.percentile(scores, 1)),
        'p05': float(np.percentile(scores, 5)),
        'p25': float(np.percentile(scores, 25)),
        'median': float(np.percentile(scores, 50)),
        'mean': float(np.mean(scores)),
        'p75': float(np.percentile(scores, 75)),
        'p95': float(np.percentile(scores, 95)),
        'p99': float(np.percentile(scores, 99)),
        'max': float(np.max(scores)),
        'std': float(np.std(scores)),
        'iqr': float(iqr),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'threshold_source': threshold_source,
        'threshold_baseline_count': int(baseline_count),
        'threshold_center': float(threshold_center),
        'mad': float(mad),
        'mad_multiplier': float(mad_multiplier),
        'candidate_count': int(candidate_count),
        'min_hub_cells': float(min_hub_cells),
        'hub_pruned': bool(hub_pruned),
    }
    return row


def log_snc_score_diagnostics(epoch, ct_specific_scores, predicted_cell_indexs,
                              graph_nx, celltype_names, output_dir, exp_name,
                              cell_age_buckets=None, cell_disease_labels=None,
                              gene_num=None, threshold_diagnostics=None):
    selected_cell_indexs = set(int(cell_index)
                               for cell_index in predicted_cell_indexs)
    rows = []

    all_scores_index = []
    for cluster, scores_index in ct_specific_scores.items():
        if len(scores_index) == 0:
            continue

        all_scores_index.extend(scores_index)
        celltype_name = celltype_names[cluster]
        row = summarize_snc_score_group(
            epoch,
            'cell_type',
            celltype_name,
            scores_index,
            selected_cell_indexs,
            None if threshold_diagnostics is None
            else threshold_diagnostics.get(cluster),
        )
        if row is not None:
            rows.append(row)

    overall_row = summarize_snc_score_group(
        epoch,
        'overall',
        'all',
        all_scores_index,
        selected_cell_indexs,
    )
    if overall_row is not None:
        rows.insert(0, overall_row)

    if cell_age_buckets is not None and gene_num is not None:
        age_bucket_scores = {}
        for score, node_index in all_scores_index:
            cell_offset = int(node_index) - gene_num
            if cell_offset < 0 or cell_offset >= len(cell_age_buckets):
                continue
            age_bucket = str(cell_age_buckets.iloc[cell_offset])
            age_bucket_scores.setdefault(age_bucket, []).append([score, node_index])

        for age_bucket, scores_index in sorted(age_bucket_scores.items()):
            row = summarize_snc_score_group(
                epoch,
                'age_bucket',
                age_bucket,
                scores_index,
                selected_cell_indexs,
            )
            if row is not None:
                rows.append(row)

    if cell_disease_labels is not None and gene_num is not None:
        disease_status_scores = {}
        for score, node_index in all_scores_index:
            cell_offset = int(node_index) - gene_num
            if cell_offset < 0 or cell_offset >= len(cell_disease_labels):
                continue
            disease_status = str(cell_disease_labels.iloc[cell_offset])
            disease_status_scores.setdefault(disease_status, []).append([score, node_index])

        for disease_status, scores_index in sorted(disease_status_scores.items()):
            row = summarize_snc_score_group(
                epoch,
                'disease_status',
                disease_status,
                scores_index,
                selected_cell_indexs,
            )
            if row is not None:
                rows.append(row)

    if len(rows) == 0:
        logger.info('SnC score diagnostics epoch %03d: no score rows were available.', epoch)
        return

    diagnostics_path = os.path.join(
        output_dir, f'{exp_name}_snc_score_diagnostics.csv')
    diagnostics_df = pd.DataFrame(rows)
    write_mode = 'w' if epoch == 0 else 'a'
    write_header = epoch == 0 or not os.path.exists(diagnostics_path)
    diagnostics_df.to_csv(diagnostics_path, mode=write_mode, header=write_header,
                          index=False)

    if overall_row is not None:
        logger.info(
            'SnC score diagnostics epoch %03d: selected=%d/%d, p95=%.6g, p99=%.6g, max=%.6g; wrote %s',
            epoch,
            overall_row['n_selected_snc'],
            overall_row['n_scored_cells'],
            overall_row['p95'],
            overall_row['p99'],
            overall_row['max'],
            diagnostics_path,
        )

    selected_celltype_rows = [
        row for row in rows
        if row['group_type'] == 'cell_type' and row['n_selected_snc'] > 0
    ]
    if selected_celltype_rows:
        selected_summary = ', '.join(
            f"{row['group']}: selected={row['n_selected_snc']}, "
            f"outliers={row['n_outliers']}, upper={row['upper_bound']:.6g}, "
            f"max={row['max']:.6g}, threshold={row['threshold_source']}"
            for row in selected_celltype_rows
        )
        logger.info('SnC score diagnostics epoch %03d selected cell types: %s',
                    epoch, selected_summary)
    else:
        celltype_rows = [row for row in rows if row['group_type'] == 'cell_type']
        if celltype_rows:
            strongest_row = max(celltype_rows, key=lambda row: row['n_outliers'])
            logger.info(
                'SnC score diagnostics epoch %03d: no cell type passed selection; strongest outlier count was %d in %s.',
                epoch,
                strongest_row['n_outliers'],
                strongest_row['group'],
            )


#NEW START: phenotype-specific MAD thresholding
# Phenotype runs use a per-cell-type MAD threshold instead of the legacy IQR +
# minimum-outlier rule so small phenotype cohorts do not force mixed thresholds.
PHENOTYPE_MAD_MULTIPLIER = 3.0
PHENOTYPE_MIN_HUB_CELLS = 3


def calculate_mad_outliers(scores_index, mad_multiplier=PHENOTYPE_MAD_MULTIPLIER,
                           min_hub_cells=PHENOTYPE_MIN_HUB_CELLS):
    # Use a robust per-cell-type threshold for phenotype runs so age/disease
    # context does not get mixed into a single global outlier cutoff.
    scores_index = np.array(scores_index)
    scores = scores_index[:, 0].astype(float)
    indexs = scores_index[:, 1].astype(int)
    center = np.median(scores)
    mad = np.median(np.abs(scores - center))
    lower_bound = center - mad_multiplier * mad
    upper_bound = center + mad_multiplier * mad
    selected_mask = scores > upper_bound
    candidate_count = int(selected_mask.sum())
    hub_pruned = candidate_count < min_hub_cells

    threshold_info = {
        'threshold_source': 'cell_type_mad',
        'baseline_count': len(scores),
        'iqr': np.nan,
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'threshold_center': float(center),
        'mad': float(mad),
        'mad_multiplier': float(mad_multiplier),
        'candidate_count': candidate_count,
        'min_hub_cells': int(min_hub_cells),
        'hub_pruned': bool(hub_pruned),
    }

    if hub_pruned:
        return 0, [], [], threshold_info

    selected_scores = scores[selected_mask]
    selected_indexs = indexs[selected_mask]
    outliers_ls = [[float(score), int(index)]
                   for score, index in zip(selected_scores, selected_indexs)]

    return int(selected_mask.sum()), selected_indexs.astype(int).tolist(), outliers_ls, threshold_info
#NEW END: phenotype-specific MAD thresholding


def extract_cell_indexs(ct_specific_scores, min_outliers=10):
    import numpy as np
    from scipy.special import softmax
    
    print("extract_cell_indexs ... ")
    
    data_for_plotting = []
    categories = []

    snc_indexs=[]

    ct_specific_outliers={}
    strongest_count = 0
    #NEW START: threshold diagnostics propagation
    threshold_diagnostics = {}


    for key, values in ct_specific_scores.items():
        values_ls=np.array(values)
        data_for_plotting.extend(values_ls[:,0])
        categories.extend([celltype_names[key]] * len(values))

        if use_phenotype_attention:
            # Phenotype injection scores each cell type against its own robust
            # MAD threshold and keeps only stable local hubs.
            counts, snc_index, outliers_ls, threshold_info = calculate_mad_outliers(values_ls)
            threshold_diagnostics[key] = threshold_info
            candidate_count = threshold_info['candidate_count']
        else:
            counts,snc_index,outliers_ls=calculate_outliers_v1(values_ls)
            threshold_diagnostics[key] = None
            candidate_count = counts

        if candidate_count > strongest_count:
            strongest_count = candidate_count

        if use_phenotype_attention:
            if counts > 0:
                ct_specific_outliers[key]=outliers_ls
                snc_indexs=snc_indexs+snc_index
        elif counts>=min_outliers:
            ct_specific_outliers[key]=outliers_ls
            snc_indexs=snc_indexs+snc_index

    if use_phenotype_attention and len(snc_indexs) == 0:
        logger.info(
            'No cell type produced at least %d MAD-threshold SnCs; strongest cell-type candidate count was %d. Contrastive cell optimization will be skipped for this epoch.',
            PHENOTYPE_MIN_HUB_CELLS,
            strongest_count,
        )

    return snc_indexs, threshold_diagnostics
    #NEW END: threshold diagnostics propagation


cellmodel = Sencell(args.emb_size).to(device)
data=data.to(device)
lr=0.01
optimizer = torch.optim.Adam(cellmodel.parameters(), lr=lr,
                        weight_decay=1e-3)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.85)
sencell_dict=None


# Extracting attention scores
edge_index_selfloop_cell,attention_scores_cell = get_graph_attention_scores(model, data)
attention_scores_cell=attention_scores_cell.cpu().detach()
edge_index_selfloop_cell=edge_index_selfloop_cell.cpu().detach()

model.eval()
for epoch in range(5):
    print(f'{datetime.datetime.now()}: Contrastive learning Epoch: {epoch:03d}')
    
    # cell part
    ct_specific_scores=generate_ct_specific_scores(sen_gene_ls,gene_cell,
                                                    edge_index_selfloop_cell,
                                                    attention_scores_cell,
                                                    graph_nx,celltype_names)
    

    #NEW START: phenotype diagnostics and empty-SnC safeguards in contrastive loop
    predicted_cell_indexs, threshold_diagnostics=extract_cell_indexs(ct_specific_scores)
    if use_phenotype_attention:
        # The phenotype branch logs richer diagnostics because its thresholds are
        # dynamic and cell-type-local rather than a single global heuristic.
        log_snc_score_diagnostics(epoch, ct_specific_scores, predicted_cell_indexs,
                                  graph_nx, celltype_names, args.output_dir,
                                  args.exp_name, cell_age_buckets,
                                  disease_status_labels, args.gene_num,
                                  threshold_diagnostics)
    check_celltypes(predicted_cell_indexs,graph_nx,celltype_names)
    
    if sencell_dict is not None:
        old_sencell_dict=sencell_dict
    else:
        old_sencell_dict=None
    
    GAT_embeddings=forward_graph(model, data).detach()
    sencell_dict, nonsencell_dict=build_cell_dict(gene_cell,predicted_cell_indexs,GAT_embeddings,graph_nx)
    
    if old_sencell_dict is not None:
        # Phenotype-guided selection can legitimately produce an empty SnC set for
        # an epoch, so guard the legacy overlap metric and keep training stable.
        if use_phenotype_attention and len(sencell_dict) == 0:
            ratio_cell = 0.0
            print('sencell cover: 0.0 (current sencell set is empty)')
        else:
            ratio_cell = utils.get_sencell_cover(old_sencell_dict, sencell_dict)

    
    # gene part
    should_optimize_cells = len(sencell_dict) > 0 and len(nonsencell_dict) > 0
    cellmodel, sencell_dict, nonsencell_dict = cell_optim(cellmodel, optimizer,
                                                            sencell_dict, nonsencell_dict,
                                                            None,
                                                            args,
                                                            train=True)
    if (not use_phenotype_attention) or should_optimize_cells:
        scheduler.step()
    else:
        print('Skipping scheduler step because no cells were optimized in this epoch.')
    current_lr = optimizer.param_groups[0]['lr']
    print('current lr:',current_lr)
    
    # update gat embeddings

    new_GAT_embeddings=GAT_embeddings
    for key,value in sencell_dict.items():
        if use_phenotype_attention:
            # In phenotype mode, cell_optim may leave the updated embedding in a
            # different slot than the legacy branch, so normalize that here.
            updated_embedding = value[2] if torch.is_tensor(value[2]) else value[0]
            new_GAT_embeddings[key]=updated_embedding.detach()
        else:
            new_GAT_embeddings[key]=sencell_dict[key][2].detach()
    for key,value in nonsencell_dict.items():
        if use_phenotype_attention:
            updated_embedding = value[2] if torch.is_tensor(value[2]) else value[0]
            new_GAT_embeddings[key]=updated_embedding.detach()
        else:
            new_GAT_embeddings[key]=nonsencell_dict[key][2].detach()
        
    data.x=new_GAT_embeddings
    edge_index_selfloop,attention_scores = get_graph_attention_scores(model, data)
    
    attention_scores=attention_scores.to('cpu')
    edge_index_selfloop=edge_index_selfloop.to('cpu')
    if use_phenotype_attention:
        # Reuse the refreshed attention tensors for phenotype diagnostics and the
        # next epoch's phenotype-aware cell scoring.
        attention_scores_cell=attention_scores.detach()
        edge_index_selfloop_cell=edge_index_selfloop.detach()
    #NEW END: phenotype diagnostics and empty-SnC safeguards in contrastive loop
    
    old_sengene_indexs=sen_gene_ls
    #     sen_gene_ls=identify_sengene_v2(new_data,
    #         sencell_dict,gene_cell,edge_index_selfloop,attention_scores,sen_gene_ls)
    #     ratio_gene = utils.get_sengene_cover(old_sengene_indexs, sen_gene_ls)
    sen_gene_ls=identify_sengene_v1(
        sencell_dict,gene_cell,edge_index_selfloop,attention_scores,sen_gene_ls)
    ratio_gene = utils.get_sengene_cover(old_sengene_indexs, sen_gene_ls)
    
    torch.save([sencell_dict,sen_gene_ls,attention_scores,edge_index_selfloop],
               os.path.join(args.output_dir, f'{args.exp_name}_sencellgene-epoch{epoch}.data'))
    
