import json
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

import utils
from model_AE import reduction_AE
from model_GAT import GAEModel
from model_Sencell import Sencell, get_cluster_cell_dict, getPrototypeEmb
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
logger.info("====== Part 1: load and process data ======")
if 'data1' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data1()
elif 'rep' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_rep(args.exp_name)
elif 'example' in args.exp_name:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_example_data()
else:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data1(args.input_data_count)

# Ensure phenotype exists
if 'Condition' not in adata.obs.columns:
    raise ValueError("Missing 'Condition' column in adata.obs. Please ensure it's added before running.")

if args.use_hvg_deg:
    print("Running per-cell-type HVG filtering...")

    import scanpy as sc

    all_selected_genes = set()

    for ct in adata.obs['clusters'].unique():
        print(f"Processing cell type: {ct}")
        adata_ct = adata[adata.obs['clusters'] == ct].copy()

        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)

        # Select top 10,000 highly variable genes
        sc.pp.highly_variable_genes(adata_ct, n_top_genes=1000, flavor='seurat')
        hvg = adata_ct.var[adata_ct.var['highly_variable']].index.tolist()

        print(f"[{ct}] Selected {len(hvg)} HVGs.")
        all_selected_genes.update(hvg)

    # Final filtering
    print(f"Total unique HVGs across all cell types: {len(all_selected_genes)}")
    adata = adata[:, list(all_selected_genes)]



new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names = utils.process_data(
    adata, cluster_cell_ls, cell_cluster_arr, args)

new_data.write_h5ad(os.path.join(args.output_dir, f'{args.exp_name}_new_data.h5ad'))

gene_cell = new_data.X.toarray().T
args.gene_num = gene_cell.shape[0]
args.cell_num = gene_cell.shape[1]

print(f'cell num: {new_data.shape[0]}, gene num: {new_data.shape[1]}')

if args.retrain:
    graph_nx, edge_indexs, ccc_matrix = utils.build_graph_nx(
        new_data, gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names, args)
    
    # Add phenotype to graph_nx nodes
    for i in range(gene_cell.shape[1]):
        graph_nx.nodes[i + gene_cell.shape[0]]['Condition'] = new_data.obs['Condition'][i]

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
        print('Start remove batch effect, the default batch annotation in adata.obs["Sample"] ...')
        sp.pp.combat(adata, key=batch_name)
    else:
        print('Do not remove batch effect ...')
    sp.tl.pca(adata, svd_solver='arpack')
    sp.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sp.tl.umap(adata,n_components=args.emb_size)
    
    return adata.obsm['X_umap']

# === Additional functions for phenotype-specific scoring ===
def generate_pheno_specific_scores(sen_gene_ls, gene_cell, edge_index_selfloop, attention_scores, graph_nx):
    print('generate_pheno_specific_scores ...')
    gene_index = torch.tensor(sen_gene_ls)
    gene_mask = torch.zeros(gene_cell.shape[0]+gene_cell.shape[1], dtype=torch.bool)
    gene_mask[gene_index] = True

    pheno_specific_scores = {}

    for cell_index in range(gene_cell.shape[0], gene_cell.shape[0] + gene_cell.shape[1]):
        connected_genes = edge_index_selfloop[0][edge_index_selfloop[1] == cell_index]
        if len(connected_genes[gene_mask[connected_genes]]) == 0:
            continue
        else:
            attention_edge = torch.sum(attention_scores[edge_index_selfloop[1] == cell_index][gene_mask[connected_genes]], axis=1)
            attention_s = torch.mean(attention_edge)

            phenotype = graph_nx.nodes[int(cell_index)]['Condition']
            if phenotype in pheno_specific_scores:
                pheno_specific_scores[phenotype].append([float(attention_s), int(cell_index)])
            else:
                pheno_specific_scores[phenotype] = [[float(attention_s), int(cell_index)]]

    return pheno_specific_scores

def extract_phenotype_specific_indexs(pheno_specific_scores):
    print("extract_phenotype_specific_indexs ...")
    import numpy as np
    phenotype_snc_indexs = []
    for phenotype, values in pheno_specific_scores.items():
        values_ls = np.array(values)
        scores = values_ls[:, 0]
        indexs = values_ls[:, 1]
        Q1 = np.percentile(scores, 25)
        Q3 = np.percentile(scores, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        for i, score in enumerate(scores):
            if score > upper_bound:
                phenotype_snc_indexs.append(indexs[i])
    return phenotype_snc_indexs


if use_autoencoder:
    if args.retrain:
        gene_embed, cell_embed = reduction_AE(gene_cell, device)
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
        cell_embed=run_scanpy(new_data.copy(),batch_remove=True)
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

logger.info("Part 2, AE end!")
logger.info("====== Part 3: GAT training ======")

data = graph_pyg
data=data.to(device)
torch.cuda.empty_cache() 


if args.retrain:
    # Initialize model and optimizer
    model = GAEModel(args.emb_size, args.emb_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    gat_losses = []

    # Training loop
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # Training the model
    for epoch in range(args.gat_epoch):
        loss = train()
        gat_losses.append(loss)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

    GAT_path=os.path.join(args.output_dir, f'{args.exp_name}_GAT.pt')
    torch.save(model, GAT_path)
    print(f'GAT model saved! {GAT_path}')
else:
    GAT_path=os.path.join(args.output_dir, f'{args.exp_name}_GAT.pt')
    print(f'Load GAT from {GAT_path}')
    model=torch.load(GAT_path)
    model=model.to(device)
    
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


def extract_cell_indexs(ct_specific_scores):
    import numpy as np
    from scipy.special import softmax
    
    print("extract_cell_indexs ... ")
    
    data_for_plotting = []
    categories = []

    snc_indexs=[]

    ct_specific_outliers={}


    for key, values in ct_specific_scores.items():
        values_ls=np.array(values)
        data_for_plotting.extend(values_ls[:,0])
        categories.extend([celltype_names[key]] * len(values))

        counts,snc_index,outliers_ls=calculate_outliers_v1(values_ls)
        if counts>=10:
            ct_specific_outliers[key]=outliers_ls
            snc_indexs=snc_indexs+snc_index


    return snc_indexs


cellmodel = Sencell(args.emb_size).to(device)
data=data.to(device)
lr=0.01
optimizer = torch.optim.Adam(cellmodel.parameters(), lr=lr,
                        weight_decay=1e-3)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.85)
sencell_dict=None

epoch_metrics_log = []

# Extracting attention scores
edge_index_selfloop_cell,attention_scores_cell = model.get_attention_scores(data)
attention_scores_cell=attention_scores_cell.cpu().detach()
edge_index_selfloop_cell=edge_index_selfloop_cell.cpu().detach()

model.eval()
for epoch in range(5):
    print(f'{datetime.datetime.now()}: Contrastive learning Epoch: {epoch:03d}')

    # cell type–specific scoring
    ct_specific_scores = generate_ct_specific_scores(sen_gene_ls, gene_cell,
                                                    edge_index_selfloop_cell,
                                                    attention_scores_cell,
                                                    graph_nx, celltype_names)
    predicted_cell_indexs = extract_cell_indexs(ct_specific_scores)
    check_celltypes(predicted_cell_indexs, graph_nx, celltype_names)

    # phenotype-specific scoring
    pheno_specific_scores = generate_pheno_specific_scores(sen_gene_ls, gene_cell,
                                                           edge_index_selfloop_cell,
                                                           attention_scores_cell,
                                                           graph_nx)
    phenotype_predicted_indexs = extract_phenotype_specific_indexs(pheno_specific_scores)

    # merge both types of predictions (optional or use either)
    predicted_cell_indexs = list(set(predicted_cell_indexs + phenotype_predicted_indexs))

    if sencell_dict is not None:
        old_sencell_dict = sencell_dict
    else:
        old_sencell_dict = None

    GAT_embeddings = model(data.x, data.edge_index).detach()
    sencell_dict, nonsencell_dict = build_cell_dict(gene_cell, predicted_cell_indexs, GAT_embeddings, graph_nx)

    if old_sencell_dict is not None:
        ratio_cell = utils.get_sencell_cover(old_sencell_dict, sencell_dict)

    # gene part
    cellmodel, sencell_dict, nonsencell_dict, mdr_loss = cell_optim(cellmodel, optimizer,
                                                          sencell_dict, nonsencell_dict,
                                                          None,
                                                          args,
                                                          train=True)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print('current lr:', current_lr)

    # update gat embeddings
    new_GAT_embeddings = GAT_embeddings
    for key, value in sencell_dict.items():
        new_GAT_embeddings[key] = sencell_dict[key][2].detach()
    for key, value in nonsencell_dict.items():
        new_GAT_embeddings[key] = nonsencell_dict[key][2].detach()

    data.x = new_GAT_embeddings
    edge_index_selfloop, attention_scores = model.get_attention_scores(data)

    attention_scores = attention_scores.to('cpu')
    edge_index_selfloop = edge_index_selfloop.to('cpu')

    old_sengene_indexs = sen_gene_ls
    sen_gene_ls = identify_sengene_v1(
        sencell_dict, gene_cell, edge_index_selfloop, attention_scores, sen_gene_ls)
    ratio_gene = utils.get_sengene_cover(old_sengene_indexs, sen_gene_ls)

    epoch_entry = {
        'epoch': epoch,
        'mdr_loss': mdr_loss,
        'snc_cover_ratio': ratio_cell if old_sencell_dict is not None else None,
        'sng_cover_ratio': float(ratio_gene),
        'num_snc': len(sencell_dict),
        'num_sng': len(sen_gene_ls),
    }
    epoch_metrics_log.append(epoch_entry)
    logger.info(f"Epoch {epoch} | MDR loss: {mdr_loss:.6f} | SnC cover: {epoch_entry['snc_cover_ratio']} | SnG cover: {epoch_entry['sng_cover_ratio']:.4f} | #SnC: {epoch_entry['num_snc']} | #SnG: {epoch_entry['num_sng']}")

    torch.save([sencell_dict, sen_gene_ls, attention_scores, edge_index_selfloop],
               os.path.join(args.output_dir, f'{args.exp_name}_sencellgene-epoch{epoch}.data'))

# ============================================================
# POST-TRAINING EVALUATION METRICS
# ============================================================
logger.info("====== Post-training: Computing evaluation metrics ======")

from scipy.stats import mannwhitneyu
from sklearn.metrics import silhouette_score, roc_auc_score
import pandas as pd

post_metrics = {}

# ------------------------------------------------------------------
# 1a. Latent space separation: d1, d2, d3
# ------------------------------------------------------------------
logger.info("Computing d1 / d2 / d3 latent space distances ...")
cellmodel.eval()
with torch.no_grad():
    cluster_sencell, cluster_nonsencell = get_cluster_cell_dict(sencell_dict, nonsencell_dict)
    prototype_emb = getPrototypeEmb(sencell_dict, cluster_sencell)
    d1 = cellmodel.get_d1(sencell_dict, cluster_sencell, prototype_emb)
    d2 = cellmodel.get_d2(sencell_dict, cluster_sencell, prototype_emb)
    d3 = cellmodel.get_d3(nonsencell_dict, cluster_nonsencell, prototype_emb)

d1_flat = [d.item() for cluster in d1 for d in cluster]
d2_flat = [d.item() for cluster in d2 for d in cluster]
d3_flat = [d.item() for cluster in d3 for d in cluster]

post_metrics['mean_d1'] = float(np.mean(d1_flat)) if d1_flat else None
post_metrics['mean_d2'] = float(np.mean(d2_flat)) if d2_flat else None
post_metrics['mean_d3'] = float(np.mean(d3_flat)) if d3_flat else None
post_metrics['d1_lt_d2'] = bool(post_metrics['mean_d1'] < post_metrics['mean_d2']) if (post_metrics['mean_d1'] is not None and post_metrics['mean_d2'] is not None) else None
post_metrics['d1_lt_d3'] = bool(post_metrics['mean_d1'] < post_metrics['mean_d3']) if (post_metrics['mean_d1'] is not None and post_metrics['mean_d3'] is not None) else None
logger.info(f"  mean d1={post_metrics['mean_d1']:.4f}  d2={post_metrics['mean_d2']:.4f}  d3={post_metrics['mean_d3']:.4f}  d1<d2={post_metrics['d1_lt_d2']}  d1<d3={post_metrics['d1_lt_d3']}")

# ------------------------------------------------------------------
# 1b. Feature distribution concentration φ' (ProtoNCE prototype loss)
# ------------------------------------------------------------------
with torch.no_grad():
    phi_prime = cellmodel.prototypeLoss(d1)
post_metrics['phi_prime'] = float(phi_prime.item())
logger.info(f"  φ' (prototype concentration) = {post_metrics['phi_prime']:.6f}")

# ------------------------------------------------------------------
# 1c. Final MDR (contrastive) loss
# ------------------------------------------------------------------
with torch.no_grad():
    final_mdr = cellmodel.loss(sencell_dict, nonsencell_dict)
post_metrics['final_mdr_loss'] = float(final_mdr.item())
logger.info(f"  Final MDR loss = {post_metrics['final_mdr_loss']:.6f}")

# ------------------------------------------------------------------
# 2a. Hallmark gene enrichment: intersection with SenMayo/Fridman/CellAge
# ------------------------------------------------------------------
logger.info("Computing hallmark SnG enrichment ...")
marker_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'senescence_marker_list.csv')
if os.path.exists(marker_path):
    marker_df = pd.read_csv(marker_path)
    hallmark_genes = {}
    for col in marker_df.columns:
        hallmark_genes[col] = set(marker_df[col].dropna().str.strip().tolist())
    all_hallmarks = set().union(*hallmark_genes.values())

    predicted_gene_names = set(new_data.var_names[[int(g) for g in sen_gene_ls]])
    post_metrics['predicted_sng_count'] = len(predicted_gene_names)

    for hallmark_name, hallmark_set in hallmark_genes.items():
        intersection = hallmark_set & predicted_gene_names
        union = hallmark_set | predicted_gene_names
        post_metrics[f'hallmark_precision_{hallmark_name}'] = float(len(intersection) / len(predicted_gene_names)) if predicted_gene_names else 0.0
        post_metrics[f'hallmark_recall_{hallmark_name}'] = float(len(intersection) / len(hallmark_set)) if hallmark_set else 0.0
        post_metrics[f'hallmark_iou_{hallmark_name}'] = float(len(intersection) / len(union)) if union else 0.0
        logger.info(f"  {hallmark_name}: precision={post_metrics[f'hallmark_precision_{hallmark_name}']:.4f}  recall={post_metrics[f'hallmark_recall_{hallmark_name}']:.4f}  IoU={post_metrics[f'hallmark_iou_{hallmark_name}']:.4f}")

    all_intersection = all_hallmarks & predicted_gene_names
    all_union = all_hallmarks | predicted_gene_names
    post_metrics['hallmark_all_iou'] = float(len(all_intersection) / len(all_union)) if all_union else 0.0
    logger.info(f"  All hallmarks combined IoU = {post_metrics['hallmark_all_iou']:.4f}")
else:
    logger.warning(f"  senescence_marker_list.csv not found at {marker_path}, skipping hallmark enrichment.")

# ------------------------------------------------------------------
# 2b. Phenotype DEG overlap (IoU of predicted SnGs vs. phenotype DEGs)
# ------------------------------------------------------------------
deg_dir = os.path.join(args.output_dir, 'Senescent_Tables')
if os.path.exists(deg_dir):
    logger.info("Computing phenotype DEG overlap (IoU) ...")
    import glob
    deg_files = glob.glob(os.path.join(deg_dir, '*_DEG_results.csv'))
    if deg_files and 'predicted_gene_names' in dir():
        all_deg_genes = set()
        for f in deg_files:
            try:
                deg_df = pd.read_csv(f)
                sig_degs = deg_df[deg_df['p_val_adj'] < 0.05]['gene'].tolist()
                all_deg_genes.update(sig_degs)
            except Exception:
                pass
        if all_deg_genes:
            deg_intersection = predicted_gene_names & all_deg_genes
            deg_union = predicted_gene_names | all_deg_genes
            post_metrics['deg_overlap_iou'] = float(len(deg_intersection) / len(deg_union))
            post_metrics['deg_overlap_count'] = len(deg_intersection)
            logger.info(f"  SnG–DEG IoU = {post_metrics['deg_overlap_iou']:.4f}  ({post_metrics['deg_overlap_count']} overlapping genes)")

# ------------------------------------------------------------------
# 3. Phenotype separation: AUROC + Mann-Whitney U
# ------------------------------------------------------------------
logger.info("Computing phenotype separation metrics ...")
pheno_scores_final = generate_pheno_specific_scores(
    sen_gene_ls, gene_cell, edge_index_selfloop, attention_scores, graph_nx)
phenotype_conditions = list(pheno_scores_final.keys())
post_metrics['phenotype_conditions'] = phenotype_conditions

if len(phenotype_conditions) == 2:
    scores_a = [v[0] for v in pheno_scores_final[phenotype_conditions[0]]]
    scores_b = [v[0] for v in pheno_scores_final[phenotype_conditions[1]]]
    if scores_a and scores_b:
        mw_stat, mw_p = mannwhitneyu(scores_a, scores_b, alternative='two-sided')
        all_scores = scores_a + scores_b
        labels_pheno = [1] * len(scores_a) + [0] * len(scores_b)
        auroc = roc_auc_score(labels_pheno, all_scores)
        post_metrics['phenotype_mannwhitney_stat'] = float(mw_stat)
        post_metrics['phenotype_mannwhitney_p'] = float(mw_p)
        post_metrics['phenotype_auroc'] = float(auroc)
        post_metrics['phenotype_mean_score'] = {
            phenotype_conditions[0]: float(np.mean(scores_a)),
            phenotype_conditions[1]: float(np.mean(scores_b)),
        }
        logger.info(f"  Phenotype AUROC = {auroc:.4f}  MW p = {mw_p:.4e}")
elif len(phenotype_conditions) > 2:
    logger.info(f"  >2 phenotypes detected ({phenotype_conditions}); reporting per-condition mean SnC score.")
    post_metrics['phenotype_mean_score'] = {
        cond: float(np.mean([v[0] for v in pheno_scores_final[cond]])) for cond in phenotype_conditions
    }

# ------------------------------------------------------------------
# 4. SnC silhouette score (SnC vs. non-SnC in embedding space)
# ------------------------------------------------------------------
logger.info("Computing SnC silhouette score ...")
snc_keys = list(sencell_dict.keys())
nonsnc_keys = list(nonsencell_dict.keys())
if snc_keys and nonsnc_keys:
    snc_embs = torch.stack([sencell_dict[k][2].detach().cpu() for k in snc_keys])
    nonsnc_embs = torch.stack([nonsencell_dict[k][2].detach().cpu() for k in nonsnc_keys])
    all_embs = torch.cat([snc_embs, nonsnc_embs], dim=0).numpy()
    sil_labels = [1] * len(snc_keys) + [0] * len(nonsnc_keys)
    sil_score = silhouette_score(all_embs, sil_labels)
    post_metrics['snc_silhouette_score'] = float(sil_score)
    logger.info(f"  SnC silhouette score = {sil_score:.4f}")

# ------------------------------------------------------------------
# 5. GAE graph reconstruction BCE loss (final)
# ------------------------------------------------------------------
logger.info("Computing final GAE reconstruction loss ...")
model.eval()
with torch.no_grad():
    z_final = model.encode(data.x, data.edge_index)
    gae_recon_loss = model.recon_loss(z_final, data.edge_index)
post_metrics['gae_final_recon_loss'] = float(gae_recon_loss.item())
logger.info(f"  Final GAE BCE recon loss = {post_metrics['gae_final_recon_loss']:.6f}")

if args.retrain:
    post_metrics['gat_loss_trajectory'] = gat_losses
    post_metrics['gae_initial_recon_loss'] = gat_losses[0] if gat_losses else None
    post_metrics['gae_loss_reduction'] = float(gat_losses[0] - gat_losses[-1]) if len(gat_losses) > 1 else None

# ------------------------------------------------------------------
# 6. Convergence: per-epoch log summary
# ------------------------------------------------------------------
post_metrics['epoch_metrics'] = epoch_metrics_log
final_snc_cover = epoch_metrics_log[-1]['snc_cover_ratio'] if epoch_metrics_log else None
final_sng_cover = epoch_metrics_log[-1]['sng_cover_ratio'] if epoch_metrics_log else None
post_metrics['final_snc_cover_ratio'] = final_snc_cover
post_metrics['final_sng_cover_ratio'] = final_sng_cover
logger.info(f"  Final SnC cover ratio = {final_snc_cover}  |  Final SnG cover ratio = {final_sng_cover}")

# ------------------------------------------------------------------
# Save all metrics to JSON
# ------------------------------------------------------------------
metrics_path = os.path.join(args.output_dir, f'{args.exp_name}_eval_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(post_metrics, f, indent=2, default=str)
logger.info(f"All evaluation metrics saved to: {metrics_path}")
logger.info("====== Evaluation complete ======")

