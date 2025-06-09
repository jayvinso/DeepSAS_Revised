import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

import utils
from model_AE import reduction_AE
from model_GAT import GAEModel
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
    cellmodel, sencell_dict, nonsencell_dict = cell_optim(cellmodel, optimizer,
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

    torch.save([sencell_dict, sen_gene_ls, attention_scores, edge_index_selfloop],
               os.path.join(args.output_dir, f'{args.exp_name}_sencellgene-epoch{epoch}.data'))
