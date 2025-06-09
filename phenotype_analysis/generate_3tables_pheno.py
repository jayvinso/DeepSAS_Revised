import os
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch_scatter

import scanpy as sp
import utils


args = utils.parse_args()
print(vars(args))

args.output_dir=f"./outputs/{args.exp_name}"
print("Outputs dir:",args.output_dir)

os.makedirs(args.output_dir, exist_ok=True)

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
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_example_data() 
else:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data1(args.input_data_count)


    
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
    


new_data, markers_index,\
    raw_sen_gene_ls, nonsen_gene_ls, gene_names = utils.process_data(
        adata, cluster_cell_ls, cell_cluster_arr,args)
    
# raw sene gene idx: gene name
initial_marker = pd.DataFrame({'gene_idx': raw_sen_gene_ls, 'gene_name': list(new_data.var_names[raw_sen_gene_ls])})


file_path=os.path.join(args.output_dir, f'{args.exp_name}_sencellgene-epoch4.data')
output_path=os.path.join(args.output_dir, 'Senescent_Tables')

os.makedirs(output_path, exist_ok=True)

# Load the saved object
loaded_data = torch.load(file_path)

# Unpack the loaded data, sen_gene_ls is new sene gene list from DeepSAS
sencell_dict, sen_gene_ls, attention_scores, edge_index_selfloop = loaded_data

print(f"Number of SnCs: {len(sencell_dict.keys())}")
print(f"Number of SnGs: {len(sen_gene_ls)}")



base_filtered_gene_names = []
for gene_idx in sen_gene_ls:
    gene_idx = gene_idx
    print(int(gene_idx))
    base_filtered_gene_names.append(new_data.var.index[int(gene_idx)])
base_gene_idx2names = {'gene_idx': sen_gene_ls, 'gene_name': base_filtered_gene_names}
base_gene_idx2names_df = pd.DataFrame(base_gene_idx2names)
base_predict_sens_list = list(set(base_gene_idx2names_df['gene_name']).difference(set(initial_marker['gene_name'])))
print(base_predict_sens_list)
print(len(base_predict_sens_list))


# Filter only valid cell indices (i >= gene_count)
cell_offset = new_data.shape[1]
sencell_indexs = [i for i in sencell_dict.keys() if i >= cell_offset]

# Check bounds to avoid IndexError
assert max(sencell_indexs) < (cell_offset + new_data.shape[0]), \
    "Index exceeds total number of graph nodes — check sencell_dict"

# Extract corresponding clusters
sencell_cluster = []
for i in sencell_indexs:
    ct = new_data.obs.iloc[i - cell_offset].clusters
    sencell_cluster.append(ct)

# Store in dataframe
sencell_df = pd.DataFrame({'sencell_index': sencell_indexs, 'sencell_cluster': sencell_cluster})


gene_cell=new_data.X.T

def AttentionEachCell(gene_cell, sen_gene_ls, edge_index_selfloop):
    """
    Calculate SnCs score based on senescent gene list
    Same as in the main4.py
    """
    # Convert the list of senescent genes to a tensor index
    # gene_index = torch.tensor(sen_gene_ls, dtype=torch.long)
    gene_index = sen_gene_ls
    
    # Create a boolean mask for senescent genes
    total_nodes = gene_cell.shape[0] + gene_cell.shape[1]
    gene_mask = torch.zeros(total_nodes, dtype=torch.bool)
    gene_mask[gene_index] = True

    # Identify edges where the target is a cell node
    cell_offset = gene_cell.shape[0]
    edge_mask_cell = edge_index_selfloop[1] >= cell_offset
    
    # Identify edges where the source is a senescent gene
    edge_mask_gene = gene_mask[edge_index_selfloop[0]]
    
    # Combined mask for edges from senescent genes to cells
    edge_mask_selected = edge_mask_cell & edge_mask_gene
    
    # Indices of selected edges
    edges_selected_indices = edge_mask_selected.nonzero().squeeze()

    # Target cell indices and attention scores for selected edges
    selected_edges_targets = edge_index_selfloop[1][edges_selected_indices]
    selected_attention_scores = attention_scores[edges_selected_indices].squeeze()
    
    # Adjust cell indices to start from 0
    cell_indices_in_range = selected_edges_targets - cell_offset
    
    # Number of cells
    num_cells = gene_cell.shape[1]
    
    # Sum and count attention scores per cell
    attention_sums = torch_scatter.scatter(selected_attention_scores, cell_indices_in_range,
                                           dim_size=num_cells, reduce='sum')
    attention_counts = torch_scatter.scatter(torch.ones_like(selected_attention_scores),
                                             cell_indices_in_range, dim_size=num_cells, reduce='sum')
    
    # Compute mean attention score per cell, handle division by zero
    attention_s_per_cell = torch.zeros(num_cells)
    valid_cells = attention_counts > 0
    # NOTE: Mean operation for each attention edge
    attention_s_per_cell[valid_cells] = attention_sums[valid_cells] / attention_counts[valid_cells]

    return attention_s_per_cell.detach().tolist()

attention_s_per_cell = AttentionEachCell(gene_cell, sen_gene_ls, edge_index_selfloop)

if_senCs = np.zeros(new_data.shape[0],dtype=np.int64)
if_senCs[[i - new_data.shape[1] for i in sencell_indexs]] = 1

SnC_scores = pd.DataFrame({'cell_id':list(range(new_data.shape[0])), 'cell_name': new_data.obs_names, 'cell_type':new_data.obs['clusters'], 'ifSnCs': list(if_senCs), 'SnC scores': attention_s_per_cell})


SnC_scores.to_csv(f"{output_path}/Cell_Table1_SnC_scores.csv", index=False)
new_data.obs = pd.merge(new_data.obs, SnC_scores, left_index=True, right_index=True)
new_data.obs['ifSnCs'] = new_data.obs['ifSnCs'].astype(str)

def DEGTable(new_data):
    # prepare adata for deg
    adata_deg=new_data.copy()
    sp.pp.normalize_total(adata_deg, target_sum=1e4)
    sp.pp.log1p(adata_deg)
    # sp.pp.scale(adata_deg)

    cell_types = adata_deg.obs['clusters'].unique()
    for cell_type in cell_types:
        adata_deg_sub=adata_deg[adata_deg.obs['clusters']==cell_type].copy()
        value_counts = (adata_deg_sub.obs['ifSnCs'] == '1').sum()
        if value_counts <= 5:
            continue

        print(f"{cell_type}, Number of SnCs: {value_counts}")    
        sp.tl.rank_genes_groups(adata_deg_sub, groupby='ifSnCs', groups=["1"], 
                                reference="0", method='wilcoxon')
        # Extract the results into a DataFrame
        degs = pd.DataFrame({
            'gene': adata_deg_sub.uns['rank_genes_groups']['names']['1'],
            'p_val': adata_deg_sub.uns['rank_genes_groups']['pvals']['1'],
            'logFC': adata_deg_sub.uns['rank_genes_groups']['logfoldchanges']['1'],
            'p_val_adj': adata_deg_sub.uns['rank_genes_groups']['pvals_adj']['1']
        })
        
        degs=degs.sort_values(by='logFC')

        save_ct_name = cell_type.replace(' ', '_')
        save_ct_name = cell_type.replace('/', '_')
        print(save_ct_name)
        degs.to_csv(f"{output_path}/{save_ct_name}_DEG_results.csv", index=False)

DEGTable(new_data)

cluster_count = pd.DataFrame(new_data.obs['clusters'].value_counts())
sencell_cluster_count = pd.DataFrame(sencell_df['sencell_cluster'].value_counts())
sencell_cluster_count=sencell_cluster_count.reset_index()
sencell_cluster_count = sencell_cluster_count['sencell_cluster']
merged_df = cluster_count.join(sencell_cluster_count).fillna(0)
merged_df = merged_df.rename(columns={'clusters': 'number_of_cells', 'sencell_cluster': "number_of_SnCs"})
merged_df = merged_df.astype(int)
merged_df.index.name = "cell_type"

merged_df.to_csv(f"{output_path}/Cell_Table2_SnCs_per_ct.csv")

ct_sencell_indexs={}
row_numbers=np.array(sencell_indexs)-new_data.shape[1]

# Dict {ct: cell_index}
for i in row_numbers:
    ct_=new_data.obs.iloc[i]['clusters']
    if ct_ in ct_sencell_indexs:
        ct_sencell_indexs[ct_].append(i+new_data.shape[1])
    else:
        ct_sencell_indexs[ct_]=[i+new_data.shape[1]]
        

def AttentionEachGene(gene_cell, cell_indices, edge_index_selfloop):
    """
    gene attention score from edges connected with cell_indices,
    the same as main4.py, but separate for each cell type.
    """
    num_genes = gene_cell.shape[0]
    num_cells = gene_cell.shape[1]
    total_nodes = num_genes + num_cells

    # Create a mask for the selected cells
    cell_mask = torch.zeros(total_nodes, dtype=torch.bool)
    cell_mask[cell_indices] = True

    # Create a mask for edges where the target node is a gene
    edge_mask_gene = edge_index_selfloop[1] < num_genes
    
    edge_mask_cell = cell_mask[edge_index_selfloop[0]]

    edge_mask_selected = edge_mask_gene & edge_mask_cell

    edges_selected_indices = edge_mask_selected.nonzero().squeeze()

    # Target cell indices and attention scores for selected edges
    selected_edges_targets = edge_index_selfloop[1][edges_selected_indices]
    selected_attention_scores = attention_scores[edges_selected_indices].squeeze()

    # Compute per-gene sums and counts using torch_scatter
    # Counts: Number of times each gene appears in masked_genes
    counts = torch_scatter.scatter(torch.ones_like(selected_attention_scores), selected_edges_targets,
                     dim=0, dim_size=num_genes, reduce='sum')
    # Sums: Sum of attention scores per gene
    sums = torch_scatter.scatter(selected_attention_scores, selected_edges_targets,
                   dim=0, dim_size=num_genes, reduce='sum')
      
    # Avoid division by zero
    res = torch.zeros(num_genes, dtype=torch.float32)
    nonzero_mask = counts > 0
    res[nonzero_mask] = sums[nonzero_mask] / counts[nonzero_mask] # Mean

    return res.detach()

ct2gene_score = {}
# SnGs for each cell type
for ct, cell_indices in ct_sencell_indexs.items():
    score_per_gene = AttentionEachGene(gene_cell, cell_indices, edge_index_selfloop)
    sene_gene_score = score_per_gene[sen_gene_ls].tolist()
    ct2gene_score[ct] = sene_gene_score
    
    
ct2gene_score_df = pd.DataFrame(ct2gene_score)

# align gene index with gene names
var_df = new_data.var.reset_index()

sene_gene_names = []

for gene_idx in sen_gene_ls:
    gene_idx = gene_idx
    gene_name = var_df.iloc[int(gene_idx)]['index']
    sene_gene_names.append(gene_name)
ct2gene_score_df.index = sene_gene_names


ct2gene_score_df.to_csv(f"{output_path}/Gene_Table1_SnG_scores_per_ct.csv")

def GeneTable2(new_data, ct2gene_score_df):
    cell_types = new_data.obs['clusters'].unique()

    df_list = []
    for ct in cell_types:
        save_ct_name = ct.replace(' ', '_')
        save_ct_name = ct.replace('/', '_')
        degs_ct_path = f"{output_path}/{save_ct_name}_DEG_results.csv"
        if os.path.exists(degs_ct_path):
            degs_ct_df = pd.read_csv(degs_ct_path)
            sub_degs_ct_df = degs_ct_df[degs_ct_df['gene'].isin(ct2gene_score_df.index)]
            sub_degs_ct_df = sub_degs_ct_df[sub_degs_ct_df['logFC'] >= 0.25] # fliter DEG genes based on logFC
            sub_degs_ct_df['cell_type'] = ct
            sub_degs_ct_df[f'SnG_score'] = sub_degs_ct_df['gene'].map(ct2gene_score_df[ct].to_dict()) # SnG scores fo this cell_type
            df_list.append(sub_degs_ct_df)

    all_df = pd.concat(df_list)
    all_df = all_df[all_df["SnG_score"] != 0.]
    return all_df
total_df = GeneTable2(new_data, ct2gene_score_df)
total_df

total_df.to_csv(f"{output_path}/Gene_Table2_DEG_ct_SnG_score.csv", index=False)

# Group by 'gene' and aggregate cell_type into a list or join them
grouped_df = total_df.groupby('gene').agg({
    'cell_type': lambda x: ', '.join(x.unique()),  # Combine unique cell_types
    'p_val': 'mean',   # Example of how you can aggregate other columns
    'logFC': 'mean',
    'p_val_adj': 'mean',
    'SnG_score': 'mean'
}).reset_index()

grouped_df['hallmarker'] = grouped_df['gene'].isin(initial_marker['gene_name'])

grouped_df.to_csv(f"{output_path}/Gene_Table3_gene_ct_count.csv", index=False)

# Filter rows where "cell_type" column contains only one cell type
df_filtered = grouped_df[grouped_df["cell_type"].apply(lambda x: len(x.split(",")) == 1)]

df_filtered.to_csv(f"{output_path}/Gene_newTable3_gene_ct_count.csv", index=False)


 # --------------------------
# PHENOTYPE SECTION : SnG Score per CellType × Phenotype
# --------------------------
pheno_output = os.path.join(args.output_dir, 'Phenotype_SnG_Results')
os.makedirs(pheno_output, exist_ok=True)
logger.info("Generating Gene_Table1_SnG_scores_per_ct_pheno.csv")

ct_pheno_scores = []
cell_offset = gene_cell.shape[0]
celltypes = new_data.obs['clusters'].unique()
phenotypes = new_data.obs['Condition'].unique()

for ct in celltypes:
    for pheno in phenotypes:
        mask = (new_data.obs['clusters'] == ct) & (new_data.obs['Condition'] == pheno)
        if mask.sum() == 0:
            continue
        cell_ids = mask[mask].index
        model_cell_indices = [cell_offset + new_data.obs.index.get_loc(cid) for cid in cell_ids]
        scores = AttentionEachGene(gene_cell, model_cell_indices, edge_index_selfloop)[sen_gene_ls].tolist()
        for gene, score in zip(sene_gene_names, scores):
            ct_pheno_scores.append({"gene": gene, "cell_type": ct, "phenotype": pheno, "score": score})

ct_pheno_df = pd.DataFrame(ct_pheno_scores)

# Save long-form table
ct_pheno_df.to_csv(os.path.join(pheno_output, 'Gene_Table1_SnG_scores_per_ct_pheno.csv'), index=False)

# Generate pivot table (wide-format for easier viewing)
ct_pheno_pivot = ct_pheno_df.pivot_table(index='gene', columns=['cell_type', 'phenotype'], values='score')
ct_pheno_pivot.to_csv(os.path.join(pheno_output, 'Gene_Table1_SnG_scores_per_ct_pheno_pivot.csv'))

# Generate Z-score normalized pivot table
ct_pheno_pivot_zscore = ct_pheno_pivot.apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=0)
ct_pheno_pivot_zscore.to_csv(os.path.join(pheno_output, 'Gene_Table1_SnG_scores_per_ct_pheno_pivot_zscore.csv'))

print("Raw and Z-score normalized SnG score tables written.")


# Extract all genes with z-score >= 2.0 per cell_type × phenotype
Z_THRESH = 2.0
specific_gene_rows = []

for col in ct_pheno_pivot_zscore.columns:
    col_z = ct_pheno_pivot_zscore[col].dropna()
    selected_genes = col_z[col_z >= Z_THRESH]
    for gene, z in selected_genes.items():
        if isinstance(col, tuple):
            cell_type, phenotype = col
        else:
            cell_type, phenotype = col.split('|') if '|' in col else ('Unknown', col)
        specific_gene_rows.append({
            'gene': gene,
            'cell_type': cell_type,
            'phenotype': phenotype,
            'z_score': z
        })

specific_gene_df = pd.DataFrame(specific_gene_rows)
specific_gene_df.to_csv(os.path.join(pheno_output, 'Gene_Specific_SnGs_per_ct_pheno_filtered.csv'), index=False)
print("Specific SnGs (z ≥ 2.0) extracted and saved.")

# --------------------------
# PHENOTYPE SECTION: SnC per CellType × Phenotype
# --------------------------
logger.info("Generating Cell_Table2_SnCs_per_ct_pheno.csv")

ct_pheno_counts = (
    new_data.obs
    .groupby(['clusters', 'Condition'])
    .agg(number_of_cells=('ifSnCs', 'count'),
         number_of_SnCs=('ifSnCs', lambda x: (x == '1').sum()))
    .reset_index()
    .rename(columns={'clusters': 'cell_type', 'Condition': 'phenotype'})
)

ct_pheno_counts.to_csv(os.path.join(pheno_output, 'Cell_Table2_SnCs_per_ct_pheno.csv'), index=False)
print("Cell-type × phenotype SnC summary written.")