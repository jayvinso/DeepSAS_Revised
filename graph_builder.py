import numpy as np
import torch
from torch_geometric.data import Data as Graphdata
from torch_geometric.utils import to_undirected


#START - Phenotype node construction and cell-to-phenotype edge creation
def build_phenotype_features(adata, sample_col='Sample', disease_col='Condition',
                             lobe_col=None, age_col=None, batch_col=None):
    """Build phenotype feature vectors for each unique sample.
    
    Returns:
        sample_ids: list of unique sample identifiers
        phenotype_features: np.ndarray of shape (n_samples, feat_dim)
        cell_to_sample: np.ndarray mapping each cell index to its sample index
        age_col_idx: int or None, column index of age in phenotype_features
    """
    obs = adata.obs
    sample_ids = sorted(obs[sample_col].unique()) if sample_col in obs.columns else ['single_sample']
    sample_to_idx = {s: i for i, s in enumerate(sample_ids)}

    if sample_col in obs.columns:
        cell_to_sample = np.array([sample_to_idx[s] for s in obs[sample_col]])
    else:
        cell_to_sample = np.zeros(obs.shape[0], dtype=np.int64)

    feature_parts = []
    age_col_idx = None
    col_offset = 0

    def _get_unique_sample_value(obs_subset, col, sample_id):
        """Return the single unique value for a metadata column within a sample.
        Raises if the sample has conflicting values."""
        vals = obs_subset[col].unique()
        if len(vals) != 1:
            raise ValueError(
                f"Sample '{sample_id}' has {len(vals)} distinct values for "
                f"'{col}': {vals.tolist()}. Expected exactly one.")
        return vals[0]

    # Disease / condition one-hot
    if disease_col and disease_col in obs.columns:
        disease_categories = sorted(obs[disease_col].unique())
        disease_map = {d: i for i, d in enumerate(disease_categories)}
        disease_onehot = np.zeros((len(sample_ids), len(disease_categories)))
        for s_idx, s_id in enumerate(sample_ids):
            mask = obs[sample_col] == s_id if sample_col in obs.columns else np.ones(len(obs), dtype=bool)
            d = _get_unique_sample_value(obs.loc[mask], disease_col, s_id)
            disease_onehot[s_idx, disease_map[d]] = 1.0
        feature_parts.append(disease_onehot)
        col_offset += len(disease_categories)

    # Lobe one-hot
    if lobe_col and lobe_col in obs.columns:
        lobe_categories = sorted(obs[lobe_col].unique())
        lobe_map = {l: i for i, l in enumerate(lobe_categories)}
        lobe_onehot = np.zeros((len(sample_ids), len(lobe_categories)))
        for s_idx, s_id in enumerate(sample_ids):
            mask = obs[sample_col] == s_id if sample_col in obs.columns else np.ones(len(obs), dtype=bool)
            l = _get_unique_sample_value(obs.loc[mask], lobe_col, s_id)
            lobe_onehot[s_idx, lobe_map[l]] = 1.0
        feature_parts.append(lobe_onehot)
        col_offset += len(lobe_categories)

    # Age: z-scored continuous
    if age_col and age_col in obs.columns:
        sample_ages = np.zeros(len(sample_ids))
        for s_idx, s_id in enumerate(sample_ids):
            mask = obs[sample_col] == s_id if sample_col in obs.columns else np.ones(len(obs), dtype=bool)
            sample_ages[s_idx] = float(_get_unique_sample_value(obs.loc[mask], age_col, s_id))
        age_mean = sample_ages.mean()
        age_std = sample_ages.std() + 1e-8
        age_zscore = ((sample_ages - age_mean) / age_std).reshape(-1, 1)
        feature_parts.append(age_zscore)
        age_col_idx = col_offset
        col_offset += 1

    # Batch one-hot
    if batch_col and batch_col in obs.columns:
        batch_categories = sorted(obs[batch_col].unique())
        batch_map = {b: i for i, b in enumerate(batch_categories)}
        batch_onehot = np.zeros((len(sample_ids), len(batch_categories)))
        for s_idx, s_id in enumerate(sample_ids):
            mask = obs[sample_col] == s_id if sample_col in obs.columns else np.ones(len(obs), dtype=bool)
            b = _get_unique_sample_value(obs.loc[mask], batch_col, s_id)
            batch_onehot[s_idx, batch_map[b]] = 1.0
        feature_parts.append(batch_onehot)

    if len(feature_parts) == 0:
        phenotype_features = np.ones((len(sample_ids), 1))
    else:
        phenotype_features = np.concatenate(feature_parts, axis=1)

    assert phenotype_features.shape[0] == len(sample_ids), (
        f"Phenotype feature rows ({phenotype_features.shape[0]}) != "
        f"n_samples ({len(sample_ids)})")

    return sample_ids, phenotype_features, cell_to_sample, age_col_idx


def build_phenotype_edges(cell_to_sample, gene_num, cell_num, n_samples):
    """Build E_CP edges connecting each cell node to its sample-context phenotype node.
    
    Phenotype nodes are indexed after gene + cell nodes:
      phenotype_node_i = gene_num + cell_num + sample_idx
    
    Returns:
        edge_cp: torch.LongTensor of shape (2, n_cells) — cell-to-phenotype edges
    """
    assert len(cell_to_sample) == cell_num, (
        f"cell_to_sample length ({len(cell_to_sample)}) != "
        f"cell_num ({cell_num}). adata.obs and gene_cell are out of sync.")

    cell_indices = np.arange(cell_num) + gene_num
    pheno_indices = cell_to_sample + gene_num + cell_num

    edge_cp = torch.tensor(
        np.stack([cell_indices, pheno_indices], axis=0), dtype=torch.long
    )
    return edge_cp


def build_extended_graph_pyg(gene_cell, gene_embed, cell_embed,
                             phenotype_features, cell_to_sample,
                             edge_indexs, ccc_matrix=None):
    """Build PyG graph with cell, gene, AND phenotype-context nodes.
    
    Node ordering: [genes (0..G-1), cells (G..G+C-1), phenotype (G+C..G+C+P-1)]
    """
    gene_num = gene_cell.shape[0]
    cell_num = gene_cell.shape[1]
    n_samples = phenotype_features.shape[0]

    # y: legacy boolean mask (True=gene, False=other). Still used by
    # model_GAT.GAEModel to separate gene vs cell projections. For 3-way
    # node identification, use node_type instead.
    y = [True] * gene_num + [False] * cell_num + [False] * n_samples
    y = torch.tensor(y)

    # node_type: 0=gene, 1=cell, 2=phenotype
    node_type = torch.tensor(
        [0] * gene_num + [1] * cell_num + [2] * n_samples, dtype=torch.long
    )

    # Phenotype features need to be projected to emb_size in the encoder,
    # so pad them to match embedding dim for graph construction
    emb_dim = gene_embed.shape[1]
    pheno_tensor = torch.zeros(n_samples, emb_dim,
                               dtype=gene_embed.dtype, device=gene_embed.device)
    # Store raw phenotype features separately
    pheno_raw = torch.tensor(phenotype_features, dtype=torch.float32)

    x = torch.cat([gene_embed.detach(), cell_embed.detach(), pheno_tensor], dim=0)

    # Build cell-to-phenotype edges
    edge_cp = build_phenotype_edges(cell_to_sample, gene_num, cell_num, n_samples)

    # Combine original edges + phenotype edges
    if ccc_matrix is None:
        combined_edges = torch.cat([edge_indexs, edge_cp], dim=1)
        edge_index = to_undirected(combined_edges)
        graph_pyg = Graphdata(x=x, edge_index=edge_index, y=y)
    else:
        flatten_edge_features = ccc_matrix[ccc_matrix != 0]
        if len(flatten_edge_features) == 0:
            # No nonzero CCC edges — treat as if ccc_matrix is None
            combined_edges = torch.cat([edge_indexs, edge_cp], dim=1)
            edge_index = to_undirected(combined_edges)
            graph_pyg = Graphdata(x=x, edge_index=edge_index, y=y)
        else:
            min_val = np.min(flatten_edge_features)
            max_val = np.max(flatten_edge_features)
            denom = max(max_val - min_val, 1e-8)
            normalized_array = (flatten_edge_features - min_val) / denom

            # NOTE: This assumes the LAST len(normalized_array) edges in
            # edge_indexs are the CCC-derived edges, and all preceding
            # edges are non-CCC (weight=1.0). This ordering must be
            # maintained by the upstream build_graph_nx() function.
            n_non_ccc = edge_indexs.shape[1] - len(normalized_array)
            assert n_non_ccc >= 0, (
                f"More nonzero CCC entries ({len(normalized_array)}) than "
                f"edges in edge_indexs ({edge_indexs.shape[1]})")
            edge_attr_orig = np.concatenate([
                np.ones(n_non_ccc),
                normalized_array
            ])
            # Phenotype edges get weight 1.0
            edge_attr_pheno = np.ones(edge_cp.shape[1])
            edge_attr = np.concatenate([edge_attr_orig, edge_attr_pheno])

            combined_edges = torch.cat([edge_indexs, edge_cp], dim=1)
            undirected_edge_index, undirected_edge_attr = to_undirected(
                combined_edges, edge_attr=torch.tensor(edge_attr), reduce='mean'
            )
            graph_pyg = Graphdata(
                x=x, edge_index=undirected_edge_index,
                edge_attr=undirected_edge_attr, y=y
            )

    graph_pyg.node_type = node_type
    graph_pyg.pheno_raw = pheno_raw
    graph_pyg.n_genes = gene_num
    graph_pyg.n_cells = cell_num
    graph_pyg.n_phenotype = n_samples
    graph_pyg.cell_to_sample = torch.tensor(cell_to_sample, dtype=torch.long)

    print(f'Extended PyG graph: {graph_pyg}')
    print(f'  Genes: {gene_num}, Cells: {cell_num}, Phenotype nodes: {n_samples}')
    return graph_pyg
#END
