import torch
import numpy as np
import pandas as pd


#START - Gate value extraction for phenotype interpretability
def extract_gate_values(gate_values, cell_to_sample, graph_nx, gene_num,
                        celltype_names=None, sample_ids=None):
    """Extract and organize gate values (g_i) per cell for interpretation.
    
    The gate value indicates how heavily each cell incorporates its
    global sample phenotype context. High gate values suggest the cell's
    representation is strongly influenced by sample-level metadata.
    
    Args:
        gate_values: (n_cells,) tensor of gate values from GatedPhenotypeInjection
        cell_to_sample: (n_cells,) sample assignment per cell
        graph_nx: networkx graph with node attributes
        gene_num: number of gene nodes (offset for cell indices)
        celltype_names: list of cell type names
        sample_ids: list of sample identifiers
        
    Returns:
        gate_df: DataFrame with columns [cell_index, cell_type, sample, gate_value]
        summary: dict with per-celltype and per-sample gate statistics
    """
    g_np = gate_values.detach().cpu().numpy()
    c2s = cell_to_sample.detach().cpu().numpy() if isinstance(
        cell_to_sample, torch.Tensor) else cell_to_sample

    records = []
    for i in range(len(g_np)):
        cell_idx = i + gene_num
        ct_idx = graph_nx.nodes[int(cell_idx)].get('cluster', -1)
        ct_name = celltype_names[ct_idx] if celltype_names and ct_idx >= 0 else str(ct_idx)
        s_name = sample_ids[c2s[i]] if sample_ids else str(c2s[i])

        records.append({
            'cell_index': cell_idx,
            'cell_type': ct_name,
            'sample': s_name,
            'gate_value': float(g_np[i]),
        })

    gate_df = pd.DataFrame(records)

    summary = {}
    if len(gate_df) > 0:
        summary['per_celltype'] = gate_df.groupby('cell_type')['gate_value'].agg(
            ['mean', 'std', 'min', 'max']).to_dict('index')
        summary['per_sample'] = gate_df.groupby('sample')['gate_value'].agg(
            ['mean', 'std', 'min', 'max']).to_dict('index')
        summary['global_mean'] = float(gate_df['gate_value'].mean())
        summary['global_std'] = float(gate_df['gate_value'].std())

    return gate_df, summary


def print_gate_summary(summary, top_k=5):
    """Print a human-readable summary of gate value statistics."""
    print(f"\nGate Value Summary:")
    print(f"  Global: mean={summary.get('global_mean', 0):.4f}, "
          f"std={summary.get('global_std', 0):.4f}")

    if 'per_celltype' in summary:
        print(f"\n  Per Cell Type:")
        for ct, stats in sorted(summary['per_celltype'].items(),
                                 key=lambda x: x[1]['mean'], reverse=True)[:top_k]:
            print(f"    {ct}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    if 'per_sample' in summary:
        print(f"\n  Per Sample:")
        for s, stats in sorted(summary['per_sample'].items(),
                                key=lambda x: x[1]['mean'], reverse=True)[:top_k]:
            print(f"    {s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
#END
