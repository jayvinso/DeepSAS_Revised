# DeepSAS

DeepSAS (**Deep**-learning framework for cell-type-specific **S**nCs **A**nd **S**nGs) is a computational framework designed to identify senescent cells and senescence-associated genes from single-cell RNA sequencing data.

## Overview

Cellular senescence is a state of permanent cell cycle arrest that plays important roles in development, tissue homeostasis, aging, and disease. Identifying senescent cells in heterogeneous tissues is challenging due to the lack of universal markers. DeepSAS leverages graph neural networks and contrastive learning to identify senescent cells and their associated gene signatures from single-cell RNA sequencing data.

The framework integrates several key components:
1. Graph representation of cell-gene interactions
2. Graph Attention Networks (GAT) for capturing complex relationships
3. Contrastive learning with multi-level distance optimization
4. Attention mechanisms for identifying senescence-associated genes

## Features

- Identifies senescent cells in heterogeneous tissues with cell-type specificity
- Discovers senescence-associated genes specific to each cell type
- Leverages both gene expression patterns and cell-cell interactions
- Robust to batch effects and technical variations through built-in normalization
- Works across different cell types and senescence induction methods
- Scales to large datasets with optimized sampling strategies
- Provides comprehensive visualization and analysis tools

## Installation

This project is developed and tested on Linux and macOS environments.


1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chthub/deepsas.git
   cd sencell/
   git checkout deepsas-v1
   ```

2. **Set Up a uv Environment** (recommended):
   We recommend to use uv for the environment mangement. Check this [link](https://docs.astral.sh/uv/) to install uv.

   ```bash
   uv venv --python 3.8.20
   source .venv/bin/activate
   ```

4. **Install Dependencies**:
   
   ```bash
   uv pip install numpy seaborn matplotlib pandas tabulate linetimer scikit-learn ipykernel 'scanpy[leiden]' tqdm gseapy 
   ```
   For [Pytorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html),  ensure you select the CUDA version that best suits your system. Below is an example from our test environment:
   ```bash
   uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
   uv pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 
   ```

## Quick Start

### Basic Usage

To run DeepSAS on the provided example data:

```bash
uv run python -u deepsas_v1.py --exp_name example --device_index 0 --retrain > ./example.log
```

To run in background with logging (recommended for longer analyses):

```bash
nohup uv run python -u deepsas_v1.py --exp_name your_experiment --device_index 0 --retrain > ./your_experiment.log 2>&1 &
```

### Analysis Workflow

DeepSAS follows a 4-step workflow:

1. **Data Loading & Preprocessing**: Filters genes/cells and constructs the cell-gene graph
2. **Initial Embedding Generation**: Creates embeddings via scanpy or autoencoder
3. **Graph Attention Network Training**: Learns the graph structure with GATConv layers
4. **Contrastive Learning**: Refines embeddings to identify senescent cells and genes

### Results Analysis

After running DeepSAS, generate tables of senescence-associated cells and genes:

```bash
uv run python -u generate_3tables.py --output_dir ./outputs --exp_name example --device_index 0
```

This generates several output tables:

1. **Cell-level analysis**: 
   - Cell senescence scores and binary classification (senescent/non-senescent)
   - Distribution of senescent cells across cell types

2. **Gene-level analysis**: 
   - Differentially expressed genes between senescent and non-senescent cells
   - Cell-type specific senescence-associated genes with statistical measures
   - Aggregated gene information across multiple cell types

3. **Statistical measures**:
   - p-values and adjusted p-values from Wilcoxon rank-sum tests
   - Log fold-changes showing expression differences
   - Senescence-associated gene (SnG) scores derived from attention weights

For visualization and downstream analysis, follow the tutorial in [`tutorial.ipynb`](./tutorial.ipynb), which demonstrates:
- UMAP visualization of senescent cells
- Gene set enrichment analysis of identified senescence markers
- Cell-type specific senescence marker analysis and interpretation

### Large Dataset Analysis

For datasets with many cells (>50,000), use the sampling approach detailed in [Sampling_Tutorial.md](./Sampling_Tutorial.md), which provides:
- Subsampling strategies
- Batch processing scripts
- Result integration methods


## Input Data Format

DeepSAS works with h5ad format (AnnData objects from Scanpy). The input data should include:
- Gene expression matrix (cells × genes) in sparse or dense format
- Cell type annotations in `adata.obs['clusters']` or another specified column
- (Optional) Additional metadata like batch information for batch effect correction

The input expression data should be normalized counts (not log-transformed). DeepSAS will handle normalization, scaling, and batch correction internally using scanpy functions.

## Parameters

DeepSAS accepts the following parameters:

### Input/Output Parameters
- `--input_data_count`: Path to input data (h5ad format)
- `--output_dir`: Base output directory
- `--exp_name`: Experiment name (used for output directory naming)
- `--device_index`: CUDA device index to use
- `--retrain`: Whether to retrain models or use saved ones
- `--timestamp`: Timestamp for the experiment (optional)

### Model Configuration
- `--seed`: Random seed for reproducibility
- `--n_genes`: Number of genes to use (3000, 8000 or full)
- `--ccc`: Cell-cell edge type: type1 (binary), type2 (continuous), type3 (none)
- `--gene_set`: Gene set to use (senmayo, fridman, etc.)
- `--emb_size`: Embedding dimension size

### Training Parameters
- `--gat_epoch`: Number of epochs to train the GAT model
- `--sencell_num`: Number of senescent cells to use
- `--sengene_num`: Number of senescence-associated genes to use
- `--sencell_epoch`: Number of epochs to train the Sencell model
- `--cell_optim_epoch`: Number of epochs for cell embedding optimization
- `--learning_rate`: Initial learning rate
- `--batch_id`: Batch ID for processing

## Output Files

DeepSAS generates the following output files in the specified output directory:

- `{exp_name}_new_data.h5ad`: Processed AnnData object with filtered genes/cells
- `{exp_name}_graphnx.data`: NetworkX graph representation of cell-gene interactions
- `{exp_name}_graphpyg.data`: PyTorch Geometric graph representation for GAT model
- `{exp_name}_GAT.pt`: Trained Graph Attention Network model
- `{exp_name}_sencellgene-epoch{epoch}.data`: Identified senescent cells and genes at each training epoch

The final epoch output contains:
- List of identified senescent cells with attention scores
- List of senescence-associated genes ranked by importance
- Attention matrix capturing gene-cell relationships
- Edge index information for the graph structure

After running `generate_3tables.py`, you'll also get a folder `Senescent_Tables` in the output path. In this folder you have:

1. **Cell_Table1_SnC_scores.csv**: Information about each cell and its senescence score
   - Contains cell IDs, names, types, binary senescent indicator, and senescence scores

2. **Gene_Table2_DEG_ct_SnG_score.csv**: Differentially expressed genes between senescent and non-senescent cells
   - Includes gene names, cell types, p-values, log fold-changes, adjusted p-values, senescence scores, and hallmark status

3. **Gene_newTable3_gene_ct_count.csv**: Summary of genes across multiple cell types
   - Aggregates information by gene, showing in how many cell types each gene is differentially expressed

4. **Additional tables**:
   - **Cell_Table2_SnCs_per_ct.csv**: Counts of total cells and senescent cells per cell type
   - **table2ByCelltype.csv**: Table 2 grouped by cell type
   - **table2ByGene.csv**: Table 2 grouped by gene

For detailed explanations of each table and column, see [Senescent_Tables_Explanation.md](./Senescent_Tables_Explanation.md)


## Citation

If you use DeepSAS in your research, please cite:

```
@article{
   xxxx
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
