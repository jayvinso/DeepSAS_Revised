In this tutorial, we provide the guidance for phenotype-related analysis. The mian program with the phenotype enhenced is `deepsas_v1_phenotype_v2.py`.

### 1. Phenotype-aware senescence prediction
   
The pipeline now incorporates the _Condition_ metadata field from _adata.obs_ to stratify SnC prediction by phenotype (e.g., treatment, disease state).

During graph construction, each cell node is annotated with its phenotype.

Results include per-phenotype SnC and SnG statistics, as well as z-scored SnG signatures per (cell type × phenotype) combination.

### 2. Cell-type–specific HVG filtering
   
Instead of applying a all genes, DeepSAS-phenotype applies independent HVG selection for each cell type through using _--use_hvg_deg_

This ensures biologically relevant genes are preserved for identifying senescence programs specific to each cell lineage.

### 3. Enhanced Output Tables
   
After running the `generate_3tables_pheno.py`, the output now includes:

`Gene_Table1_SnG_scores_per_ct_pheno.csv`: raw SnG scores for each gene across (cell type × phenotype)

`Gene_Table1_SnG_scores_per_ct_pheno_pivot[_zscore].csv`: pivot tables (z-scored and raw)

`Gene_Specific_SnGs_per_ct_pheno_filtered.csv`: filtered SnGs with z-score ≥ 2.0 for each (cell type × phenotype)

`Cell_Table2_SnCs_per_ct_pheno.csv`: SnC and total cell counts per (cell type × phenotype)

All outputs are saved in a `Phenotype_SnG_Results` directory.
