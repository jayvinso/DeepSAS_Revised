This version introduces several major enhancements:

1. Phenotype-aware senescence prediction
   
The pipeline now incorporates the Condition metadata field from 'adata.obs' to stratify SnC prediction by phenotype (e.g., treatment, disease state).

During graph construction, each cell node is annotated with its phenotype.

Results include per-phenotype SnC and SnG statistics, as well as z-scored SnG signatures per (cell type × phenotype) combination.

3. Cell-type–specific HVG filtering
   
Instead of applying a all genes, DeepSAS-phenotype applies independent HVG selection for each cell type through using --use_hvg_deg

This ensures biologically relevant genes are preserved for identifying senescence programs specific to each cell lineage.

5. Enhanced Output Tables
   
The output now includes:

Gene_Table1_SnG_scores_per_ct_pheno.csv: raw SnG scores for each gene across (cell type × phenotype)

Gene_Table1_SnG_scores_per_ct_pheno_pivot[_zscore].csv: pivot tables (z-scored and raw)

Gene_Specific_SnGs_per_ct_pheno_filtered.csv: filtered SnGs with z-score ≥ 2.0 for each (cell type × phenotype)

Cell_Table2_SnCs_per_ct_pheno.csv: SnC and total cell counts per (cell type × phenotype)

All outputs are saved in a Phenotype_SnG_Results directory.
