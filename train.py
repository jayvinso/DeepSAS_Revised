"""
Phenotype-Aware DeepSAS Training Module

Extends utils.parse_args with phenotype-specific hyperparameters and provides
the training loop integrating the heterogeneous graph with disentangled embeddings.

Usage:
    python train.py --exp_name example_pheno --retrain \
        --sample_col Sample --disease_col Condition \
        --lambda_ctx 0.1 --lambda_orth 0.01
"""
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

import utils
from model_AE import reduction_AE
from model_GAT import GAEModel
from model_Sencell import Sencell, cell_optim, update_cell_embeddings
from graph_builder import (
    build_phenotype_features,
    build_extended_graph_pyg,
)
from encoders import PhenotypeAwareEncoder
from losses import ContextObjective, MetadataPredictionFromCtx, orthogonality_loss, combined_loss
from evaluate import LeakageProbe
from interpretability import extract_gate_values, print_gate_summary

import logging
import os
import json
import random
import datetime
import argparse
import scanpy as sp


#START - Extended argument parsing for phenotype-aware training
def parse_phenotype_args():
    """Extends the base DeepSAS args with phenotype-specific hyperparameters."""
    parser = argparse.ArgumentParser(
        description='Phenotype-Aware DeepSAS training',
        parents=[],
    )

    # === Base DeepSAS arguments (mirrored from utils.parse_args) ===
    parser.add_argument('--input_data_count', type=str,
                        default="/bmbl_data/huchen/deepSAS_data/fixed_data_0525.h5ad")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--device_index', type=int, default=0)
    parser.add_argument('--retrain', action='store_true', default=False)
    parser.add_argument('--timestamp', type=str, default="")
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--n_genes', type=str, default='full')
    parser.add_argument('--ccc', type=str, default='type1')
    parser.add_argument('--gene_set', type=str, default='full')
    parser.add_argument('--emb_size', type=int, default=12)
    parser.add_argument('--gat_epoch', type=int, default=30)
    parser.add_argument('--sencell_num', type=int, default=600)
    parser.add_argument('--sengene_num', type=int, default=200)
    parser.add_argument('--sencell_epoch', type=int, default=40)
    parser.add_argument('--cell_optim_epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_id', type=int, default=0)

    # === Phenotype-specific arguments ===
    parser.add_argument('--sample_col', type=str, default='Sample',
                        help='Column in adata.obs for sample/donor identity')
    parser.add_argument('--disease_col', type=str, default='Condition',
                        help='Column in adata.obs for disease phenotype')
    parser.add_argument('--lobe_col', type=str, default=None,
                        help='Column in adata.obs for anatomical lobe (optional)')
    parser.add_argument('--age_col', type=str, default=None,
                        help='Column in adata.obs for donor age (continuous)')
    parser.add_argument('--batch_col', type=str, default=None,
                        help='Column in adata.obs for technical batch')

    # Loss weights
    parser.add_argument('--lambda_ctx', type=float, default=0.1,
                        help='Weight for phenotype prediction loss')
    parser.add_argument('--lambda_orth', type=float, default=0.01,
                        help='Weight for orthogonality regularization')
    parser.add_argument('--lambda_rec', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--reconstruct_ecp', action='store_true', default=False,
                        help='Ablation: include E_CP edges in reconstruction loss')

    # Architecture
    parser.add_argument('--sen_dim', type=int, default=None,
                        help='Dimension of z_sen branch (default: emb_size)')
    parser.add_argument('--ctx_dim', type=int, default=None,
                        help='Dimension of z_ctx branch (default: emb_size // 2)')
    parser.add_argument('--use_fourier_age', action='store_true', default=False,
                        help='Use Fourier features for age encoding')
    parser.add_argument('--n_fourier', type=int, default=8,
                        help='Number of Fourier frequencies for age encoding')
    parser.add_argument('--predict_age', action='store_true', default=False,
                        help='Add age regression head to phenotype loss')
    parser.add_argument('--beta_ctx', type=float, default=0.1,
                        help='Weight of sample-level phenotype loss inside L_ctx')

    # Leakage check
    parser.add_argument('--leakage_check_interval', type=int, default=2,
                        help='Run leakage probe every N epochs (0 to disable)')

    #START - Ablation mode selector
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['A', 'B', 'C', 'D', 'E'],
                        help=(
                            'Ablation mode: '
                            'A=original backbone (no phenotype), '
                            'B=phenotype nodes+injection only (no ctx/orth loss), '
                            'C=full arch with orth only (no ctx loss), '
                            'D=current impl (consistency+orth), '
                            'E=proposal-faithful (metadata prediction from z_ctx+orth)'
                        ))
    #END

    args = parser.parse_args()

    if args.emb_size <= 0:
        parser.error("Embedding size must be positive")
    if args.sen_dim is None:
        args.sen_dim = args.emb_size
    if args.ctx_dim is None:
        args.ctx_dim = max(args.emb_size // 2, 4)

    return args
#END


#START - Ablation configuration
ABLATION_DESCRIPTIONS = {
    'A': 'Original backbone — no phenotype nodes, no injection, no z_ctx losses',
    'B': 'Phenotype nodes + gated injection — no context or orthogonality loss',
    'C': 'Full architecture — orthogonality loss only (no context loss)',
    'D': 'Current implementation — consistency alignment + sample phenotype + orthogonality',
    'E': 'Proposal-faithful — metadata prediction from z_ctx + orthogonality',
}


def apply_ablation_config(args):
    """Override args based on --ablation mode. Returns the mode description."""
    mode = args.ablation
    if mode is None:
        return 'custom (no ablation preset)'

    if mode == 'A':
        args.lambda_ctx = 0.0
        args.lambda_orth = 0.0
        args._skip_phenotype = True
    elif mode == 'B':
        args.lambda_ctx = 0.0
        args.lambda_orth = 0.0
        args._skip_phenotype = False
    elif mode == 'C':
        args.lambda_ctx = 0.0
        args.lambda_orth = 0.01
        args._skip_phenotype = False
    elif mode == 'D':
        # Current implementation defaults — keep as-is
        args._skip_phenotype = False
    elif mode == 'E':
        args._skip_phenotype = False
        args._use_metadata_from_ctx = True

    if not hasattr(args, '_skip_phenotype'):
        args._skip_phenotype = False
    if not hasattr(args, '_use_metadata_from_ctx'):
        args._use_metadata_from_ctx = False

    return ABLATION_DESCRIPTIONS.get(mode, 'custom')
#END


#START - Build phenotype label tensors for loss computation
def build_phenotype_labels(adata, cell_to_sample, sample_ids,
                           disease_col=None, lobe_col=None, age_col=None,
                           sample_col='Sample'):
    """Create per-sample and per-cell label tensors.

    Returns a dict with:
        'disease_sample': (n_samples,) int labels
        'disease_cell':   (n_cells,)   int labels  (for leakage probing)
        'n_diseases':     int
        ... and similarly for lobe / age.
    """
    obs = adata.obs
    labels = {}
    n_samples = len(sample_ids)

    if disease_col and disease_col in obs.columns:
        disease_cats = sorted(obs[disease_col].unique())
        disease_map = {d: i for i, d in enumerate(disease_cats)}
        sample_disease = {}
        for s_idx, s_id in enumerate(sample_ids):
            mask = obs[sample_col] == s_id if sample_col in obs.columns else np.ones(len(obs), dtype=bool)
            sample_disease[s_idx] = disease_map[obs.loc[mask, disease_col].iloc[0]]
        labels['disease_sample'] = torch.tensor(
            [sample_disease[s] for s in range(n_samples)], dtype=torch.long)
        labels['disease_cell'] = torch.tensor(
            [sample_disease[s] for s in cell_to_sample], dtype=torch.long)
        labels['n_diseases'] = len(disease_cats)

    if lobe_col and lobe_col in obs.columns:
        lobe_cats = sorted(obs[lobe_col].unique())
        lobe_map = {l: i for i, l in enumerate(lobe_cats)}
        sample_lobe = {}
        for s_idx, s_id in enumerate(sample_ids):
            mask = obs[sample_col] == s_id if sample_col in obs.columns else np.ones(len(obs), dtype=bool)
            sample_lobe[s_idx] = lobe_map[obs.loc[mask, lobe_col].iloc[0]]
        labels['lobe_sample'] = torch.tensor(
            [sample_lobe[s] for s in range(n_samples)], dtype=torch.long)
        labels['n_lobes'] = len(lobe_cats)

    if age_col and age_col in obs.columns:
        sample_age = {}
        for s_idx, s_id in enumerate(sample_ids):
            mask = obs[sample_col] == s_id if sample_col in obs.columns else np.ones(len(obs), dtype=bool)
            sample_age[s_idx] = float(obs.loc[mask, age_col].iloc[0])
        ages_sample = np.array([sample_age[s] for s in range(n_samples)])
        age_mean, age_std = ages_sample.mean(), ages_sample.std() + 1e-8
        labels['age_sample'] = torch.tensor(
            (ages_sample - age_mean) / age_std, dtype=torch.float32)

    return labels
#END


if __name__ == '__main__':
    args = parse_phenotype_args()

    #START - Apply ablation config and set up results tracking
    ablation_desc = apply_ablation_config(args)
    mode_label = args.ablation if args.ablation else 'custom'
    print(f'Ablation mode: {mode_label} — {ablation_desc}')
    print(vars(args))

    args.output_dir = f"./outputs/{args.exp_name}"
    print("Outputs dir:", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    results_dir = os.path.join('.', 'ablation_results')
    os.makedirs(results_dir, exist_ok=True)
    run_metrics = {
        'mode': mode_label,
        'description': ablation_desc,
        'exp_name': args.exp_name,
        'args': {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, str, bool, type(None)))},
        'part3_epochs': [],
        'part4_epochs': [],
    }
    #END

    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='# %Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger()

    # ====== Part 1: Load and process data ======
    logger.info("====== Part 1: load and process data ======")
    if 'data1' in args.exp_name:
        adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data1()
    elif 'rep' in args.exp_name:
        adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data_rep(args.exp_name)
    elif 'example' in args.exp_name:
        adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_example_data()
    else:
        adata, cluster_cell_ls, cell_cluster_arr, celltype_names = utils.load_data1(args.input_data_count)

    new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names = utils.process_data(
        adata, cluster_cell_ls, cell_cluster_arr, args)

    new_data.write_h5ad(os.path.join(args.output_dir, f'{args.exp_name}_new_data.h5ad'))

    gene_cell = new_data.X.toarray().T
    args.gene_num = gene_cell.shape[0]
    args.cell_num = gene_cell.shape[1]
    print(f'cell num: {new_data.shape[0]}, gene num: {new_data.shape[1]}')

    #START - Build phenotype features and labels from adata metadata
    if getattr(args, '_skip_phenotype', False):
        sample_ids = sorted(new_data.obs[args.sample_col].unique()) if args.sample_col in new_data.obs.columns else ['all']
        phenotype_features = None
        cell_to_sample = [0] * args.cell_num
        age_col_idx = None
        pheno_labels = {}
        print('Ablation A: skipping phenotype feature construction')
    else:
        sample_ids, phenotype_features, cell_to_sample, age_col_idx = build_phenotype_features(
            new_data,
            sample_col=args.sample_col,
            disease_col=args.disease_col,
            lobe_col=args.lobe_col,
            age_col=args.age_col,
            batch_col=args.batch_col,
        )
        print(f'Phenotype: {len(sample_ids)} samples, feature dim={phenotype_features.shape[1]}')

        pheno_labels = build_phenotype_labels(
            new_data, cell_to_sample, sample_ids,
            disease_col=args.disease_col,
            lobe_col=args.lobe_col,
            age_col=args.age_col,
            sample_col=args.sample_col,
        )
    #END

    if args.retrain:
        graph_nx, edge_indexs, ccc_matrix = utils.build_graph_nx(
            new_data, gene_cell, cell_cluster_arr, sen_gene_ls, nonsen_gene_ls, gene_names, args)

    logger.info("Part 1, data loading and processing end!")

    # ====== Part 2: Generate init embeddings ======
    logger.info("====== Part 2: generate init embedding ======")

    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    args.device = device

    def run_scanpy(adata_input, batch_remove=False, batch_name='Sample'):
        adata_copy = adata_input.copy()
        sp.pp.normalize_total(adata_copy, target_sum=1e4)
        sp.pp.log1p(adata_copy)
        sp.pp.scale(adata_copy, max_value=10)
        if batch_remove:
            print('Start remove batch effect ...')
            sp.pp.combat(adata_copy, key=batch_name)
        else:
            print('Do not remove batch effect ...')
        sp.tl.pca(adata_copy, svd_solver='arpack')
        sp.pp.neighbors(adata_copy, n_neighbors=10, n_pcs=40)
        sp.tl.umap(adata_copy, n_components=args.emb_size)
        return adata_copy.obsm['X_umap']

    if args.retrain:
        cell_embed = run_scanpy(new_data.copy(), batch_remove=True)
        print('cell embedding generated!')
        gene_embed = run_scanpy(new_data.copy().T)
        print('gene embedding generated!')
        cell_embed = torch.tensor(cell_embed)
        gene_embed = torch.tensor(gene_embed)

        graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)

        #START - Build extended graph with phenotype nodes
        if getattr(args, '_skip_phenotype', False):
            graph_pyg = build_extended_graph_pyg(
                gene_cell, gene_embed, cell_embed,
                None, cell_to_sample,
                edge_indexs, ccc_matrix,
            )
        else:
            graph_pyg = build_extended_graph_pyg(
                gene_cell, gene_embed, cell_embed,
                phenotype_features, cell_to_sample,
                edge_indexs, ccc_matrix,
            )
        #END

        torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        torch.save(graph_pyg, os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
        print('graph nx and pyg saved!')
    else:
        print('Load graph nx and pyg ...')
        graph_nx = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        graph_pyg = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))

    logger.info("Part 2, embedding generation end!")

    # ====== Part 3: GAT training with phenotype-aware encoder ======
    logger.info("====== Part 3: GAT training ======")

    data = graph_pyg.to(device)
    torch.cuda.empty_cache()

    #START - Prepare phenotype data tensors for training
    pheno_raw = data.pheno_raw.to(device) if hasattr(data, 'pheno_raw') else None
    cell_to_sample_tensor = data.cell_to_sample.to(device) if hasattr(data, 'cell_to_sample') else None

    # Build edge_index for reconstruction loss (local only by default)
    n_genes = data.n_genes
    n_cells = data.n_cells
    n_pheno = data.n_phenotype if hasattr(data, 'n_phenotype') else 0

    if args.reconstruct_ecp:
        recon_edge_index = data.edge_index
    else:
        local_mask = (data.edge_index[0] < n_genes + n_cells) & (data.edge_index[1] < n_genes + n_cells)
        recon_edge_index = data.edge_index[:, local_mask]
    #END

    if args.retrain:
        #START - Initialize phenotype-aware encoder and context objective
        pheno_raw_dim = phenotype_features.shape[1] if phenotype_features is not None else 1
        encoder = PhenotypeAwareEncoder(
            cell_dim=args.emb_size,
            pheno_raw_dim=pheno_raw_dim,
            sen_dim=args.sen_dim,
            ctx_dim=args.ctx_dim,
            use_fourier_age=args.use_fourier_age,
            n_fourier=args.n_fourier,
            age_col_idx=age_col_idx,
        ).to(device)

        #START - Select context objective based on ablation mode
        if getattr(args, '_use_metadata_from_ctx', False):
            ctx_objective = MetadataPredictionFromCtx(
                ctx_dim=args.ctx_dim,
                n_diseases=pheno_labels.get('n_diseases', 0),
                n_lobes=pheno_labels.get('n_lobes', 0),
                predict_age=args.predict_age,
            ).to(device)
        else:
            ctx_objective = ContextObjective(
                pheno_dim=args.emb_size,
                ctx_dim=args.ctx_dim,
                n_diseases=pheno_labels.get('n_diseases', 0),
                n_lobes=pheno_labels.get('n_lobes', 0),
                predict_age=args.predict_age,
                beta=args.beta_ctx,
            ).to(device)
        #END

        # Move label tensors to device
        device_labels = {}
        for k, v in pheno_labels.items():
            if isinstance(v, torch.Tensor):
                device_labels[k] = v.to(device)

        # GAE wrapper for reconstruction loss (uses InnerProductDecoder only;
        # the GATEncoder inside gae_model is unused).
        gae_model = GAEModel(args.emb_size, args.emb_size).to(device)

        all_params = (
            list(encoder.parameters()) +
            list(ctx_objective.parameters())
        )
        optimizer_gat = torch.optim.Adam(all_params, lr=0.001)
        #END

        #START - GAT + phenotype-aware training loop
        leakage_probe = LeakageProbe()

        for epoch in range(args.gat_epoch):
            encoder.train()
            ctx_objective.train()
            optimizer_gat.zero_grad()

            # Forward through phenotype-aware encoder
            x_out, z_sen, z_ctx, gate_values, z_pheno = encoder(
                data, pheno_raw=pheno_raw, cell_to_sample=cell_to_sample_tensor
            )

            # Reconstruction loss (on local graph edges)
            z_for_recon = x_out[:n_genes + n_cells]
            loss_rec = gae_model.recon_loss(z_for_recon, recon_edge_index)

            # Context objective: consistency + sample-level phenotype
            if args.lambda_ctx > 0.0 and z_pheno is not None:
                loss_ctx, ctx_detail = ctx_objective(
                    z_ctx, z_pheno, cell_to_sample_tensor,
                    disease_labels_sample=device_labels.get('disease_sample'),
                    lobe_labels_sample=device_labels.get('lobe_sample'),
                    age_targets_sample=device_labels.get('age_sample'),
                )
            else:
                loss_ctx = None
                ctx_detail = {}

            # Orthogonality loss
            loss_orth = orthogonality_loss(z_sen, z_ctx)

            # Combined loss: reconstruction is the primary DeepSAS objective
            # in Part 3 (no contrastive loss yet), passed as loss_deepsas.
            total_loss = combined_loss(
                loss_deepsas=loss_rec,
                loss_ctx=loss_ctx,
                loss_orth=loss_orth,
                lambda_ctx=args.lambda_ctx,
                lambda_orth=args.lambda_orth,
            )

            total_loss.backward()
            optimizer_gat.step()

            consist_val = ctx_detail.get('consistency', torch.tensor(0.0)).item()
            pheno_val = sum(v.item() for k, v in ctx_detail.items()
                           if k not in ('consistency', 'total'))
            print(f'Epoch {epoch:03d}, Total: {total_loss.item():.4f}, '
                  f'Rec: {loss_rec.item():.4f}, Consist: {consist_val:.4f}, '
                  f'SamplePheno: {pheno_val:.4f}, Orth: {loss_orth.item():.6f}')

            #START - Collect Part 3 epoch metrics
            epoch_metrics = {
                'epoch': epoch,
                'total_loss': total_loss.item(),
                'rec_loss': loss_rec.item(),
                'ctx_loss': loss_ctx.item() if loss_ctx is not None else 0.0,
                'consistency': consist_val,
                'sample_pheno': pheno_val,
                'orth_loss': loss_orth.item(),
            }
            #END

            # Leakage check
            if (args.leakage_check_interval > 0 and
                    epoch > 0 and epoch % args.leakage_check_interval == 0):
                encoder.eval()
                with torch.no_grad():
                    _, z_sen_eval, z_ctx_eval, _, _ = encoder(
                        data, pheno_raw=pheno_raw, cell_to_sample=cell_to_sample_tensor
                    )
                if 'disease_cell' in device_labels:
                    pheno_probe = leakage_probe.probe_phenotype_from_sen(
                        z_sen_eval, device_labels['disease_cell']
                    )
                    #START - Record leakage probe results
                    epoch_metrics['pheno_from_z_sen_acc'] = pheno_probe['accuracy']
                    epoch_metrics['pheno_from_z_sen_chance'] = pheno_probe['chance_level']
                    epoch_metrics['pheno_from_z_sen_leaked'] = pheno_probe['leaked']
                    #END
                    print(f'[Epoch {epoch}] Leakage check — phenotype from z_sen: '
                          f'acc={pheno_probe["accuracy"]:.3f} '
                          f'(chance={pheno_probe["chance_level"]:.3f}) '
                          f'{"LEAKED" if pheno_probe["leaked"] else "OK"}')

            run_metrics['part3_epochs'].append(epoch_metrics)
        #END

        torch.save(encoder, os.path.join(args.output_dir, f'{args.exp_name}_encoder.pt'))
        torch.save(ctx_objective, os.path.join(args.output_dir, f'{args.exp_name}_ctx_objective.pt'))
        torch.save(gae_model, os.path.join(args.output_dir, f'{args.exp_name}_GAT.pt'))
        print('Models saved!')
    else:
        GAT_path = os.path.join(args.output_dir, f'{args.exp_name}_GAT.pt')
        print(f'Load GAT from {GAT_path}')
        gae_model = torch.load(GAT_path)
        gae_model = gae_model.to(device)
        encoder = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_encoder.pt'))
        encoder = encoder.to(device)

    torch.cuda.empty_cache()
    logger.info("Part 3, GAT training end!")

    # ====== Part 4: Contrastive learning ======
    logger.info("====== Part 4: Contrastive learning ======")

    def check_celltypes(predicted_cell_indexs, graph_nx_local, celltype_names_local):
        cell_types = []
        for i in predicted_cell_indexs:
            cluster = graph_nx_local.nodes[int(i)]['cluster']
            cell_types.append(celltype_names_local[cluster])
        from collections import Counter
        print("snc in different cell types: ", Counter(cell_types))

    def build_cell_dict(gene_cell_local, predicted_cell_indexs, GAT_embeddings, graph_nx_local):
        sencell_dict = {}
        nonsencell_dict = {}
        for i in range(gene_cell_local.shape[0], gene_cell_local.shape[0] + gene_cell_local.shape[1]):
            if i in predicted_cell_indexs:
                sencell_dict[i] = [
                    GAT_embeddings[i],
                    graph_nx_local.nodes[int(i)]['cluster'],
                    0,
                    i]
            else:
                nonsencell_dict[i] = [
                    GAT_embeddings[i],
                    graph_nx_local.nodes[int(i)]['cluster'],
                    0,
                    i]
        return sencell_dict, nonsencell_dict

    def identify_sengene_v1(sencell_dict, gene_cell_local, edge_index_selfloop, attention_scores, sen_gene_ls_local):
        print("identify_sengene_v1 ... ")
        cell_index = torch.tensor(list(sencell_dict.keys()))
        cell_mask = torch.zeros(gene_cell_local.shape[0] + gene_cell_local.shape[1], dtype=torch.bool)
        cell_mask[cell_index] = True
        res = []
        score_sengene_ls = []
        for gene_index in range(gene_cell_local.shape[0]):
            connected_cells = edge_index_selfloop[0][edge_index_selfloop[1] == gene_index]
            masked_connected_cells = connected_cells[cell_mask[connected_cells]]
            if masked_connected_cells.numel() == 0:
                res.append(0)
            else:
                tmp = attention_scores[edge_index_selfloop[1] == gene_index]
                attention_edge = torch.sum(tmp[cell_mask[connected_cells]], dim=1)
                attention_s = torch.mean(attention_edge)
                res.append(attention_s.item())
            if gene_index in sen_gene_ls_local:
                score_sengene_ls.append(res[-1])
        num = 10
        res1 = torch.tensor(res)
        new_genes = torch.argsort(res1)[-num:]
        score_sengene_ls = torch.tensor(score_sengene_ls)
        if isinstance(sen_gene_ls_local, torch.Tensor):
            new_sen_gene_ls = sen_gene_ls_local[torch.argsort(score_sengene_ls)[num:].tolist()]
        else:
            new_sen_gene_ls = torch.tensor(sen_gene_ls_local)[torch.argsort(score_sengene_ls)[num:].tolist()]
        new_sen_gene_ls = torch.cat((new_sen_gene_ls, new_genes))
        return new_sen_gene_ls

    def generate_ct_specific_scores(sen_gene_ls_local, gene_cell_local, edge_index_selfloop,
                                    attention_scores, graph_nx_local, celltype_names_local):
        print('generate_ct_specific_scores ...')
        gene_index = torch.tensor(sen_gene_ls_local)
        gene_mask = torch.zeros(gene_cell_local.shape[0] + gene_cell_local.shape[1], dtype=torch.bool)
        gene_mask[gene_index] = True
        ct_specific_scores = {}
        for cell_index in range(gene_cell_local.shape[0], gene_cell_local.shape[0] + gene_cell_local.shape[1]):
            connected_genes = edge_index_selfloop[0][edge_index_selfloop[1] == cell_index]
            if len(connected_genes[gene_mask[connected_genes]]) == 0:
                continue
            else:
                attention_edge = torch.sum(
                    attention_scores[edge_index_selfloop[1] == cell_index][gene_mask[connected_genes]], axis=1)
                attention_s = torch.mean(attention_edge)
                cluster = graph_nx_local.nodes[int(cell_index)]['cluster']
                if cluster in ct_specific_scores:
                    ct_specific_scores[cluster].append([float(attention_s), int(cell_index)])
                else:
                    ct_specific_scores[cluster] = [[float(attention_s), int(cell_index)]]
        return ct_specific_scores

    def calculate_outliers_v1(scores_index):
        scores_index = np.array(scores_index)
        snc_index = []
        outliers_ls = []
        scores = scores_index[:, 0]
        indexs = scores_index[:, 1]
        Q1 = np.percentile(scores, 25)
        Q3 = np.percentile(scores, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        for i, score in enumerate(scores):
            if score > upper_bound:
                snc_index.append(indexs[i])
                outliers_ls.append([score, indexs[i]])
        return len(snc_index), snc_index, outliers_ls

    def extract_cell_indexs(ct_specific_scores):
        print("extract_cell_indexs ... ")
        snc_indexs = []
        for key, values in ct_specific_scores.items():
            values_ls = np.array(values)
            counts, snc_index, outliers_ls = calculate_outliers_v1(values_ls)
            if counts >= 10:
                snc_indexs = snc_indexs + snc_index
        return snc_indexs

    # Contrastive learning loop using z_sen from the phenotype-aware encoder
    cellmodel = Sencell(args.emb_size).to(device)
    data = data.to(device)
    lr = args.learning_rate
    optimizer = torch.optim.Adam(cellmodel.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.85)
    sencell_dict = None

    # Get attention scores from the phenotype-aware encoder (not the
    # disconnected gae_model encoder) so weights are consistent.
    encoder.eval()
    with torch.no_grad():
        edge_index_selfloop_cell, attention_scores_cell = encoder.get_attention_scores(
            data, pheno_raw=pheno_raw, cell_to_sample=cell_to_sample_tensor
        )
    attention_scores_cell = attention_scores_cell.cpu().detach()
    edge_index_selfloop_cell = edge_index_selfloop_cell.cpu().detach()

    #START - Contrastive learning loop with disentangled embeddings
    leakage_probe = LeakageProbe()

    for epoch in range(5):
        print(f'{datetime.datetime.now()}: Contrastive learning Epoch: {epoch:03d}')

        ct_specific_scores = generate_ct_specific_scores(
            sen_gene_ls, gene_cell, edge_index_selfloop_cell,
            attention_scores_cell, graph_nx, celltype_names)

        predicted_cell_indexs = extract_cell_indexs(ct_specific_scores)
        check_celltypes(predicted_cell_indexs, graph_nx, celltype_names)

        if sencell_dict is not None:
            old_sencell_dict = sencell_dict
        else:
            old_sencell_dict = None

        # Get embeddings from phenotype-aware encoder (use z_sen for contrastive)
        with torch.no_grad():
            x_out, z_sen, z_ctx, gate_values, _ = encoder(
                data, pheno_raw=pheno_raw, cell_to_sample=cell_to_sample_tensor
            )

        # Replace the cell portion of x_out with z_sen so the contrastive
        # loop operates on the disentangled senescence branch only.
        GAT_embeddings = x_out.detach().clone()
        GAT_embeddings[n_genes:n_genes + n_cells] = z_sen.detach()
        sencell_dict, nonsencell_dict = build_cell_dict(
            gene_cell, predicted_cell_indexs, GAT_embeddings, graph_nx)

        if old_sencell_dict is not None:
            ratio_cell = utils.get_sencell_cover(old_sencell_dict, sencell_dict)

        # Contrastive cell optimization (operates on z_sen branch embeddings)
        cellmodel, sencell_dict, nonsencell_dict = cell_optim(
            cellmodel, optimizer, sencell_dict, nonsencell_dict,
            None, args, train=True)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print('current lr:', current_lr)

        # Update embeddings
        new_GAT_embeddings = GAT_embeddings
        for key, value in sencell_dict.items():
            new_GAT_embeddings[key] = sencell_dict[key][2].detach()
        for key, value in nonsencell_dict.items():
            new_GAT_embeddings[key] = nonsencell_dict[key][2].detach()

        data.x = new_GAT_embeddings
        with torch.no_grad():
            edge_index_selfloop, attention_scores = encoder.get_attention_scores(
                data, pheno_raw=pheno_raw, cell_to_sample=cell_to_sample_tensor
            )
        attention_scores = attention_scores.to('cpu')
        edge_index_selfloop = edge_index_selfloop.to('cpu')

        old_sengene_indexs = sen_gene_ls
        sen_gene_ls = identify_sengene_v1(
            sencell_dict, gene_cell, edge_index_selfloop, attention_scores, sen_gene_ls)
        ratio_gene = utils.get_sengene_cover(old_sengene_indexs, sen_gene_ls)

        # Gate value interpretability
        if gate_values is not None:
            gate_df, gate_summary = extract_gate_values(
                gate_values, cell_to_sample, graph_nx, args.gene_num,
                celltype_names=celltype_names, sample_ids=sample_ids,
            )
            print_gate_summary(gate_summary)

        #START - Collect Part 4 epoch metrics
        p4_metrics = {
            'epoch': epoch,
            'n_senescent_cells': len(predicted_cell_indexs),
            'n_senescent_genes': len(sen_gene_ls),
            'lr': current_lr,
        }

        # Count senescent cells per cell type
        snc_celltypes = {}
        for idx in predicted_cell_indexs:
            ct = celltype_names[graph_nx.nodes[int(idx)]['cluster']]
            snc_celltypes[ct] = snc_celltypes.get(ct, 0) + 1
        p4_metrics['snc_per_celltype'] = snc_celltypes

        if gate_values is not None and 'gate_summary' in dir():
            p4_metrics['gate_mean'] = {str(k): v.get('mean', 0.0)
                                       for k, v in gate_summary.get('per_celltype', {}).items()}
        #END

        # Leakage checks during contrastive learning
        if (args.leakage_check_interval > 0 and
                epoch % args.leakage_check_interval == 0 and
                'disease_cell' in pheno_labels):
            with torch.no_grad():
                _, z_sen_eval, z_ctx_eval, _, _ = encoder(
                    data, pheno_raw=pheno_raw, cell_to_sample=cell_to_sample_tensor
                )
            # Build senescence binary labels from current predictions
            sen_binary = torch.zeros(n_cells, dtype=torch.long)
            for idx in predicted_cell_indexs:
                cell_offset = int(idx) - args.gene_num
                if 0 <= cell_offset < n_cells:
                    sen_binary[cell_offset] = 1
            pheno_probe_p4, sen_probe_p4 = leakage_probe.run_leakage_checks(
                z_sen_eval, z_ctx_eval,
                phenotype_labels=pheno_labels['disease_cell'],
                senescence_labels=sen_binary,
                epoch=epoch,
            )
            #START - Record Part 4 leakage results
            p4_metrics['pheno_from_z_sen_acc'] = pheno_probe_p4['accuracy']
            p4_metrics['pheno_from_z_sen_leaked'] = pheno_probe_p4['leaked']
            p4_metrics['sen_from_z_ctx_acc'] = sen_probe_p4['accuracy']
            p4_metrics['sen_from_z_ctx_leaked'] = sen_probe_p4['leaked']
            #END

        run_metrics['part4_epochs'].append(p4_metrics)

        torch.save([sencell_dict, sen_gene_ls, attention_scores, edge_index_selfloop],
                   os.path.join(args.output_dir, f'{args.exp_name}_sencellgene-epoch{epoch}.data'))
    #END

    #START - Save ablation results to JSON
    results_path = os.path.join(results_dir, f'{mode_label}_{args.exp_name}.json')
    with open(results_path, 'w') as f:
        json.dump(run_metrics, f, indent=2, default=str)
    print(f'Ablation results saved to {results_path}')
    #END

    logger.info("Training complete!")
