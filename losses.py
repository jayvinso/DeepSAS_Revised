#
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear


#START - Context objective: consistency + sample-level phenotype classification
class ContextObjective(nn.Module):
    """Context loss for z_ctx disentanglement.

    L_ctx = L_consistency + beta * L_sample_phenotype

    L_consistency: cosine alignment between each cell's z_ctx and the
        projected phenotype-node embedding for that cell's sample.
    L_sample_phenotype: small classifier/regressor applied to the
        phenotype-node embeddings (one per sample, NOT per cell).
    """
    def __init__(self, pheno_dim, ctx_dim, n_diseases=2, n_lobes=0,
                 predict_age=False, beta=0.1):
        super().__init__()
        self.beta = beta
        self.n_diseases = n_diseases
        self.n_lobes = n_lobes
        self.predict_age = predict_age

        # Projection to align z_pheno (pheno_dim) → ctx_dim so cosine
        # similarity with z_ctx works when dimensions differ.
        self.align_proj = Linear(pheno_dim, ctx_dim, bias=False)

        # Sample-level phenotype heads (operate on z_pheno, not per-cell)
        if n_diseases > 0:
            self.disease_head = nn.Sequential(
                Linear(pheno_dim, pheno_dim),
                nn.ReLU(),
                Linear(pheno_dim, n_diseases),
            )
        if n_lobes > 0:
            self.lobe_head = nn.Sequential(
                Linear(pheno_dim, pheno_dim),
                nn.ReLU(),
                Linear(pheno_dim, n_lobes),
            )
        if predict_age:
            self.age_head = nn.Sequential(
                Linear(pheno_dim, pheno_dim),
                nn.ReLU(),
                Linear(pheno_dim, 1),
            )

    def forward(self, z_ctx, z_pheno, cell_to_sample,
                disease_labels_sample=None, lobe_labels_sample=None,
                age_targets_sample=None):
        """Compute L_ctx = L_consistency + beta * L_sample_phenotype.

        Args:
            z_ctx: (n_cells, ctx_dim) context embeddings per cell.
            z_pheno: (n_samples, pheno_dim) projected phenotype-node embeddings.
            cell_to_sample: (n_cells,) mapping cell -> sample index.
            disease_labels_sample: (n_samples,) per-sample disease labels.
            lobe_labels_sample: (n_samples,) per-sample lobe labels.
            age_targets_sample: (n_samples,) per-sample normalised age.

        Returns:
            loss_ctx: scalar combined context loss.
            loss_dict: dict with individual components for logging.
        """
        loss_dict = {}

        # --- L_consistency: cosine alignment between z_ctx and z_pheno ---
        z_pheno_per_cell = self.align_proj(z_pheno[cell_to_sample])  # (n_cells, pheno_dim)
        loss_consistency = 1.0 - F.cosine_similarity(
            z_ctx, z_pheno_per_cell, dim=1).mean()
        loss_dict['consistency'] = loss_consistency

        # --- L_sample_phenotype: small heads on z_pheno (per-sample) ---
        loss_pheno = torch.tensor(0.0, device=z_ctx.device)
        if self.n_diseases > 0 and disease_labels_sample is not None:
            logits = self.disease_head(z_pheno)
            l = F.cross_entropy(logits, disease_labels_sample)
            loss_dict['disease'] = l
            loss_pheno = loss_pheno + l
        if self.n_lobes > 0 and lobe_labels_sample is not None:
            logits = self.lobe_head(z_pheno)
            l = F.cross_entropy(logits, lobe_labels_sample)
            loss_dict['lobe'] = l
            loss_pheno = loss_pheno + l
        if self.predict_age and age_targets_sample is not None:
            pred = self.age_head(z_pheno).squeeze(1)
            l = F.mse_loss(pred, age_targets_sample)
            loss_dict['age'] = l
            loss_pheno = loss_pheno + l

        loss_ctx = loss_consistency + self.beta * loss_pheno
        loss_dict['total'] = loss_ctx
        return loss_ctx, loss_dict


def orthogonality_loss(z_sen, z_ctx):
    """L_orth = ||Z_sen^T @ Z_ctx||_F^2
    
    Prevents phenotype context from becoming a shortcut for senescence.
    """
    z_sen_centered = z_sen - z_sen.mean(dim=0, keepdim=True)
    z_ctx_centered = z_ctx - z_ctx.mean(dim=0, keepdim=True)
    cross = z_sen_centered.T @ z_ctx_centered  # (sen_dim, ctx_dim)
    return torch.norm(cross, p='fro') ** 2 / (z_sen.shape[0] ** 2)


def combined_loss(loss_deepsas, loss_ctx, loss_orth,
                  loss_rec=None, lambda_ctx=0.1, lambda_orth=0.01,
                  lambda_rec=1.0):
    """Combine all loss terms:
    L = L_DeepSAS + lambda_ctx * L_ctx + lambda_orth * L_orth [+ lambda_rec * L_rec]

    Args:
        loss_deepsas: primary objective (reconstruction in Part 3, contrastive in Part 4).
        loss_ctx: scalar context loss from ContextObjective (already includes
                  consistency + beta * sample_phenotype internally).
        loss_orth: orthogonality regulariser.
        loss_rec: optional additional reconstruction term.
    """
    total = loss_deepsas

    if loss_ctx is not None:
        total = total + lambda_ctx * loss_ctx

    total = total + lambda_orth * loss_orth

    if loss_rec is not None:
        total = total + lambda_rec * loss_rec

    return total
#END
