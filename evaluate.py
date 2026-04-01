import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np


#START - Leakage probing for disentangled embeddings
class LeakageProbe:
    """Validates clean separation between z_sen and z_ctx.
    
    Tests:
    1. Whether phenotype metadata can be predicted from z_sen (should fail).
    2. Whether senescence labels leak into z_ctx (should fail).
    
    Uses stratified 5-fold cross-validation to avoid overfitting on train data.
    """
    def __init__(self, n_folds=5):
        self.n_folds = n_folds

    def _probe_cv(self, z, labels):
        """Run stratified k-fold logistic regression and return mean accuracy."""
        z_np = z.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy() if isinstance(
            labels, torch.Tensor) else np.array(labels)

        n_classes = len(np.unique(labels_np))
        if n_classes < 2:
            return {'accuracy': 0.0, 'chance_level': 1.0, 'leaked': False}

        # Need at least n_folds samples per class for stratified CV
        min_class_count = min(np.bincount(labels_np.astype(int)))
        k = min(self.n_folds, min_class_count)
        if k < 2:
            return {'accuracy': 0.0, 'chance_level': 1.0 / n_classes, 'leaked': False}

        clf = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='multinomial')
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(clf, z_np, labels_np, cv=cv, scoring='accuracy')
        acc = float(scores.mean())
        chance = 1.0 / n_classes

        return {
            'accuracy': acc,
            'chance_level': chance,
            'leaked': acc > chance + 0.15,
        }

    def probe_phenotype_from_sen(self, z_sen, phenotype_labels):
        """Check if phenotype can be predicted from z_sen (should fail)."""
        return self._probe_cv(z_sen, phenotype_labels)

    def probe_senescence_from_ctx(self, z_ctx, senescence_labels):
        """Check if senescence labels can be predicted from z_ctx (should fail)."""
        return self._probe_cv(z_ctx, senescence_labels)

    def run_leakage_checks(self, z_sen, z_ctx, phenotype_labels,
                           senescence_labels, epoch=None):
        """Run both leakage probes and print results."""
        prefix = f"[Epoch {epoch}] " if epoch is not None else ""

        pheno_probe = self.probe_phenotype_from_sen(z_sen, phenotype_labels)
        sen_probe = self.probe_senescence_from_ctx(z_ctx, senescence_labels)

        print(f"{prefix}Leakage check — phenotype from z_sen: "
              f"acc={pheno_probe['accuracy']:.3f} "
              f"(chance={pheno_probe['chance_level']:.3f}) "
              f"{'LEAKED' if pheno_probe['leaked'] else 'OK'}")

        print(f"{prefix}Leakage check — senescence from z_ctx: "
              f"acc={sen_probe['accuracy']:.3f} "
              f"(chance={sen_probe['chance_level']:.3f}) "
              f"{'LEAKED' if sen_probe['leaked'] else 'OK'}")

        return pheno_probe, sen_probe
#END
