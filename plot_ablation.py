"""
Plot ablation experiment results from ablation_results/ JSON files.

Usage:
    python plot_ablation.py                          # all results
    python plot_ablation.py --modes A D E             # specific modes
    python plot_ablation.py --results_dir ./my_runs   # custom dir
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


#START - Load and organize ablation result files
def load_results(results_dir, modes=None):
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            data = json.load(f)
        mode = data.get('mode', fname.replace('.json', ''))
        if modes and mode not in modes:
            continue
        results[mode] = data
    return results
#END


#START - Plotting helpers
def _extract_series(epochs_list, key, default=None):
    xs, ys = [], []
    for ep in epochs_list:
        if key in ep:
            xs.append(ep['epoch'])
            ys.append(ep[key])
        elif default is not None:
            xs.append(ep['epoch'])
            ys.append(default)
    return xs, ys


def plot_part3_losses(results, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ('total_loss', 'Total Loss'),
        ('rec_loss', 'Reconstruction Loss'),
        ('ctx_loss', 'Context Loss'),
        ('consistency', 'Consistency'),
        ('sample_pheno', 'Sample Phenotype Loss'),
        ('orth_loss', 'Orthogonality Loss'),
    ]

    for ax, (key, title) in zip(axes.flat, metrics):
        for mode, data in results.items():
            xs, ys = _extract_series(data.get('part3_epochs', []), key)
            if xs:
                ax.plot(xs, ys, marker='.', label=mode, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Part 3 (GAT Training) — Loss Comparison', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'part3_losses.png'), dpi=150)
    plt.close(fig)
    print(f'Saved part3_losses.png')


def plot_leakage(results, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Part 3: phenotype from z_sen
    ax = axes[0]
    for mode, data in results.items():
        xs, ys = _extract_series(data.get('part3_epochs', []), 'pheno_from_z_sen_acc')
        if xs:
            ax.plot(xs, ys, marker='o', label=mode, alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='chance (2-class)')
    ax.set_title('Part 3 — Phenotype Leakage into z_sen')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Probe Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Part 4: both probes
    ax = axes[1]
    for mode, data in results.items():
        xs_p, ys_p = _extract_series(data.get('part4_epochs', []), 'pheno_from_z_sen_acc')
        xs_s, ys_s = _extract_series(data.get('part4_epochs', []), 'sen_from_z_ctx_acc')
        if xs_p:
            ax.plot(xs_p, ys_p, marker='o', label=f'{mode} pheno→z_sen', alpha=0.8)
        if xs_s:
            ax.plot(xs_s, ys_s, marker='s', linestyle='--',
                    label=f'{mode} sen→z_ctx', alpha=0.8)
    ax.set_title('Part 4 — Leakage Probes')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Probe Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'leakage_probes.png'), dpi=150)
    plt.close(fig)
    print(f'Saved leakage_probes.png')


def plot_senescent_cells(results, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total senescent cell count over contrastive epochs
    ax = axes[0]
    for mode, data in results.items():
        xs, ys = _extract_series(data.get('part4_epochs', []), 'n_senescent_cells')
        if xs:
            ax.plot(xs, ys, marker='o', label=mode, alpha=0.8)
    ax.set_title('Senescent Cell Count per Epoch')
    ax.set_xlabel('Contrastive Epoch')
    ax.set_ylabel('# Senescent Cells')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final epoch: per-celltype breakdown (grouped bar chart)
    ax = axes[1]
    final_data = {}
    all_celltypes = set()
    for mode, data in results.items():
        p4 = data.get('part4_epochs', [])
        if p4:
            ct_counts = p4[-1].get('snc_per_celltype', {})
            final_data[mode] = ct_counts
            all_celltypes.update(ct_counts.keys())
    all_celltypes = sorted(all_celltypes)
    if final_data and all_celltypes:
        x_pos = np.arange(len(all_celltypes))
        width = 0.8 / max(len(final_data), 1)
        for i, (mode, ct_counts) in enumerate(final_data.items()):
            vals = [ct_counts.get(ct, 0) for ct in all_celltypes]
            ax.bar(x_pos + i * width, vals, width, label=mode, alpha=0.8)
        ax.set_xticks(x_pos + width * (len(final_data) - 1) / 2)
        ax.set_xticklabels(all_celltypes, rotation=45, ha='right', fontsize=7)
        ax.set_title('Final Epoch — SnC per Cell Type')
        ax.set_ylabel('# Senescent Cells')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'senescent_cells.png'), dpi=150)
    plt.close(fig)
    print(f'Saved senescent_cells.png')


def plot_gate_values(results, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False
    for mode, data in results.items():
        p4 = data.get('part4_epochs', [])
        for ep in p4:
            gmean = ep.get('gate_mean', {})
            if gmean:
                has_data = True
                cts = sorted(gmean.keys())
                vals = [gmean[ct] for ct in cts]
                ax.bar([f'{mode}\n{ct}' for ct in cts], vals, alpha=0.7, label=mode)
    if has_data:
        ax.set_title('Mean Gate Values per Cell Type (final epoch)')
        ax.set_ylabel('Gate Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No gate data available\n(modes A may not have gates)',
                ha='center', va='center', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'gate_values.png'), dpi=150)
    plt.close(fig)
    print(f'Saved gate_values.png')
#END


#START - Summary table
def print_summary_table(results):
    header = f'{"Mode":<8} {"Desc":<60} {"FinalRec":>10} {"FinalOrth":>10} {"SnC":>6} {"PhLeaked":>10} {"SenLeaked":>10}'
    print('\n' + '=' * len(header))
    print('ABLATION SUMMARY')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for mode, data in sorted(results.items()):
        desc = data.get('description', '')[:58]
        p3 = data.get('part3_epochs', [])
        p4 = data.get('part4_epochs', [])
        rec = p3[-1]['rec_loss'] if p3 else float('nan')
        orth = p3[-1]['orth_loss'] if p3 else float('nan')
        snc = p4[-1].get('n_senescent_cells', 0) if p4 else 0
        ph_leak = p4[-1].get('pheno_from_z_sen_leaked', '-') if p4 else '-'
        sen_leak = p4[-1].get('sen_from_z_ctx_leaked', '-') if p4 else '-'
        print(f'{mode:<8} {desc:<60} {rec:>10.4f} {orth:>10.6f} {snc:>6} {str(ph_leak):>10} {str(sen_leak):>10}')
    print('=' * len(header) + '\n')
#END


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ablation results')
    parser.add_argument('--results_dir', type=str, default='./ablation_results')
    parser.add_argument('--modes', nargs='*', default=None,
                        help='Filter to specific modes (e.g. A D E)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots (default: same as results_dir)')
    args = parser.parse_args()

    save_dir = args.save_dir or args.results_dir
    os.makedirs(save_dir, exist_ok=True)

    results = load_results(args.results_dir, modes=args.modes)
    if not results:
        print(f'No results found in {args.results_dir}')
        exit(1)

    print(f'Loaded {len(results)} ablation result(s): {", ".join(sorted(results.keys()))}')

    print_summary_table(results)
    plot_part3_losses(results, save_dir)
    plot_leakage(results, save_dir)
    plot_senescent_cells(results, save_dir)
    plot_gate_values(results, save_dir)

    print(f'\nAll plots saved to {save_dir}/')
