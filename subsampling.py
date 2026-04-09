import scanpy as sc
import numpy as np
import os
import argparse

def stratified_subsample(adata, sample_size, cluster_key='clusters', already_sampled=set()):
    """
    Perform stratified subsampling on an AnnData object **without replacement**.

    Parameters:
    - adata: AnnData object
    - sample_size: Number of cells to subsample
    - cluster_key: Key in adata.obs to use for stratification
    - already_sampled: Set of previously sampled cell indices to exclude

    Returns:
    - Subsampled AnnData object with unique cells
    """
    # Exclude already sampled cells
    available_indices = list(set(adata.obs.index) - already_sampled)

    # Stop if no more cells are left
    if len(available_indices) == 0:
        return None

    # Subset to only available cells
    available_adata = adata[available_indices]

    # Get cluster proportions from available cells
    cluster_counts = available_adata.obs[cluster_key].value_counts()
    cluster_proportions = cluster_counts / cluster_counts.sum()

    # Compute per-cluster sample sizes
    cluster_sample_sizes = (cluster_proportions * sample_size).round().astype(int)

    # Perform stratified sampling
    sampled_indices = set()
    for cluster, count in cluster_sample_sizes.items():
        cluster_indices = available_adata.obs.index[available_adata.obs[cluster_key] == cluster]
        
        # Sample without replacement, ensuring unique selections
        selected = np.random.choice(cluster_indices, size=min(len(cluster_indices), count), replace=False)
        sampled_indices.update(selected)

    # Update the set of already sampled cells
    already_sampled.update(sampled_indices)

    # Return the subsampled AnnData object
    return adata[list(sampled_indices)].copy()


def main(input_file, subsample_size, output_dir):
    """
    Main function to perform **non-overlapping** subsampling and save results.

    Parameters:
    - input_file: Path to input .h5ad file
    - subsample_size: Number of cells per subsample
    - output_dir: Directory to save subsampled datasets
    """
    # Load the dataset
    print(f"Loading data from: {input_file}")
    adata = sc.read_h5ad(input_file)

    # Compute number of possible iterations
    num_iterations = int(np.ceil(adata.n_obs / subsample_size))
    print(f"Total cells: {adata.n_obs} | Subsample size: {subsample_size} | Estimated max iterations: {num_iterations}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Track already sampled cells
    already_sampled = set()

    # Perform non-overlapping subsampling and save outputs
    for i in range(num_iterations):
        subsampled_adata = stratified_subsample(adata, subsample_size, cluster_key='clusters', already_sampled=already_sampled)

        # Stop when all cells have been sampled
        if subsampled_adata is None or subsampled_adata.n_obs == 0:
            print("All cells have been sampled. Stopping further iterations.")
            break

        # Save the subsampled dataset
        output_file = os.path.join(output_dir, f"subsample_{i}.h5ad")
        subsampled_adata.write(output_file)
        print(f"Iteration {i+1}: Saved {subsampled_adata.n_obs} cells to {output_file}")

    print(f"Subsampling completed! All files saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample an h5ad file ensuring unique cells per batch.")
    parser.add_argument("--input", required=True, help="Path to the input .h5ad file")
    parser.add_argument("--subsample_size", type=int, required=True, help="Number of cells per subsample")
    parser.add_argument("--output", required=True, help="Directory to save the output subsampled files")

    args = parser.parse_args()
    main(args.input, args.subsample_size, args.output)
#teest