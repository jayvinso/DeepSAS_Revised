import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv


#START - PhenotypeEncoder: type-specific projection for phenotype-context nodes
class PhenotypeEncoder(nn.Module):
    """Projects raw phenotype features into the shared latent dimension.
    
    Implements M_PhiP from the proposal: Z_P(0) = M_PhiP * X_P
    Includes a small MLP with optional Fourier features for continuous age encoding.
    """
    def __init__(self, in_dim, latent_dim, use_fourier_age=False, n_fourier=8):
        super().__init__()
        self.use_fourier_age = use_fourier_age
        self.n_fourier = n_fourier

        if use_fourier_age:
            self.age_fourier_freqs = nn.Parameter(
                torch.randn(n_fourier) * 2.0, requires_grad=False
            )
            effective_in = in_dim - 1 + 2 * n_fourier  # replace 1 age dim with 2*n_fourier
        else:
            effective_in = in_dim

        self.mlp = nn.Sequential(
            Linear(effective_in, latent_dim),
            nn.CELU(),
            Linear(latent_dim, latent_dim),
            nn.CELU(),
        )

    def forward(self, x_pheno, age_col_idx=None):
        """
        Args:
            x_pheno: (n_samples, in_dim) raw phenotype features
            age_col_idx: index of the age column in x_pheno for Fourier encoding
        """
        if self.use_fourier_age and age_col_idx is not None:
            age = x_pheno[:, age_col_idx:age_col_idx+1]
            other = torch.cat([x_pheno[:, :age_col_idx], x_pheno[:, age_col_idx+1:]], dim=1)
            # gamma(age) = [sin(freq_k * age), cos(freq_k * age)]
            fourier = torch.cat([
                torch.sin(self.age_fourier_freqs * age),
                torch.cos(self.age_fourier_freqs * age)
            ], dim=1)
            x_pheno = torch.cat([other, fourier], dim=1)

        return self.mlp(x_pheno)
#END


#START - GatedPhenotypeInjection: gated residual branch for phenotype broadcast
class GatedPhenotypeInjection(nn.Module):
    """Injects phenotype context into cell embeddings via a gated residual.
    
    h_i(l+1) = h_i_local(l+1) + g_i * W_P * h_p_norm
    g_i = sigmoid(W_g * [h_i || h_p_norm])
    h_p_norm = h_p / sqrt(deg(p))
    """
    def __init__(self, cell_dim, pheno_dim):
        super().__init__()
        self.W_P = Linear(pheno_dim, cell_dim, bias=False)
        self.W_g = Linear(cell_dim + pheno_dim, 1, bias=True)

    def forward(self, h_cells, h_pheno, cell_to_sample, sample_degrees):
        """
        Args:
            h_cells: (n_cells, cell_dim) local cell embeddings after GAT
            h_pheno: (n_samples, pheno_dim) phenotype embeddings
            cell_to_sample: (n_cells,) mapping each cell to its sample index
            sample_degrees: (n_samples,) number of cells per sample
        """
        h_p = h_pheno[cell_to_sample]  # (n_cells, pheno_dim)
        deg_norm = torch.sqrt(sample_degrees[cell_to_sample].float().clamp(min=1.0)).unsqueeze(1)
        h_p_norm = h_p / deg_norm

        gate_input = torch.cat([h_cells, h_p_norm], dim=1)
        g = torch.sigmoid(self.W_g(gate_input))  # (n_cells, 1)

        pheno_shift = self.W_P(h_p_norm)  # (n_cells, cell_dim)
        h_out = h_cells + g * pheno_shift

        return h_out, g.squeeze(1)
#END


#START - PhenotypeAwareEncoder: full encoder with disentangled z_sen/z_ctx branches
class PhenotypeAwareEncoder(nn.Module):
    """Extended DeepSAS encoder with phenotype-context support.
    
    Produces disentangled cell embeddings: z_i = [z_i_sen || z_i_ctx]
    - z_sen: senescence-discriminative (used by original contrastive objective)
    - z_ctx: phenotype-context (used by phenotype prediction loss)
    """
    def __init__(self, cell_dim=128, pheno_raw_dim=1, sen_dim=None, ctx_dim=None,
                 use_fourier_age=False, n_fourier=8, age_col_idx=None,
                 dropout=0.3):
        super().__init__()
        if sen_dim is None:
            sen_dim = cell_dim
        if ctx_dim is None:
            ctx_dim = cell_dim // 2

        self.cell_dim = cell_dim
        self.sen_dim = sen_dim
        self.ctx_dim = ctx_dim
        self.age_col_idx = age_col_idx
        self.dropout = dropout

        # Type-specific linear transforms (M_PhiC, M_PhiG from original)
        self.linear_gene = Linear(cell_dim, cell_dim)
        self.linear_cell = Linear(cell_dim, cell_dim)

        # GAT layers for local cell-gene and cell-cell message passing
        self.conv1 = GATConv(cell_dim, cell_dim, add_self_loops=False)
        self.conv2 = GATConv(cell_dim, cell_dim, add_self_loops=False)
        self.act = nn.CELU()

        # Phenotype encoder (M_PhiP)
        self.pheno_encoder = PhenotypeEncoder(
            pheno_raw_dim, cell_dim,
            use_fourier_age=use_fourier_age, n_fourier=n_fourier
        )

        # Gated phenotype injection (1 layer, after local updates)
        self.pheno_gate = GatedPhenotypeInjection(cell_dim, cell_dim)

        # Disentanglement heads
        self.head_sen = nn.Sequential(
            Linear(cell_dim, sen_dim),
            nn.CELU(),
        )
        self.head_ctx = nn.Sequential(
            Linear(cell_dim, ctx_dim),
            nn.CELU(),
        )

    def _resolve_node_counts(self, graph):
        """Derive n_genes, n_cells, n_pheno from graph.node_type (preferred)
        or fall back to stored counts."""
        if hasattr(graph, 'node_type'):
            nt = graph.node_type
            n_genes = int((nt == 0).sum())
            n_cells = int((nt == 1).sum())
            n_pheno = int((nt == 2).sum())
        else:
            n_genes = graph.n_genes if hasattr(graph, 'n_genes') else int(graph.y.sum())
            n_cells = graph.n_cells if hasattr(graph, 'n_cells') else (graph.x.shape[0] - n_genes)
            n_pheno = graph.n_phenotype if hasattr(graph, 'n_phenotype') else 0
        return n_genes, n_cells, n_pheno

    def _project_and_encode_pheno(self, x, edge_index, graph, n_genes, n_cells,
                                   n_pheno, pheno_raw, device):
        """Shared logic: type-specific projections, phenotype encoding,
        local GAT message passing. Returns (x_local, x_new, local_edge_index)."""
        # Use node_type for indexing when available (safer than positional slicing)
        if hasattr(graph, 'node_type'):
            is_gene = (graph.node_type == 0)
            is_cell = (graph.node_type == 1)
        else:
            n_total = x.shape[0]
            is_gene = torch.zeros(n_total, dtype=torch.bool, device=device)
            is_gene[:n_genes] = True
            is_cell = torch.zeros(n_total, dtype=torch.bool, device=device)
            is_cell[n_genes:n_genes + n_cells] = True

        x_gene = F.relu(self.linear_gene(x[is_gene]))
        x_cell = F.relu(self.linear_cell(x[is_cell]))

        x_new = torch.zeros_like(x)
        x_new[is_gene] = x_gene
        x_new[is_cell] = x_cell
        if pheno_raw is not None and n_pheno > 0:
            z_pheno = self.pheno_encoder(pheno_raw.to(device), age_col_idx=self.age_col_idx)
            if hasattr(graph, 'node_type'):
                x_new[graph.node_type == 2] = z_pheno
            else:
                x_new[n_genes + n_cells:] = z_pheno

        local_mask = (edge_index[0] < n_genes + n_cells) & (edge_index[1] < n_genes + n_cells)
        local_edge_index = edge_index[:, local_mask]

        x_local = self.conv1(x_new, local_edge_index)
        x_local = self.act(x_local)
        x_local = F.dropout(x_local, p=self.dropout, training=self.training)
        x_local = self.conv2(x_local, local_edge_index)
        x_local = self.act(x_local)

        return x_local, x_new, local_edge_index

    def forward(self, graph, pheno_raw=None, cell_to_sample=None):
        """
        Args:
            graph: PyG Data object with x, edge_index, node_type
            pheno_raw: (n_samples, pheno_raw_dim) raw phenotype features
            cell_to_sample: (n_cells,) sample assignment per cell
            
        Returns:
            x_out: full node embeddings (genes + cells + phenotype)
            z_sen: (n_cells, sen_dim) senescence branch embeddings
            z_ctx: (n_cells, ctx_dim) context branch embeddings
            gate_values: (n_cells,) gate values for interpretability
        """
        x, edge_index = graph.x, graph.edge_index
        n_genes, n_cells, n_pheno = self._resolve_node_counts(graph)

        x_local, x_new, _ = self._project_and_encode_pheno(
            x, edge_index, graph, n_genes, n_cells, n_pheno, pheno_raw, x.device)

        # Extract cell embeddings after local message passing
        h_cells_local = x_local[n_genes:n_genes + n_cells]

        # Gated phenotype injection
        gate_values = None
        if pheno_raw is not None and cell_to_sample is not None and n_pheno > 0:
            z_pheno = x_new[n_genes + n_cells:]
            sample_degrees = torch.bincount(
                cell_to_sample.to(x.device), minlength=n_pheno
            )
            h_cells_injected, gate_values = self.pheno_gate(
                h_cells_local, z_pheno, cell_to_sample.to(x.device), sample_degrees
            )
        else:
            h_cells_injected = h_cells_local

        # Disentangled embeddings
        z_sen = self.head_sen(h_cells_injected)
        z_ctx = self.head_ctx(h_cells_injected)

        # Build full output embedding (for compatibility with GAE decoder)
        x_out = x_local.clone()
        x_out[n_genes:n_genes + n_cells] = h_cells_injected

        #START - Expose projected phenotype node embeddings for context loss
        z_pheno = x_new[n_genes + n_cells:] if n_pheno > 0 else None
        #END

        return x_out, z_sen, z_ctx, gate_values, z_pheno

    def get_attention_scores(self, graph, pheno_raw=None, **kwargs):
        """Return (edge_index_with_selfloop, attention_weights) from conv1.

        Note: returns first-layer (conv1) attention only — this captures
        direct gene-cell edge weights used by identify_sengene_v1. It does
        NOT reflect conv2, phenotype injection, or disentanglement heads.
        """
        x, edge_index = graph.x, graph.edge_index
        n_genes, n_cells, n_pheno = self._resolve_node_counts(graph)

        # Reuse shared projection + phenotype encoding (but not the full
        # GAT forward — we only need the projected x_new for conv1).
        _, x_new, local_edge_index = self._project_and_encode_pheno(
            x, edge_index, graph, n_genes, n_cells, n_pheno, pheno_raw, x.device
        )

        _, (edge_index_att, alpha) = self.conv1(
            x_new, local_edge_index, return_attention_weights=True
        )
        return edge_index_att, alpha
#END
