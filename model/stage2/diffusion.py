import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torch.utils.checkpoint import checkpoint


# ==========================================
# 1. Utils & Embeddings
# ==========================================

class DiffusionEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion time steps.
    Projects scalar time steps into a high-dimensional feature space.
    """

    def __init__(self, num_steps, embedding_dim):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(num_steps, embedding_dim / 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


def index_to_log_onehot(x, num_classes):
    """
    Converts indices to log-one-hot representation for discrete diffusion.
    """
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    """
    Converts log-one-hot representation back to indices.
    """
    return log_x.argmax(1)


# ==========================================
# 2. Core Modules: Regime-Adaptive Denoising Block
# ==========================================

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network (GCN) Layer.
    Refines spatial features based on the learned adjacency matrix from Stage 1.
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Args:
            x: Input features [B, C, N, D]
            adj: Learned adjacency matrix [B, C, C]
        """
        x_proj = self.proj(x)
        B, C, N, D = x_proj.shape
        # Flatten patches to apply GCN across the variable dimension
        x_flat = rearrange(x_proj, 'b c n d -> b c (n d)')
        out = torch.bmm(adj, x_flat)
        out = rearrange(out, 'b c (n d) -> b c n d', n=N)
        return self.dropout(out)


class CSDI_GCN_Block(nn.Module):
    """
    Factorized Spatio-Temporal Denoising Block.
    Integrates AdaLN-Zero modulation for regime adaptation, temporal attention,
    and spatial GCN refinement.
    """

    def __init__(self, d_model, nhead, d_side_info, dropout=0.1):
        super().__init__()

        # AdaLN Modulation Head: Regresses scale (gamma) and shift (beta) parameters
        # from the condition embedding (time step + global regime).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model)
        )
        # Initialize last layer to zero for identity mapping at initialization
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

        self.cond_projection = nn.Linear(d_side_info, 2 * d_model)

        self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gcn = GCNLayer(d_model, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        self.dropout = nn.Dropout(dropout)
        self.mid_projection = nn.Linear(d_model, 2 * d_model)
        self.output_projection = nn.Linear(d_model, 2 * d_model)

    def modulate(self, x, shift, scale):
        """
        Applies element-wise affine modulation: x = x * (1 + scale) + shift
        """
        out = x.clone()
        out.mul_(scale + 1.0)
        out.add_(shift)
        return out

    def forward(self, x, side_info, diffusion_emb, adj_matrix):
        """
        Args:
            x: Hidden states [B, C, N, D]
            side_info: Positional and variable embeddings
            diffusion_emb: Combined condition embedding (Time + Regime)
            adj_matrix: Graph structure from Stage 1
        """
        B, C, N, D = x.shape

        # 1. Parameter Regression (AdaLN-Zero)
        # Regress modulation parameters for temporal and spatial branches
        shift_t, scale_t, shift_g, scale_g = self.adaLN_modulation(diffusion_emb).chunk(4, dim=-1)

        # --- 2. Temporal Branch (Transformer) ---
        # A. Norm
        yt = rearrange(x, 'b c n d -> (b c) n d')
        yt = self.norm1(yt)
        yt = rearrange(yt, '(b c) n d -> b c n d', c=C)

        # B. Modulate
        yt = self.modulate(yt, shift_t, scale_t)

        # C. Attention
        yt = rearrange(yt, 'b c n d -> (b c) n d')
        yt, _ = self.time_attn(yt, yt, yt)
        yt = rearrange(yt, '(b c) n d -> b c n d', c=C)

        # D. Residual
        x = x + self.dropout(yt)

        # --- 3. Spatial Branch (GCN) ---
        # A. Norm
        yg = self.norm2(x)

        # B. Modulate
        yg = self.modulate(yg, shift_g, scale_g)

        # C. GCN (Guided by learned structure A)
        out_gcn = self.gcn(yg, adj_matrix)

        # D. Residual
        x = x + out_gcn

        # --- 4. Feed-Forward & Gated Tanh Unit (GTU) ---
        y = self.mid_projection(x)
        cond_feat = self.cond_projection(side_info)

        # Inject side information
        y = y + cond_feat

        # GTU Activation
        gate, filter = torch.chunk(y, 2, dim=-1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        out = self.output_projection(y)
        residual_out, skip_out = torch.chunk(out, 2, dim=-1)

        return (x + residual_out) / math.sqrt(2.0), skip_out


# ==========================================
# 3. Factorized Transformer (Backbone)
# ==========================================

class FactorizedMTSTransformer(nn.Module):
    """
    Backbone for the Regime-Adaptive Discrete Diffusion Model.
    Stacks multiple CSDI_GCN_Blocks to denoise the input tokens.
    """

    def __init__(self, in_channels, config):
        super().__init__()
        self.in_channels = in_channels
        self.num_patch = config['transformer']['num_patch']  # N patches
        self.codebook_size = config['VQ']['codebook_size'] + 1
        self.d_model = config['transformer']['stage2_d_model']
        self.stage1_d_model = config['transformer'].get('stage1_d_model', self.d_model)

        # Projection for global regime vector
        self.global_cond_proj = nn.Linear(self.stage1_d_model, self.d_model)

        # Embedding Dimensions
        self.d_time_emb = 64
        self.d_feature_emb = 16

        # 1. Embeddings
        self.content_emb = nn.Embedding(self.codebook_size, self.d_model)
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config['diffusion']['num_timesteps'],
            embedding_dim=self.d_model
        )
        self.feature_emb = nn.Embedding(self.in_channels, self.d_feature_emb)
        self.d_side_info = self.d_time_emb + self.d_feature_emb

        # 2. Stacked Denoising Layers
        self.layers = nn.ModuleList([
            CSDI_GCN_Block(
                d_model=self.d_model,
                nhead=config['decoder']['n_heads'],
                d_side_info=self.d_side_info,
                dropout=0.1
            )
            for _ in range(config['decoder']['n_layers'])
        ])

        # 3. Output Heads
        self.output_projection1 = nn.Linear(self.d_model, self.d_model)
        self.output_projection2 = nn.Linear(self.d_model, self.codebook_size - 1)

        self.use_checkpointing = config.get('use_checkpointing', True)

    def get_concatenated_side_info(self, B, C, N, device):
        """
        Constructs side information I (Positional Encoding + Variable Embeddings).
        """
        # Position Encoding for Patch Sequence (N)
        pe = torch.zeros(N, self.d_time_emb, device=device)
        position = torch.arange(0, N, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_time_emb, 2).float().to(device) * (-math.log(10000.0) / self.d_time_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        time_emb = pe.unsqueeze(0).unsqueeze(0).expand(B, C, N, -1)

        # Feature/Variable Embedding (C)
        feature_idx = torch.arange(C, device=device)
        feat_emb = self.feature_emb(feature_idx).unsqueeze(0).unsqueeze(2).expand(B, C, N, -1)

        side_info = torch.cat([time_emb, feat_emb], dim=-1)
        return side_info

    def forward(self, x_indices, t, adj_matrix, z_global):
        """
        Forward pass for the denoising backbone.
        Args:
            x_indices: Noisy token indices [B, C, N]
            t: Diffusion time step [B]
            adj_matrix: Learned graph structure [B, C, C]
            z_global: Quantized regime prototype [B, D_stage1]
        """
        B, C, N = x_indices.shape
        device = x_indices.device

        # 1. Base Content Embedding
        x = self.content_emb(x_indices)

        # 2. Condition Construction
        # Embed diffusion step
        diffusion_emb = self.diffusion_embedding(t).view(B, 1, 1, -1)

        # Project global regime prototype
        global_cond = self.global_cond_proj(z_global).view(B, 1, 1, -1)

        # Fuse Step Index + Regime Prototype -> Combined Condition c
        combined_cond_emb = diffusion_emb + global_cond

        # 3. Side Info Construction
        side_info = self.get_concatenated_side_info(B, C, N, device)

        # 4. Sequential Processing
        skip_connections = []
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x, skip = checkpoint(layer, x, side_info, combined_cond_emb, adj_matrix, use_reentrant=False)
            else:
                x, skip = layer(x, side_info, combined_cond_emb, adj_matrix)

            skip_connections.append(skip)

        # 5. Output Projection
        x = torch.sum(torch.stack(skip_connections), dim=0) / math.sqrt(len(self.layers))
        x = self.output_projection1(x)
        x = F.relu(x)
        logits = self.output_projection2(x)  # [B, C, N, Codebook_Size]

        return logits.permute(0, 3, 1, 2)  # [B, Codebook_Size, C, N]


# ==========================================
# 4. Main Model Wrapper: MTSVQDiffusion
# ==========================================

class MTSVQDiffusion(nn.Module):
    """
    Regime-Adaptive Discrete Diffusion Model.
    Manages the forward diffusion process (corruption) and the reverse denoising process.
    """

    def __init__(self, in_channels, config):
        super().__init__()
        self.config = config
        self.stage1_d_model = config['transformer']['stage1_d_model']
        self.d_model = config['transformer']['stage2_d_model']
        self.num_classes = config['VQ']['codebook_size']
        self.num_timesteps = config['diffusion']['num_timesteps']
        self.mask_token_id = self.num_classes

        self.transformer = FactorizedMTSTransformer(in_channels, config)
        self.auxiliary_loss_weight = config.get('aux_weight', 0.1)

        self.register_buffer('alpha_bar', self._get_alpha_bar(self.num_timesteps))

    def _get_alpha_bar(self, T):
        """
        Computes the cosine noise schedule for the absorbing state transition.
        """
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2)
        return alpha_bar

    def forward(self, x_indices, adj_matrix, z_global):
        """
        Training forward pass.
        1. Corrupts clean tokens (x_indices) using the forward process.
        2. Predicts original tokens using the regime-adaptive backbone.
        3. Computes cross-entropy loss.
        """
        B, C, N = x_indices.shape
        device = x_indices.device

        # Sample random time steps
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()

        # q_sample: Corrupt clean tokens to noisy state at step t
        log_x_start = index_to_log_onehot(x_indices, self.num_classes + 1)
        log_xt = self.q_sample(log_x_start, t)
        xt_indices = log_onehot_to_index(log_xt)

        # Denoise: Predict p(x_0 | x_t, g_hat, A)
        logits_model = self.transformer(xt_indices, t, adj_matrix=adj_matrix, z_global=z_global)

        # Calculate Cross-Entropy Loss
        loss_diffusion = F.cross_entropy(logits_model, x_indices)

        return loss_diffusion, logits_model

    def q_sample(self, log_x_start, t):
        """
        Forward Diffusion Process: Absorbing State Transition.
        """
        alpha_t_bar = self.alpha_bar[t].view(-1, 1, 1, 1).to(log_x_start.device)
        log_mask = torch.zeros_like(log_x_start)
        log_mask[:, :-1, :, :] = -1e30
        log_mask[:, -1, :, :] = 0  # Mask token position

        # Mix clean data probability with mask probability based on alpha_bar
        log_xt = torch.logaddexp(
            torch.log(alpha_t_bar + 1e-30) + log_x_start,
            torch.log(1 - alpha_t_bar + 1e-30) + log_mask
        )
        return log_xt

    def fast_iterative_denoise(self, x_masked, adj_matrix, z_global, start_step, steps=10):
        """
        Iterative Denoising (Greedy) with Global Regime Injection.
        Used for imputation during inference.

        Args:
            x_masked: Initial input tokens with masks [B, C, N]
            adj_matrix: Learned graph structure [B, C, C]
            z_global: Global regime prototype [B, D]
            start_step: Starting diffusion step (e.g., M/2)
        """
        B, C, N = x_masked.shape
        device = x_masked.device
        curr_indices = x_masked.clone()
        fixed_mask = (x_masked != self.mask_token_id)

        timesteps = torch.linspace(start_step, 0, steps, device=device, dtype=torch.long)

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict x_0 from current state
            logits_x0 = self.transformer(curr_indices, t_batch, adj_matrix=adj_matrix, z_global=z_global)
            pred_x0 = logits_x0.argmax(dim=1)

            # Update only the masked tokens, keep observed context fixed
            curr_indices[~fixed_mask] = pred_x0[~fixed_mask]

        return curr_indices

    # ==========================================
    # Inference with Stochastic Imputation Protocol
    # ==========================================
    @torch.no_grad()
    def reconstruct_random_masking(self, z_content, adj_matrix, z_global, mask_ratio=0.5):
        """
        Single-Path Reconstruction with Stochastic Random Masking.

        Args:
            z_content: [B, C, N] Input discrete tokens
            adj_matrix: [B, C, C] Graph structure
            z_global: [B, 1, D] Global regime vector
            mask_ratio: Float, proportion of tokens to mask (default 0.5)
        """
        B, C, N = z_content.shape
        device = z_content.device

        # --- 1. Generate Stochastic Mask ---
        # Generate binary mask via Bernoulli sampling
        rand_probs = torch.rand((B, C, N), device=device)
        mask = rand_probs < mask_ratio

        # --- 2. Prepare Input ---
        x_masked = z_content.clone()
        x_masked[mask] = self.mask_token_id

        # --- 3. Denoise ---
        # Determine starting diffusion step based on corruption intensity
        if mask_ratio == 0.5:
            start_step = self.num_timesteps // 2
        else:
            start_step = int(self.num_timesteps * mask_ratio)

        # Perform iterative denoising conditioned on observed tokens
        rec = self.fast_iterative_denoise(x_masked, adj_matrix, z_global, start_step)

        # --- 4. Final Output ---
        # Impute masked positions while preserving observed ground truth
        z_final = z_content.clone()
        z_final[mask] = rec[mask]

        return z_final