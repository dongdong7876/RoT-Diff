import torch
import torch.nn as nn
from einops import rearrange
import pytorch_lightning as pl
from model.stage1.base_model import GCN, GraphGatedFusion
from model.stage1.encoder import TSTEncoder
from utils import linear_warmup_cosine_annealingLR


class ExpStage1(pl.LightningModule):
    """
    Stage 1: Robust Representation Distillation.
    This module implements a Robust Regime-Aware VQ-VAE optimized with the Student-t distribution.
    It distills contaminated data into purified tokens (LCC) and extracts macroscopic regime priors (GCC).
    """

    def __init__(self, in_channels: int, config: dict):
        super().__init__()
        # --- Configuration ---
        self.config = config
        self.num_patch = config['transformer']['num_patch']
        self.win_size = config['dataset']['win_size']
        self.d_model = config['transformer']['stage1_d_model']
        self.in_channels = in_channels

        # VQ Parameters
        self.codebook_size = config['VQ']['codebook_size']
        self.global_codebook_size = config['VQ']['global_codebook_size']
        self.codebook_weight_decay = config['VQ']['codebook_weight_decay']
        self.kmeans_init = config['VQ']['kmeans_init']
        self.kmeans_iters = config['VQ']['kmeans_iters']
        self.threshold_ema_dead_code = config['VQ']['threshold_ema_dead_code']
        self.dof = config['VQ']['dof']

        # Decoder/Transformer Parameters
        self.n_heads = config['decoder']['n_heads']
        self.d_ff = config['decoder']['d_ff']
        self.n_layers = config['decoder']['n_layers']
        self.res_attention = config['decoder']['res_attention']
        self.patch_len = self.win_size // self.num_patch

        # --- 1. Patch Embedding (Local Content) ---
        self.patch_emb = nn.Linear(self.patch_len, self.d_model)

        # Select VQ Strategy
        vq_strategy = config['VQ']['vq_strategy']
        if vq_strategy == 'ST_VQ':
            from model.stage1.vq_student_t import VectorQuantize
            VQ = VectorQuantize
        elif vq_strategy == 'VQ':
            from model.stage1.vq import VectorQuantize
            VQ = VectorQuantize
        else:
            raise FileNotFoundError(f"No {vq_strategy} Strategy!")

        # --- 2. Local Content Codebook (LCC) ---
        # Robustly quantizes local patches to filter out point-wise anomalies.
        self.feature_vq = VQ(
            dim=self.d_model,
            codebook_dim=self.d_model,
            codebook_size=self.codebook_size,
            codebook_weight_decay=self.codebook_weight_decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            use_student_t=True,
            dof=self.dof,
        )

        self.global_aggregator = nn.Linear(self.num_patch * self.d_model, self.d_model)

        # --- 3. Structure Learning Components ---
        # Learnable embedding to capture variable-specific semantics
        self.variable_emb = nn.Parameter(torch.randn(1, self.in_channels, self.d_model))

        # Projections for deriving the adjacency matrix
        self.query_proj = nn.Linear(self.d_model, self.d_model)
        self.key_proj = nn.Linear(self.d_model, self.d_model)
        self.struct_norm = nn.LayerNorm(self.in_channels)

        # --- 4. Global Concept Codebook (GCC) ---
        # Global query to extract the macroscopic system state (Regime)
        self.global_query = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Attention mechanism for global regime extraction
        self.global_aggregator_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=4,
            batch_first=True
        )

        # Quantizes the global regime vector to obtain the regime prototype
        self.global_concept_vq = VQ(
            dim=self.d_model,
            codebook_dim=self.d_model,
            codebook_size=self.global_codebook_size,
            codebook_weight_decay=self.codebook_weight_decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            use_student_t=True,
            dof=self.dof,
        )

        # --- 5. Reconstruction & Fusion Components ---
        # GCN to refine features using the learned graph structure
        self.shared_gcn = GCN(self.d_model)
        # Gated mechanism to fuse local VQ features with structure-refined features
        self.gate = GraphGatedFusion(self.d_model)

        # Transformer Encoder for final reconstruction
        self.encoder = TSTEncoder(
            self.d_model, self.n_heads, d_ff=self.d_ff,
            dropout=config['transformer']['dropout'], res_attention=self.res_attention,
            n_layers=self.n_layers
        )

        self.time_pos_emb = nn.Parameter(torch.randn(1, self.num_patch, self.d_model))
        self.head = nn.Linear(self.d_model * self.num_patch, self.win_size)
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        """
        Forward pass for Robust Representation Distillation.
        Args:
            x: Input time series [Batch, Length, Channels]
        Returns:
            Dictionary containing reconstruction, VQ losses, and latent codes.
        """
        B, L, C = x.size()
        x = x.transpose(1, 2)  # [B, C, L]

        # --- Step 1: Patching & Embedding ---
        x_patch = rearrange(x, 'b c (n p) -> b c n p', n=self.num_patch)
        x_patch_emb = self.patch_emb(x_patch)  # [B, C, N, D]

        # --- Step 2: Content VQ (Local) ---
        # Purify local features using Student-t VQ (LCC)
        x_quantized_local, vq_ind_local, vq_loss_local, perplexity_local = self.feature_vq(x_patch_emb)

        # --- Step 3: Structure Learning ---
        # Aggregate patches to obtain variable-level representation
        x_flat = rearrange(x_quantized_local, 'b c n d -> b c (n d)')
        x_global_content = self.global_aggregator(x_flat)
        x_global_content = x_global_content + self.variable_emb

        # Learn the Adjacency Matrix (Graph Structure) via Self-Attention
        Q = self.query_proj(x_global_content)
        K = self.key_proj(x_global_content)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        adj_matrix = torch.softmax(attn_logits, dim=-1)  # [B, C, C]

        # --- Step 4: Global Regime Extraction ---
        # Attention Pooling: Extract global regime vector from all variables
        global_query_expanded = self.global_query.repeat(B, 1, 1)
        x_global_concept, _ = self.global_aggregator_attn(
            query=global_query_expanded,
            key=x_global_content,
            value=x_global_content
        )

        # Quantize the regime vector using GCC
        z_global_quantized, vq_ind_global_concept, vq_loss_global_concept, perplexity_global_concept = \
            self.global_concept_vq(x_global_concept)

        # --- Step 5: Regime-Guided Reconstruction ---
        # A. Structure Refinement (GCN)
        x_local_flat = rearrange(x_quantized_local, 'b c n d -> (b n) c d')
        adj_flat = adj_matrix.unsqueeze(1).repeat(1, self.num_patch, 1, 1).reshape(-1, C, C)

        x_graph_emb = self.shared_gcn(x_local_flat, adj_flat)
        x_graph_emb = rearrange(x_graph_emb, '(b n) c d -> b c n d', n=self.num_patch)

        # B. Gated Fusion
        x_fused, gate_score = self.gate(x_quantized_local, x_graph_emb)

        # C. Prepare Decoder Input
        x_final_input = rearrange(x_fused, 'b c n d -> (b c) n d')

        # D. Regime Injection
        # Inject the global regime prototype into the feature space
        z_global_expanded = z_global_quantized.unsqueeze(1).repeat(1, C, 1, 1)
        z_global_expanded = rearrange(z_global_expanded, 'b c 1 d -> (b c) 1 d')
        x_final_input = x_final_input + z_global_expanded

        # E. Transformer Reconstruction
        x_final_input = x_final_input + self.time_pos_emb
        rec_latent = self.encoder(x_final_input)

        # F. Final Projection to Time Series
        rec_flat = rearrange(rec_latent, '(b c) n d -> b c (n d)', c=C)
        rec = self.head(rec_flat).transpose(1, 2)  # [B, L, C]

        return {
            'rec': rec,
            'vq_loss_content': vq_loss_local['loss'],
            'perplexity_content': perplexity_local,
            'vq_loss_global_concept': vq_loss_global_concept['loss'],
            'perplexity_global_concept': perplexity_global_concept,
            'z_content': vq_ind_local,
            'z_global_concept': vq_ind_global_concept
        }

    def training_step(self, batch, batch_idx):
        self.train()
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        # --- Robustness: Dynamic DoF Annealing ---
        # Anneals the degrees of freedom for Student-t distribution during early training
        max_dof = 100.0
        min_dof = 10.0
        warmup_epochs = 10

        if self.current_epoch < warmup_epochs:
            new_dof = max_dof - (max_dof - min_dof) * (self.current_epoch / warmup_epochs)
        else:
            new_dof = min_dof

        self.feature_vq._codebook.dof = new_dof
        self.global_concept_vq._codebook.dof = new_dof

        # Forward Pass
        result_dic = self.forward(x)
        loss_rec = self.criterion(result_dic['rec'], x)
        loss_content = result_dic.get('vq_loss_content', 0)
        loss_global = result_dic.get('vq_loss_global_concept', 0)

        # Total Loss: Reconstruction + Commitment Losses
        loss = loss_rec + loss_content + loss_global

        # Logging
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/rec_loss', loss_rec, on_step=False, on_epoch=True)
        self.log('train/content_loss', loss_content, on_step=False, on_epoch=True)
        self.log('train/global_concept_loss', loss_global, on_step=False, on_epoch=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        result_dic = self.forward(x)
        loss_rec = self.criterion(result_dic['rec'], x)
        loss_content = result_dic.get('vq_loss_content', 0)
        loss_global = result_dic.get('vq_loss_global_concept', 0)

        loss = loss_rec + loss_content + loss_global

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/rec_loss', loss_rec, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['lr']['stage1'], weight_decay=1e-5)
        total_steps = self.trainer.estimated_stepping_batches
        sch = linear_warmup_cosine_annealingLR(
            optimizer=opt,
            max_steps=total_steps,
            linear_warmup_rate=self.config['exp_params']['linear_warmup_rate'],
            min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ==========================================
    # Helper Methods for Stage 2 Integration
    # ==========================================

    def encode_to_z(self, x):
        """
        Encode raw data to latent codes and extract structural priors for Stage 2.
        Returns:
            - vq_ind_local: Indices of purified local patches (LCC).
            - z_global_quantized: The quantized global regime prototype (GCC).
            - adj_matrix: The learned graph structure.
        """
        B, L, C = x.size()
        x = x.transpose(1, 2)

        # 1. Content VQ
        x_patch = rearrange(x, 'b c (n p) -> b c n p', n=self.num_patch)
        x_patch_emb = self.patch_emb(x_patch)
        x_quantized_local, vq_ind_local, _, _ = self.feature_vq(x_patch_emb)

        # 2. Structure Prep
        x_flat = rearrange(x_quantized_local, 'b c n d -> b c (n d)')
        x_global_content = self.global_aggregator(x_flat) + self.variable_emb

        # 3. Global Concept Encoding
        global_query_expanded = self.global_query.repeat(B, 1, 1)
        x_global_concept, _ = self.global_aggregator_attn(
            query=global_query_expanded, key=x_global_content, value=x_global_content
        )
        z_global_quantized, vq_ind_global_concept, _, _ = self.global_concept_vq(x_global_concept)

        # 4. Adjacency Matrix Learning
        Q = self.query_proj(x_global_content)
        K = self.key_proj(x_global_content)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        adj_matrix = torch.softmax(attn_logits, dim=-1)

        return vq_ind_local, z_global_quantized, adj_matrix

    def decode_z(self, z_indices, z_global_quantized, adj_clean):
        """
        Decode latent indices back to time series.
        Used during the inference phase of Stage 2 to reconstruct the generated tokens.

        Args:
            z_indices: Content Indices [B, C, N]
            z_global_quantized: Global Regime Prototype [B, 1, D]
            adj_clean: The learned graph structure [B, C, C]
        """
        B, C, N = z_indices.shape

        # 1. Indices -> Quantized Vectors (Local Content)
        x_quantized_local = self.feature_vq.get_codebook_entry(z_indices)

        # 2. GCN Refinement & Fusion
        x_local_flat = rearrange(x_quantized_local, 'b c n d -> (b n) c d')
        adj_flat = adj_clean.unsqueeze(1).repeat(1, self.num_patch, 1, 1).reshape(-1, C, C)

        x_graph_emb = self.shared_gcn(x_local_flat, adj_flat)
        x_graph_emb = rearrange(x_graph_emb, '(b n) c d -> b c n d', n=self.num_patch)

        x_fused, gate_score = self.gate(x_quantized_local, x_graph_emb)

        # 3. Final Reconstruction with Regime Injection
        x_final_input = rearrange(x_fused, 'b c n d -> (b c) n d')

        # Inject Global Regime
        z_global_expanded = z_global_quantized.unsqueeze(1).repeat(1, C, 1, 1)
        z_global_expanded = rearrange(z_global_expanded, 'b c 1 d -> (b c) 1 d')
        x_final_input = x_final_input + z_global_expanded

        # Transformer Encoding
        x_final_input = x_final_input + self.time_pos_emb
        rec_latent = self.encoder(x_final_input)

        rec_flat = rearrange(rec_latent, '(b c) n d -> b c (n d)', c=C)
        rec = self.head(rec_flat).transpose(1, 2)

        return rec

    # ==========================================
    # Helper Methods for Evaluation
    # ==========================================

    @torch.no_grad()
    def get_stage1_anomaly_score(self, x):
        """
        Compute the Distillation Score (Local Anomaly Score).
        Calculates the quantization error between the input patches and their matched prototypes.

        Output:
            score: Point-wise anomaly score [B, Length, Channels]
        """
        B, L, C = x.size()
        x = x.transpose(1, 2)

        # 1. Patching & Embedding
        x_patch = rearrange(x, 'b c (n p) -> b c n p', n=self.num_patch)
        x_patch_emb = self.patch_emb(x_patch)

        # 2. Content VQ
        x_quantized_local, _, _, _ = self.feature_vq(x_patch_emb)

        # 3. Compute MSE (Quantization Error)
        diff = (x_patch_emb - x_quantized_local).pow(2)
        score_patch = diff.sum(dim=-1)

        # 4. Upsample to point-level resolution
        score_point = score_patch.unsqueeze(-1).repeat(1, 1, 1, self.patch_len)
        score_point = rearrange(score_point, 'b c n p -> b (n p) c')

        # Transpose back to [B, L, C]
        score_point = score_point.transpose(1, 2)

        return score_point