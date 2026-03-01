import os
import glob
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from experiments.exp_stage1 import ExpStage1
from model.stage2.diffusion import MTSVQDiffusion
from utils import linear_warmup_cosine_annealingLR


class ExpStage2(pl.LightningModule):
    """
    Stage 2: Regime-Adaptive Conditional Generation.

    This module trains the Regime-Adaptive Discrete Diffusion Model. It leverages the
    pre-trained and frozen Stage 1 model to extract distilled discrete tokens (S_0),
    global regime prototypes (g_hat), and the learned graph structure (A).
    The diffusion model learns to reconstruct S_0 conditioned on these semantic priors.
    """

    def __init__(self, in_channels, data_name, config):
        super().__init__()
        self.config = config
        self.in_channels = in_channels

        # --- Checkpoint Management ---
        # Automatically locate the best checkpoint from Stage 1 to serve as the tokenizer
        save_dir = config['trainer_params']['save_dir']
        search_pattern = os.path.join(save_dir, data_name, 'stage1-best-*.ckpt')
        ckpt_files = glob.glob(search_pattern)

        if len(ckpt_files) > 0:
            stage1_ckpt_path = ckpt_files[0]
            print(f"Found Stage 1 checkpoint: {stage1_ckpt_path}")
            config['stage1_ckpt_path'] = stage1_ckpt_path

        # ==========================================
        # 1. Load and Freeze Stage 1 Model (Distiller)
        # ==========================================
        # Stage 1 acts as the fixed tokenizer and structure learner.
        stage1_ckpt_path = config.get('stage1_ckpt_path')
        print(f"Loading Stage 1 model from {stage1_ckpt_path}")

        self.stage1_model = ExpStage1.load_from_checkpoint(
            stage1_ckpt_path,
            in_channels=in_channels,
            config=config
        )

        # Freeze Stage 1 parameters to prevent gradient updates during Stage 2 training
        self.stage1_model.eval()
        for param in self.stage1_model.parameters():
            param.requires_grad = False

        # ==========================================
        # 2. Initialize Stage 2 Model (Generator)
        # ==========================================
        # Regime-Adaptive Discrete Diffusion Model
        self.diffusion_model = MTSVQDiffusion(in_channels, config)

        self.test_step_outputs = []

    def forward(self, x):
        """
        Forward definition reserved for inference.
        """
        pass

    def training_step(self, batch, batch_idx):
        """
        Execute one training step for the diffusion model.
        """
        # 1. Unpack batch
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        # 2. Prior Extraction (via Frozen Stage 1)
        # We extract the ground truth tokens (S_0) and condition priors (Regime & Graph)
        with torch.no_grad():
            self.stage1_model.eval()

            # encode_to_z returns:
            # - vq_ind_local: The purified discrete tokens S_0 (LCC indices)
            # - z_global_vec: The quantized regime prototype g_hat (GCC vector)
            # - adj_matrix: The learned graph structure A
            vq_ind_local, z_global_vec, adj_matrix = self.stage1_model.encode_to_z(x)

        # 3. Diffusion Forward Pass & Loss Calculation
        # The diffusion model predicts the clean tokens S_0 from the corrupted state S_m,
        # conditioned on the global regime and graph structure.
        loss, logits = self.diffusion_model(vq_ind_local, adj_matrix, z_global_vec)

        # 4. Logging
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validate the diffusion model's reconstruction capability.
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        with torch.no_grad():
            self.stage1_model.eval()

            # Extract ground truth and conditions
            vq_ind_local, z_global_vec, adj_matrix = self.stage1_model.encode_to_z(x)

            # Calculate validation loss (Cross-Entropy)
            loss, logits = self.diffusion_model(vq_ind_local, adj_matrix, z_global_vec)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        """
        Aggregate test metrics at the end of the epoch.
        """
        outputs = self.test_step_outputs

        if not outputs:
            return

        # Aggregate scores and calculate AUC/F1
        all_scores = torch.cat([x['score'] for x in outputs], dim=0).reshape(-1)

        if outputs[0].get('label') is not None:
            all_labels = torch.cat([x['label'] for x in outputs], dim=0).reshape(-1)

            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels.cpu().numpy(), all_scores.cpu().numpy())
            self.log('test/auc', auc)
            print(f"Test AUC: {auc:.4f}")

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler for Stage 2.
        Only the parameters of the diffusion model are optimized.
        """
        # 1. Optimize only diffusion_model, keep stage1_model frozen
        params = list(self.diffusion_model.parameters())

        opt = AdamW(params, lr=self.config['exp_params']['lr']['stage2'])

        # 2. Scheduler (Cosine Annealing with Linear Warmup)
        total_steps = self.trainer.estimated_stepping_batches
        sch = linear_warmup_cosine_annealingLR(
            opt,
            max_steps=total_steps,
            linear_warmup_rate=0.05,
            min_lr=1e-6
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1
            }
        }