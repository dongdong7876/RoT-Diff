import os
import glob

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_factory.data_loader_contamination import get_loader_segment
from experiments.exp_stage2 import ExpStage2
from evaluation.evaluator import Evaluator


# ==========================================
# 1. Hybrid Score Calculation (Shift-and-Average)
# ==========================================

def get_raw_hybrid_scores(model, loader, device):
    """
    Computes the Hybrid Anomaly Score using the Shift-and-Average inference protocol.

    This function implements the inference phase described in the paper:
    1. Generates multiple shifted views of the input time series to eliminate window-boundary artifacts.
    2. Calculates two complementary anomaly metrics for each view:
       - Distillation Score (Stage 1): Quantization error reflecting local consistency.
       - Imputation Score (Stage 2): Reconstruction error reflecting global context consistency.
    3. Aggregates scores across all shifted views to produce a smoothed, robust anomaly score.

    Args:
        model: The trained ExpStage2 model containing both Stage 1 and Stage 2 modules.
        loader: DataLoader for the dataset.
        device: Calculation device (CPU/GPU).

    Returns:
        scores_s1: Aggregated Distillation Scores (Stage 1).
        scores_s2: Aggregated Imputation Scores (Stage 2).
        labels: Ground truth anomaly labels.
    """
    model.eval()

    scores_s1 = []
    scores_s2 = []
    labels = []

    # Get patch length to determine the number of shifts required for full coverage
    patch_len = model.stage1_model.patch_len

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x = batch
                y = None

            x = x.to(device)
            B, L, C = x.shape

            # Accumulators for Shift-and-Average (Test-Time Augmentation)
            accum_s1 = torch.zeros((B, L), device=device)
            accum_s2 = torch.zeros((B, L), device=device)

            # --- Shift Loop: Iterate through all temporal phases ---
            # We cyclically roll the input to generate P shifted views.
            # This ensures every time point is evaluated at every position within a patch,
            # eliminating artifacts caused by fixed patch boundaries.
            for shift_i in range(patch_len):

                # 1. Roll input (Circular shift)
                x_shifted = torch.roll(x, shifts=-shift_i, dims=1)

                # ==========================
                # Metric 1: Distillation Score (Stage 1)
                # ==========================
                # Measures the quantization error: || z - q(z) ||^2.
                # High error indicates the local pattern does not match any robust prototype.
                s1_batch_shifted = model.stage1_model.get_stage1_anomaly_score(x_shifted)

                # Collapse feature dimension if necessary -> [B, L]
                if s1_batch_shifted.dim() == 3:
                    s1_batch_shifted = s1_batch_shifted.mean(dim=1)

                # ==========================
                # Metric 2: Imputation Score (Stage 2)
                # ==========================
                # Measures contextual inconsistency via masked reconstruction.

                # A. Distill Representations (S_0, g_hat, A)
                # Extract purified tokens, global regime, and graph structure.
                vq_ind_local, z_global_vec, adj_matrix = model.stage1_model.encode_to_z(x_shifted)

                # B. Masked Imputation
                # Reconstruct masked tokens conditioned on observed context and global regime.
                z_recon = model.diffusion_model.reconstruct_random_masking(
                        vq_ind_local, adj_matrix, z_global_vec
                    )

                # C. Decode to Time Series
                x_recon = model.stage1_model.decode_z(z_recon, z_global_vec, adj_matrix)

                # D. Calculate Reconstruction Error (Contextual Anomaly Score)
                mse_loss = (x_shifted - x_recon) ** 2
                score_s2_batch_shifted = mse_loss.mean(dim=-1)

                # ==========================
                # Accumulate & Realign
                # ==========================
                # Roll scores back to align with the original time steps
                s1_aligned = torch.roll(s1_batch_shifted, shifts=shift_i, dims=1)
                s2_aligned = torch.roll(score_s2_batch_shifted, shifts=shift_i, dims=1)

                accum_s1 += s1_aligned
                accum_s2 += s2_aligned

            # Average over all shifted views to obtain the final smoothed score
            avg_s1 = accum_s1 / patch_len
            avg_s2 = accum_s2 / patch_len

            # --- Collect Batch Results ---
            scores_s1.append(avg_s1.cpu().numpy())
            scores_s2.append(avg_s2.cpu().numpy())

            if y is not None:
                labels.append(y.cpu().numpy())

    scores_s1 = np.concatenate(scores_s1).reshape(-1)
    scores_s2 = np.concatenate(scores_s2).reshape(-1)

    if len(labels) > 0:
        labels = np.concatenate(labels).reshape(-1)
    else:
        labels = None

    return scores_s1, scores_s2, labels


# ==========================================
# 2. Evaluation Pipeline
# ==========================================

def evaluate_fn(args, config, batch_size, in_channels, num_workers, win_size, train_split, anomaly_ratio):
    """
    Main evaluation function.
    Loads the trained model, computes scores, applies normalization, and reports metrics.
    """
    data_name = args.data_name
    data_path = args.data_path
    save_dir = config['trainer_params']['save_dir']
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 1. Load Model Checkpoint
    # Loads the best Stage 2 model, which automatically includes the frozen Stage 1 model.
    search_pattern = os.path.join(save_dir, args.data_name, 'stage2-best-*.ckpt')
    ckpt_files = glob.glob(search_pattern)
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found at {search_pattern}")
    ckpt_files.sort(key=os.path.getmtime)
    ckpt_path = ckpt_files[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    model = ExpStage2.load_from_checkpoint(ckpt_path, in_channels=in_channels, data_name=args.data_name,
                                           config=config, strict=False)
    model.to(device)
    model.eval()

    # 2. Prepare DataLoaders
    # Training data is loaded to compute normalization statistics and thresholds.
    train_loader = load_data(data_name, data_path, win_size, train_split, num_workers, batch_size, mode='threshold')
    test_loader = load_data(data_name, data_path, win_size, train_split, num_workers, batch_size, mode='test')

    print("====================== START INFERENCE ======================")

    # 3. Compute Scores for Training Data (for Normalization)
    print("Computing Training Scores...")
    train_s1, train_s2, _ = get_raw_hybrid_scores(model, train_loader, device)

    # 4. Compute Scores for Test Data
    print("Computing Test Scores...")
    test_s1, test_s2, test_labels = get_raw_hybrid_scores(model, test_loader, device)

    # 5. Hybrid Scoring Aggregation
    # As described in the Method section, we combine the Distillation Score (S1) and
    # Imputation Score (S2) to amplify anomalies that violate both local and global consistency.
    print("Normalizing and Fusing Scores...")
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    # Fit scalers on training data to prevent data leakage
    scaler1.fit(train_s1.reshape(-1, 1))
    scaler2.fit(train_s2.reshape(-1, 1))

    # Normalize scores to [0, 1] range
    train_s1_norm = scaler1.transform(train_s1.reshape(-1, 1)).flatten()
    train_s2_norm = scaler2.transform(train_s2.reshape(-1, 1)).flatten()

    test_s1_norm = scaler1.transform(test_s1.reshape(-1, 1)).flatten()
    test_s2_norm = scaler2.transform(test_s2.reshape(-1, 1)).flatten()

    # Score Fusion: Element-wise product
    # S_final = Norm(S_dist) * Norm(S_imp)
    train_scores = train_s1_norm * train_s2_norm
    test_scores = test_s1_norm * test_s2_norm

    # 6. Determine Threshold
    # We use the Percentile method on training scores, assuming an anomaly ratio.
    print("Calculating threshold using Training Data Percentile...")
    threshold = np.percentile(train_scores, 100 - anomaly_ratio)
    print(f"Determined Threshold: {threshold:.6f}")

    # 7. Metrics Calculation
    # Convert raw probabilities to binary predictions
    print("Calculating Evaluation Metrics...")
    pred_raw = (test_scores > threshold).astype(int)

    # VUS and Range-AUC metrics (Threshold-agnostic)
    metrics_score = ["R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR", "auc_roc"]
    evaluator_score = Evaluator(metrics_score)
    results_score = evaluator_score.evaluate(test_labels, test_scores)

    # Point-wise metrics (Threshold-dependent)
    metrics_label = ["affiliation_f", "accuracy", "precision", "recall", "f_score",
                     "affiliation_precision", "affiliation_recall",
                     "adjust_precision", "adjust_recall", "adjust_f_score"]

    evaluator_label = Evaluator(metrics_label)
    results_label = evaluator_label.evaluate(test_labels, pred_raw)

    # Aggregate and Print Results
    metrics_label.extend(metrics_score)
    results_label.extend(results_score)
    results_np = np.array(results_label).reshape(1, -1).round(4)
    results_pd = pd.DataFrame(data=results_np, columns=metrics_label)

    for key, value in results_pd.items():
        print('{} : {}'.format(key, value.values[0]))

    return results_pd


def load_data(data_name, data_path, win_size, train_split, num_workers, batch_size, mode: str):
    """
    Helper to load data segments.
    """
    loader = get_loader_segment(data_path=data_path,
                                batch_size=batch_size,
                                win_size=win_size,
                                train_split=train_split,
                                mode=mode,
                                num_workers=num_workers,
                                data_name=data_name)
    return loader