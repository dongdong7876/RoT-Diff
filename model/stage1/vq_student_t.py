import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.cuda.amp import autocast
from einops import rearrange, repeat
from torch.distributions.categorical import Categorical
from typing import Union


# ==========================================
# 1. Helper Functions
# ==========================================

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    """
    In-place Exponential Moving Average (EMA) update.
    used for updating codebook prototypes without gradients.
    """
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    """
    Applies Laplace smoothing to cluster size counts to prevent division by zero.
    """
    return (x + eps) / (x.sum() + n_categories * eps)


def orthgonal_loss_fn(t):
    """
    Computes orthogonal regularization loss to encourage diverse codebook usage.
    Eq (2) from https://arxiv.org/abs/2112.00384
    """
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = einsum('i d, j d -> i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


def sample_vectors(samples, num):
    """
    Randomly samples 'num' vectors from the input batch.
    Used for initializing or re-initializing codebook vectors.
    """
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    """
    Runs k-means clustering to initialize the codebook prototypes using the
    distribution of the first batch of data.
    """
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    means = sample_vectors(samples, num_clusters)
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            means_sq = means.pow(2).sum(dim=1).unsqueeze(0)
            dot_prod = samples @ means.t()
            dists = 2 * dot_prod - means_sq
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        if use_cosine_sim:
            new_means = l2norm(new_means)
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


def softmax_sample(t, temperature, dim=-1):
    """
    Samples from a categorical distribution parameterized by logits 't'.
    Used during training if stochastic quantization is enabled.
    """
    if temperature == 0 or temperature is None:
        return t.argmax(dim=dim)
    m = Categorical(logits=t / temperature)
    return m.sample()


# ==========================================
# 2. Robust Student-t Codebook (Core Contribution)
# ==========================================

class StudentTCodebook(nn.Module):
    """
    Implements the Robust Student-t VQ mechanism described in Section 3.2.1.

    Unlike standard VQ-VAE which assumes Gaussian likelihoods (Euclidean distance),
    this module uses the Student-t distribution to calculate heavy-tailed similarity.
    This effectively saturates the cost for distant outliers, mitigating 'prototype pollution'
    from contaminated training data.
    """

    def __init__(
            self,
            dim,
            codebook_size,
            dof=1.0,  # Degrees of Freedom (nu), controls heavy-tailedness
            kmeans_init=False,
            kmeans_iters=10,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0.,
            emb_dropout=0.
    ):
        super().__init__()
        self.decay = decay
        self.dof = dof

        # Initialization strategy
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        self.emb_dropout = emb_dropout

        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        # Buffers for EMA updates
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))  # Sum of weights
        self.register_buffer('cluster_cnt', torch.zeros(codebook_size))  # Raw counts for dead code detection
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        self.embed_onehot = None
        self.perplexity = None

    @torch.jit.ignore
    def init_embed_(self, data):
        """Initialize codebook using k-means on the first batch."""
        if self.initted:
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.cluster_cnt.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        """Replace dead codes with random samples from the current batch."""
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

        # Reset statistics for replaced codes
        self.cluster_size.data.masked_fill_(mask, 1.0)
        self.cluster_cnt.data.masked_fill_(mask, self.threshold_ema_dead_code + 1)
        self.embed_avg.data.masked_scatter_(mask.unsqueeze(1), modified_codebook[mask])

    def expire_codes_(self, batch_samples):
        """
        Detects and replaces 'dead' codes (prototypes that are rarely used).
        Uses raw assignment counts (cluster_cnt) rather than weighted sums to ensure accuracy.
        """
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_cnt < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x, svq_temp=None):
        """
        Forward pass for Robust VQ.
        1. Compute Heavy-tailed Similarity (Student-t log-likelihood).
        2. Assign codes based on similarity.
        3. Perform Contamination-Resilient Weighted Update (M-estimator).
        """
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = embed.t()

        if self.emb_dropout and self.training:
            embed = F.dropout(embed, self.emb_dropout)

        # 1. Calculate Squared Euclidean Distance
        # ||h_i - z_k||^2
        flatten_sq = flatten.pow(2).sum(1, keepdim=True)
        embed_sq = embed.pow(2).sum(0, keepdim=True)
        dot_prod = flatten @ embed
        dist_sq = flatten_sq + embed_sq - 2 * dot_prod
        dist_sq = dist_sq / x.shape[-1]
        dist_sq = dist_sq.clamp(min=0.0)

        # 2. Calculate Student-t Log-Likelihood (Heavy-tailed Similarity)
        # Eq (4): phi(h_i, z_k) = -0.5 * (v + 1) * log(1 + d^2 / v)
        # This metric saturates for large distances, filtering out outliers.
        t_logits = -0.5 * (self.dof + 1) * torch.log(1 + dist_sq / self.dof)

        # 3. Code Assignment
        temp = svq_temp if svq_temp is not None else self.sample_codebook_temp
        if self.training and temp > 0:
            embed_ind = softmax_sample(t_logits, dim=-1, temperature=temp)
        else:
            embed_ind = t_logits.argmax(dim=-1)

        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        # 4. Contamination-Resilient Weighted Update
        if self.training:
            # Calculate Influence Weights
            # Eq (5): w_i = (v + 1) / (v + ||h_i - z_k||^2)
            # Weights decay rapidly for outliers, preventing them from shifting prototypes.
            chosen_dist_sq = (dist_sq * embed_onehot).sum(dim=1, keepdim=True)
            weights = (self.dof + 1) / (self.dof + chosen_dist_sq)

            # Apply weights to assignments
            embed_onehot_weighted = embed_onehot * weights

            # Update Cluster Size (Weighted Sum)
            cluster_size_weighted = embed_onehot_weighted.sum(0)
            self.all_reduce_fn(cluster_size_weighted)
            ema_inplace(self.cluster_size, cluster_size_weighted, self.decay)

            # Update Embeddings (Weighted Average of Inputs)
            # z_k = EMA( sum(w_i * h_i) / sum(w_i) )
            embed_sum = (flatten * weights).t() @ embed_onehot
            self.all_reduce_fn(embed_sum)
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)

            # Normalize embedding position
            cluster_size_smoothed = laplace_smoothing(self.cluster_size, self.codebook_size,
                                                      self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size_smoothed.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

            # Track raw usage for dead code expiration
            usage_cnt = embed_onehot.sum(0)
            self.all_reduce_fn(usage_cnt)
            ema_inplace(self.cluster_cnt, usage_cnt, self.decay)
            self.expire_codes_(x)

        # Calculate Perplexity (Codebook Utilization)
        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        self.embed_onehot = embed_onehot.detach()
        self.perplexity = perplexity.detach()

        return quantize, embed_ind


# ==========================================
# 3. Standard Euclidean Codebook (Baseline)
# ==========================================

class EuclideanCodebook(nn.Module):
    """
    Standard VQ implementation based on Euclidean distance.
    Assumes Gaussian likelihoods, which makes it sensitive to outliers.
    Reference: https://github.com/lucidrains/vector-quantize-pytorch
    """

    def __init__(
            self,
            dim,
            codebook_size,
            kmeans_init=False,
            kmeans_iters=10,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0.,
            emb_dropout=0.,
    ):
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        self.emb_dropout = emb_dropout

        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        self.embed_onehot = None
        self.perplexity = None

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x, svq_temp: Union[float, None] = None):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = embed.t()

        if self.emb_dropout and self.training:
            embed = F.dropout(embed, self.emb_dropout)

        # Squared Euclidean Distance calculation
        embed_sq = embed.pow(2).sum(0, keepdim=True)
        dot_prod = flatten @ embed
        dist = 2 * dot_prod - embed_sq

        temp = svq_temp
        # Softmax sampling or Argmax
        embed_ind = softmax_sample(dist, dim=-1, temperature=temp)

        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            # Standard EMA Update (Mean-based)
            # Note: This is sensitive to outliers as large deviations shift the mean significantly.
            cluster_size = embed_onehot.sum(0)
            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size, cluster_size, self.decay)

            embed_sum = flatten.t() @ embed_onehot
            self.all_reduce_fn(embed_sum)
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        self.embed_onehot = embed_onehot.detach()
        self.perplexity = perplexity.detach()

        return quantize, embed_ind


# ==========================================
# 4. VectorQuantize Wrapper
# ==========================================

class VectorQuantize(nn.Module):
    """
    Main wrapper module for Vector Quantization.
    Supports both Standard Euclidean VQ and Robust Student-t VQ.
    """

    def __init__(
            self,
            dim,
            codebook_size,
            codebook_dim=None,
            heads=1,
            decay=0.8,
            eps=1e-5,
            kmeans_init=False,
            kmeans_iters=10,
            use_cosine_sim=False,
            threshold_ema_dead_code=0,
            channel_last=True,
            accept_image_fmap=False,
            commitment_weight=1.,
            orthogonal_reg_weight=0.,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=0.,
            sync_codebook=False,
            emb_dropout=0.,
            pcl_temperature=None,
            use_student_t=True,
            dof=1.0,
            **kwargs
    ):
        super().__init__()
        self.heads = heads
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight

        # Regularization Configuration
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        # Initialize Codebook Strategy
        if use_student_t:
            print(f"Initializing Robust Student-t Codebook with DoF={dof}")
            self._codebook = StudentTCodebook(
                dim=codebook_dim,
                codebook_size=codebook_size,
                dof=dof,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                decay=decay,
                eps=eps,
                threshold_ema_dead_code=threshold_ema_dead_code,
                use_ddp=sync_codebook,
                learnable_codebook=has_codebook_orthogonal_loss,
                sample_codebook_temp=sample_codebook_temp,
                emb_dropout=emb_dropout,
            )
        else:
            print("Initializing Standard Euclidean Codebook")
            self._codebook = EuclideanCodebook(
                dim=codebook_dim,
                codebook_size=codebook_size,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                decay=decay,
                eps=eps,
                threshold_ema_dead_code=threshold_ema_dead_code,
                use_ddp=sync_codebook,
                learnable_codebook=has_codebook_orthogonal_loss,
                sample_codebook_temp=sample_codebook_temp,
                emb_dropout=emb_dropout,
            )

        self.codebook_size = codebook_size
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        return self._codebook.embed

    def forward(self, x, svq_temp: Union[float, None] = None):
        """
        Args:
            x: Input tensor (B, N, D) or (B, C, H, W)
        Returns:
            quantize: Quantized embeddings (B, N, D)
            embed_ind: Code indices (B, N)
            vq_loss: Dictionary containing commitment and regularization losses
            perplexity: Codebook utilization metric
        """
        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size
        need_transpose = not self.channel_last and not self.accept_image_fmap
        vq_loss = {
            'loss': torch.tensor([0.], device=device, requires_grad=self.training),
            'commit_loss': 0.,
            'orthogonal_reg_loss': 0.,
        }

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)

        if is_multiheaded:
            x = rearrange(x, 'b n (h d) -> (b h) n d', h=heads)

        # Codebook Lookup
        quantize, embed_ind = self._codebook(x, svq_temp)

        # Straight-Through Estimator (STE)
        if self.training:
            quantize = x + (quantize - x).detach()

        # Loss Calculation
        if self.training:
            # Commitment Loss: Constrain encoder outputs to stay close to codebook
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                vq_loss['commit_loss'] = commit_loss
                vq_loss['loss'] = vq_loss['loss'] + commit_loss * self.commitment_weight

            # Orthogonal Regularization: Encourage codebook diversity
            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook
                if self.orthogonal_reg_active_codes_only:
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthgonal_loss_fn(codebook)
                vq_loss['orthogonal_reg_loss'] = orthogonal_reg_loss
                vq_loss['loss'] = vq_loss['loss'] + orthogonal_reg_loss * self.orthogonal_reg_weight

        if is_multiheaded:
            quantize = rearrange(quantize, '(b h) n d -> b n (h d)', h=heads)
            embed_ind = rearrange(embed_ind, '(b h) n -> b n h', h=heads)

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)

        return quantize, embed_ind, vq_loss, self._codebook.perplexity

    def get_codebook_entry(self, indices, *kwargs):
        """
        Retrieve codebook vectors based on indices.
        Used for reconstruction in Stage 2.
        """
        z_q = F.embedding(indices, self._codebook.embed)

        if self.heads > 1:
            z_q = rearrange(z_q, '... h d -> ... (h d)')

        z_q = self.project_out(z_q)
        return z_q


if __name__ == '__main__':
    # Simple sanity check
    torch.manual_seed(0)
    B, N, D = 1024, 32, 128
    x = torch.rand((B, N, D))

    vq = VectorQuantize(dim=D, codebook_size=512, use_student_t=True, dof=1.0)

    quantize, vq_ind, vq_loss, perplexity = vq(x)
    print(f"Sanity Check Passed.")
    print(f"Index shape: {vq_ind.shape}")
    print(f"Codebook shape: {vq.codebook.shape}")