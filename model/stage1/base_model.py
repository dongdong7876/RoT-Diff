import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    """
    Standard Graph Convolutional Layer (GCN).
    Performs the operation: H = \sigma(A * X * W) + b

    This layer facilitates message passing between variables based on the
    learned adjacency matrix, refining local representations with global structural context.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        Forward pass for graph convolution.

        Args:
            x: Node features [Batch, N_nodes, In_dim]
            adj: Adjacency matrix [Batch, N_nodes, N_nodes] (or [N_nodes, N_nodes])

        Returns:
            output: Updated node features [Batch, N_nodes, Out_dim]
        """
        # 1. Feature Transformation (X * W)
        # x: [B, N, In], weight: [In, Out] -> support: [B, N, Out]
        support = torch.matmul(x, self.weight)

        # 2. Message Passing (A * Support)
        # adj: [B, N, N], support: [B, N, Out] -> output: [B, N, Out]
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) Module.

    Implements the 'Structure Refinement' component of Stage 1.
    It applies two layers of Graph Convolution with residual connections to
    explicitly incorporate inter-variable dependencies into the latent features.
    """

    def __init__(self, d_model, dropout=0.3):
        """
        Args:
            d_model: Hidden dimension of the features
            dropout: Dropout rate for regularization
        """
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(d_model, d_model)
        self.gc2 = GraphConvolution(d_model, d_model)
        self.dropout = dropout
        self.act = nn.GELU()

    def forward(self, x, adj=None):
        """
        Args:
            x: Input features [Batch, N_nodes, d_model]
            adj: Learned Adjacency Matrix [Batch, N_nodes, N_nodes]
        """
        # Residual connection
        shortcut = x

        # 1. First GCN Layer
        x = self.gc1(x, adj)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # 2. Second GCN Layer
        x = self.gc2(x, adj)

        # 3. Residual Addition
        # Ensures that structural information refines rather than replaces local features
        return x + shortcut


class GraphGatedFusion(nn.Module):
    """
    Gated Fusion Mechanism.

    Dynamically integrates the purified local content features (from LCC)
    with the structure-refined features (from GCN).

    Formula:
        Gamma = Sigmoid(W_gate * [H_local, H_graph])
        H_fuse = W_out * (Gamma * H_local + (1 - Gamma) * H_graph)
    """

    def __init__(self, dim):
        super().__init__()
        # Computes the gating coefficient Gamma
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        # Initialize bias to 0.0 to start with a balanced fusion
        nn.init.constant_(self.gate_net[0].bias, 0.0)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_local, x_graph):
        """
        Args:
            x_local: Quantized local content features [B, C, N, D]
            x_graph: Features aggregated via the learned graph [B, C, N, D]

        Returns:
            out: Fused representation [B, C, N, D]
            gate: The gating scores (for visualization/analysis) [B, C, N, D]
        """
        # Concatenate along the feature dimension
        combined = torch.cat([x_local, x_graph], dim=-1)

        # Compute gating weight: 1.0 (local dominant) <-> 0.0 (structure dominant)
        gate = self.gate_net(combined)

        # Weighted fusion
        out = gate * x_local + (1 - gate) * x_graph

        return self.proj(out), gate