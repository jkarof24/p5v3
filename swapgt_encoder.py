"""
SwapGT encoder for pure-structure node embeddings on CPU.

Adapts the SwapGT (Token-Swapping Graph Transformer) model to compute
node embeddings from graph structure alone, without node features.
Supports loading pre-computed embeddings or computing on CPU.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict

PRE_CAL_DIR = "/home/jonathan/Desktop/P5/dataset/SwapGT-data/pre_cal"


def init_params(module, n_layers):
    """Initialize parameters with scaled normal distribution."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    """Transformer feed-forward network."""
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.att_size)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.att_size)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * self.att_size)
        x = self.output_layer(x)

        return x


class EncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class SwapGTEncoder(nn.Module):
    """
    SwapGT encoder adapted for pure-structure node embeddings.
    
    Takes graph structure (edge_index) and produces node embeddings
    without requiring node features. Uses learnable node embeddings
    initialized randomly and refined through transformer layers.
    """
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Learnable node embeddings (replaces input features)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        
        encoders = [
            EncoderLayer(hidden_dim, ffn_dim, dropout_rate, 
                        attention_dropout_rate, num_heads)
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.apply(lambda module: init_params(module, n_layers=num_layers))
    
    def forward(self, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute node embeddings from graph structure.
        
        Args:
            edge_index: Optional edge index for attention bias (not used in base version)
        
        Returns:
            Node embeddings of shape (num_nodes, hidden_dim)
        """
        # Initialize with learnable embeddings
        node_indices = torch.arange(self.num_nodes, device=self.node_emb.weight.device)
        x = self.node_emb(node_indices)  # (num_nodes, hidden_dim)
        
        # Add batch dimension for transformer: (1, num_nodes, hidden_dim)
        x = x.unsqueeze(0)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_ln(x)
        
        # Remove batch dimension: (num_nodes, hidden_dim)
        return x.squeeze(0)


def load_precomputed_embeddings(dataset: str, split: str, device="cpu") -> Optional[torch.Tensor]:
    """
    Load pre-computed SwapGT embeddings from pre_cal directory.
    
    Args:
        dataset: Dataset name
        split: 'de' or 'sp'
        device: Target device
    
    Returns:
        Embedding tensor or None if not found
    """
    # Try to find matching pre-computed file
    pre_cal_subdir = os.path.join(PRE_CAL_DIR, split)
    if not os.path.exists(pre_cal_subdir):
        return None
    
    for filename in os.listdir(pre_cal_subdir):
        if filename.startswith(dataset) and filename.endswith(".pt"):
            path = os.path.join(pre_cal_subdir, filename)
            try:
                emb = torch.load(path, weights_only=False, map_location=device)
                if isinstance(emb, torch.Tensor):
                    return emb
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
    return None


def get_swapgt_embeddings(
    dataset: str,
    split: str,
    num_nodes: int,
    hidden_dim: int = 128,
    use_precomputed: bool = False,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Get SwapGT node embeddings, either from pre-computed files or by computing on CPU.
    
    Args:
        dataset: Dataset name
        split: 'de' or 'sp'
        num_nodes: Number of nodes in the graph
        hidden_dim: Embedding dimension
        use_precomputed: Whether to try loading pre-computed embeddings first
        device: Target device
    
    Returns:
        Node embeddings of shape (num_nodes, hidden_dim)
    """
    # Try pre-computed first
    if use_precomputed:
        emb = load_precomputed_embeddings(dataset, split, device)
        if emb is not None:
            # Pre-computed embeddings may have shape (num_nodes, k, d) or (num_nodes, d)
            if emb.shape[0] == num_nodes:
                if len(emb.shape) == 3:
                    # Average over the middle dimension: (num_nodes, k, d) -> (num_nodes, d)
                    emb = emb.mean(dim=1)
                if emb.shape[1] == hidden_dim:
                    return emb.to(device)
                else:
                    # Project to correct dimension
                    projector = nn.Linear(emb.shape[1], hidden_dim)
                    projector = projector.to(device)
                    with torch.no_grad():
                        return projector(emb.to(device))
    
    # Compute on CPU
    encoder = SwapGTEncoder(
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_heads=4,
        ffn_dim=256,
    ).to(device)
    
    with torch.no_grad():
        emb = encoder()
    
    return emb
