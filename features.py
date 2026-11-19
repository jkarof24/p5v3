"""
Unsupervised graph feature computation for pure-structure node classification.

Computes node-level structural features: degree, clustering coefficient, PageRank,
neighbor degree statistics, and SP-GNN position/structure features.
All operations are CPU-only.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np
from torch_geometric.utils import degree as pyg_degree
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import networkx as nx


def compute_degree(edge_index, num_nodes, dtype=torch.float):
    """Compute node degrees."""
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, 1), dtype=dtype)
    row, _ = edge_index
    deg = pyg_degree(row, num_nodes=num_nodes, dtype=dtype)
    return deg.unsqueeze(1)


def compute_clustering_coefficient(edge_index, num_nodes):
    """Compute local clustering coefficient for each node."""
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, 1), dtype=torch.float)
    
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(data, to_undirected=True)
    g.add_nodes_from(range(num_nodes))
    
    clustering_dict = nx.clustering(g)
    clustering_list = [clustering_dict.get(i, 0.0) for i in range(num_nodes)]
    return torch.tensor(clustering_list, dtype=torch.float).unsqueeze(1)


def compute_pagerank(edge_index, num_nodes, alpha=0.85):
    """Compute PageRank centrality."""
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, 1), dtype=torch.float)
    
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(data, to_undirected=False)
    g.add_nodes_from(range(num_nodes))
    
    try:
        pr_dict = nx.pagerank(g, alpha=alpha)
    except Exception:
        pr_dict = {i: 1.0 / num_nodes for i in range(num_nodes)}
    
    pr_list = [pr_dict.get(i, 0.0) for i in range(num_nodes)]
    return torch.tensor(pr_list, dtype=torch.float).unsqueeze(1)


def compute_neighbor_degree_stats(edge_index, num_nodes):
    """
    Compute statistics of neighbor degrees for each node:
    - mean neighbor degree
    - max neighbor degree
    - std neighbor degree (0 if < 2 neighbors)
    """
    if edge_index is None or edge_index.numel() == 0:
        zeros = torch.zeros((num_nodes, 1), dtype=torch.float)
        return zeros, zeros, zeros
    
    row, col = edge_index
    node_deg = pyg_degree(row, num_nodes=num_nodes, dtype=torch.float)
    
    # Build adjacency for neighbor lookup
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(data, to_undirected=True)
    g.add_nodes_from(range(num_nodes))
    
    mean_deg = []
    max_deg = []
    std_deg = []
    
    for i in range(num_nodes):
        neighbors = list(g.neighbors(i))
        if not neighbors:
            mean_deg.append(0.0)
            max_deg.append(0.0)
            std_deg.append(0.0)
        else:
            nbr_degs = [node_deg[n].item() for n in neighbors]
            mean_deg.append(np.mean(nbr_degs))
            max_deg.append(np.max(nbr_degs))
            if len(nbr_degs) >= 2:
                std_deg.append(np.std(nbr_degs))
            else:
                std_deg.append(0.0)
    
    return (
        torch.tensor(mean_deg, dtype=torch.float).unsqueeze(1),
        torch.tensor(max_deg, dtype=torch.float).unsqueeze(1),
        torch.tensor(std_deg, dtype=torch.float).unsqueeze(1),
    )


def compute_bipartite_degree(edge_index, num_nodes):
    """Compute in-degree and out-degree separately."""
    if edge_index is None or edge_index.numel() == 0:
        zeros = torch.zeros((num_nodes, 1), dtype=torch.float)
        return zeros, zeros
    
    row, col = edge_index
    in_deg = pyg_degree(col, num_nodes=num_nodes, dtype=torch.float)
    out_deg = pyg_degree(row, num_nodes=num_nodes, dtype=torch.float)
    return in_deg.unsqueeze(1), out_deg.unsqueeze(1)


def compute_eigenvector_centrality(edge_index, num_nodes):
    """Compute eigenvector centrality."""
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, 1), dtype=torch.float)
    
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(data, to_undirected=True)
    g.add_nodes_from(range(num_nodes))
    
    try:
        ec_dict = nx.eigenvector_centrality(g, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        ec_dict = {i: 1.0 / num_nodes for i in range(num_nodes)}
    
    ec_list = [ec_dict.get(i, 0.0) for i in range(num_nodes)]
    return torch.tensor(ec_list, dtype=torch.float).unsqueeze(1)


def compute_degree_assortativity(edge_index, num_nodes):
    """Compute degree assortativity coefficient (global graph property, broadcast to nodes)."""
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, 1), dtype=torch.float)
    
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(data, to_undirected=True)
    g.add_nodes_from(range(num_nodes))
    
    try:
        coeff = nx.degree_assortativity_coefficient(g)
    except Exception:
        coeff = 0.0
    
    return torch.full((num_nodes, 1), coeff, dtype=torch.float)


def compute_basic_structure_features(edge_index, num_nodes):
    """
    Compute basic structural features: degree, clustering, PageRank.
    Returns tensor of shape (num_nodes, 3).
    """
    deg = compute_degree(edge_index, num_nodes)
    clustering = compute_clustering_coefficient(edge_index, num_nodes)
    pagerank = compute_pagerank(edge_index, num_nodes)
    
    return torch.cat([deg, clustering, pagerank], dim=1)


def compute_extended_structure_features(edge_index, num_nodes):
    """
    Compute extended structural features including neighbor degree stats.
    Returns tensor of shape (num_nodes, 6).
    """
    basic = compute_basic_structure_features(edge_index, num_nodes)
    
    mean_nbr_deg, max_nbr_deg, std_nbr_deg = compute_neighbor_degree_stats(edge_index, num_nodes)
    
    return torch.cat([basic, mean_nbr_deg, max_nbr_deg, std_nbr_deg], dim=1)


def compute_full_structure_features(edge_index, num_nodes):
    """
    Compute full set of structural features:
    - degree, clustering, PageRank (3)
    - mean/max/std neighbor degree (3)
    - in-degree, out-degree (2)
    - eigenvector centrality (1)
    - degree assortativity (1, global)
    
    Returns tensor of shape (num_nodes, 10).
    """
    extended = compute_extended_structure_features(edge_index, num_nodes)
    
    in_deg, out_deg = compute_bipartite_degree(edge_index, num_nodes)
    eigvec = compute_eigenvector_centrality(edge_index, num_nodes)
    assort = compute_degree_assortativity(edge_index, num_nodes)
    
    return torch.cat([extended, in_deg, out_deg, eigvec, assort], dim=1)


def compute_spgnn_position_features(edge_index, num_nodes, n_hops=2):
    """
    Compute SP-GNN style position/structure features using random walk probabilities.
    
    Uses truncated random walks to estimate node positions in the graph.
    Features: probability of returning to node after k steps for k=1..n_hops.
    
    Returns tensor of shape (num_nodes, n_hops).
    """
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, n_hops), dtype=torch.float)
    
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(data, to_undirected=True)
    g.add_nodes_from(range(num_nodes))
    
    # Compute adjacency matrix
    adj = nx.to_scipy_sparse_array(g, dtype=np.float64)
    
    # Normalize to get transition matrix
    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    D_inv = np.diag(1.0 / row_sums.flatten())
    P = D_inv @ adj
    
    features = []
    P_k = P.copy()
    
    for k in range(1, n_hops + 1):
        # P^k[i, i] = probability of returning to node i after k steps
        return_probs = np.array(P_k.diagonal()).flatten()
        features.append(return_probs)
        
        if k < n_hops:
            P_k = P_k @ P
    
    features = np.column_stack(features)
    return torch.tensor(features, dtype=torch.float)


def compute_all_features(edge_index, num_nodes, include_spgnn=True):
    """
    Compute all available structural features.
    
    Returns:
        features: tensor of shape (num_nodes, d) with all features
        feature_names: list of feature names
    """
    full = compute_full_structure_features(edge_index, num_nodes)
    feature_names = [
        "degree", "clustering", "pagerank",
        "mean_nbr_degree", "max_nbr_degree", "std_nbr_degree",
        "in_degree", "out_degree",
        "eigenvector_centrality", "degree_assortativity"
    ]
    
    if include_spgnn:
        spgnn = compute_spgnn_position_features(edge_index, num_nodes, n_hops=2)
        full = torch.cat([full, spgnn], dim=1)
        feature_names.extend(["rw_return_1step", "rw_return_2steps"])
    
    return full, feature_names
