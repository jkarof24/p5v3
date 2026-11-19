"""
Dataset utilities for pure-structure graph node classification.

Loads .pt graph files, manages train/val/test splits, and prevents label leakage.
Supports both dense-embedding (de) and sparse (sp) splits.
"""

import os
import random
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


DATASET_DIR = "/home/jonathan/Desktop/P5/dataset/SwapGT-data/dataset"
SUPPORTED_DATASETS = ["acm", "BlogCatalog", "citeseer", "computer", "flickr", "photo", "uai", "wikics"]
SUPPORTED_SPLITS = ["de", "sp"]


def load_graph(dataset: str, split: str) -> Data:
    """
    Load a graph dataset from SwapGT-data/dataset and convert to PyG Data object.
    
    Raw format: tuple of (adj, features, y, train_mask, val_mask, test_mask)
    We use only adj (structure) and y (labels), computing edge_index from adj.
    
    Args:
        dataset: Dataset name (e.g., 'acm', 'citeseer')
        split: 'de' for dense-embedding (50/25/25), 'sp' for sparse (2.5/2.5/95)
    
    Returns:
        PyG Data object with x (structural features), edge_index, y, masks
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {SUPPORTED_DATASETS}")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unknown split: {split}. Supported: {SUPPORTED_SPLITS}")
    
    path = os.path.join(DATASET_DIR, split, f"{dataset}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    raw_content = torch.load(path, weights_only=False)
    
    # Raw format: (adj, features, y, train_mask, val_mask, test_mask)
    # We discard features (index 1) to use pure structure
    if isinstance(raw_content, list):
        adj = raw_content[0]
        # features = raw_content[1]  # discarded for pure structure
        y = raw_content[2]
        masks = raw_content[3:]
    else:
        raise ValueError(f"Unexpected raw content format in {path}")
    
    num_nodes = y.shape[0]
    
    # Convert adjacency matrix to edge_index
    if adj.is_sparse:
        adj = adj.to_dense()
    edge_index, _ = dense_to_sparse(adj)
    
    # Build PyG Data object with structural features computed from edge_index
    # Import here to avoid circular dependency
    from features import compute_basic_structure_features
    x = compute_basic_structure_features(edge_index, num_nodes)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Attach masks
    for name, mask in zip(['train_mask', 'val_mask', 'test_mask'], masks):
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask)
            if mask.ndim > 1:
                mask = mask[:, 0]
            if mask.dtype != torch.bool:
                bool_mask = torch.zeros(num_nodes, dtype=torch.bool)
                bool_mask[mask.long()] = True
                mask = bool_mask
            setattr(data, name, mask)
    
    return data


def get_split_masks(data: Data, split: str = "de") -> tuple:
    """
    Extract or create train/val/test masks.
    
    If masks already exist on the data object, return them.
    Otherwise, create deterministic splits:
    - de: 50% train, 25% val, 25% test
    - sp: 2.5% train, 2.5% val, 95% test
    
    Splits are stratified by label and seeded for reproducibility.
    """
    if hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask'):
        return data.train_mask, data.val_mask, data.test_mask
    
    num_nodes = data.x.shape[0]
    labels = data.y
    
    # Create stratified indices
    class_indices = {}
    for i in range(num_nodes):
        label = labels[i].item()
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    # Shuffle within each class using fixed seed
    rng = random.Random(42)
    for label in class_indices:
        rng.shuffle(class_indices[label])
    
    # Determine split ratios based on split type
    if split == "sp":
        train_ratio, val_ratio, test_ratio = 0.025, 0.025, 0.95
    else:  # de
        train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label, indices in class_indices.items():
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask


def get_num_classes(data: Data) -> int:
    """Return the number of classes in the dataset."""
    return int(data.y.max().item()) + 1


def get_dataset_info(dataset: str, split: str) -> dict:
    """
    Load dataset and return summary info without returning the full data.
    
    Returns dict with: num_nodes, num_edges, num_classes, train_size, val_size, test_size
    """
    data = load_graph(dataset, split)
    train_mask, val_mask, test_mask = get_split_masks(data)
    
    return {
        "dataset": dataset,
        "split": split,
        "num_nodes": data.x.shape[0],
        "num_edges": data.edge_index.shape[1],
        "num_classes": get_num_classes(data),
        "train_size": train_mask.sum().item(),
        "val_size": val_mask.sum().item(),
        "test_size": test_mask.sum().item(),
        "feature_dim": data.x.shape[1],
    }


def verify_no_label_leakage(data: Data, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor) -> bool:
    """
    Verify that masks are mutually exclusive and cover all nodes.
    
    Returns True if valid, raises AssertionError otherwise.
    """
    num_nodes = data.x.shape[0]
    
    # Check mutual exclusivity
    assert not (train_mask & val_mask).any(), "Train and val masks overlap"
    assert not (train_mask & test_mask).any(), "Train and test masks overlap"
    assert not (val_mask & test_mask).any(), "Val and test masks overlap"
    
    # Check coverage
    covered = train_mask | val_mask | test_mask
    assert covered.sum().item() == num_nodes, f"Masks don't cover all nodes: {covered.sum().item()}/{num_nodes}"
    
    return True
