#!/usr/bin/env python3
"""
P5 Experiment Runner: Pure-Structure Graph Node Classification

Runs all experiment variants, evaluates accuracy and macro-F1,
and writes results to JSON/CSV.

Experiments:
1. SwapGT + MLP (baseline)
2. SwapGT + Projector + LLM
3. Shuffled Embeddings + Projector + LLM
4. Random Embeddings + Projector + LLM
5. Stats-Only Prompt LLM

All computation is CPU-only. LLM calls go to llama-server at localhost:9931.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Optional

# Import local modules
import data_utils
import features
import swapgt_encoder
from llm_classifier import Projector, PromptBuilder, LLMServerClient


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPClassifier(nn.Module):
    """Simple MLP classifier for baseline."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_mlp(model, train_x, train_y, val_x, val_y, num_classes, 
              lr=0.01, weight_decay=5e-4, max_epochs=100, patience=10):
    """Train MLP with early stopping on validation set."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_output = model(val_x)
            val_pred = val_output.argmax(dim=1)
            val_acc = accuracy_score(val_y, val_pred)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_acc


def evaluate_model(model, test_x, test_y, num_classes) -> Dict:
    """Evaluate model on test set, return accuracy and macro-F1."""
    model.eval()
    with torch.no_grad():
        output = model(test_x)
        pred = output.argmax(dim=1)
    
    acc = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred, average='macro')
    
    return {"accuracy": acc, "macro_f1": f1}


def run_swapgt_mlp(dataset: str, split: str, dry_run: bool = False) -> Dict:
    """Experiment 1: SwapGT embeddings + MLP classifier."""
    print(f"\n[Experiment 1] SwapGT + MLP on {dataset} ({split})")
    
    data = data_utils.load_graph(dataset, split)
    train_mask, val_mask, test_mask = data_utils.get_split_masks(data, split)
    num_classes = data_utils.get_num_classes(data)
    
    if dry_run:
        return {"dataset": dataset, "split": split, "experiment": "swapgt_mlp", "dry_run": True}
    
    # Compute SwapGT embeddings on CPU
    print("  Computing SwapGT embeddings...")
    embeddings = swapgt_encoder.get_swapgt_embeddings(
        dataset, split, num_nodes=data.x.shape[0], 
        hidden_dim=128, use_precomputed=False, device="cpu"
    )
    
    # Prepare data
    train_x = embeddings[train_mask]
    train_y = data.y[train_mask]
    val_x = embeddings[val_mask]
    val_y = data.y[val_mask]
    test_x = embeddings[test_mask]
    test_y = data.y[test_mask]
    
    # Train MLP
    print("  Training MLP...")
    model = MLPClassifier(input_dim=128, hidden_dim=64, num_classes=num_classes)
    model, best_val_acc = train_mlp(model, train_x, train_y, val_x, val_y, num_classes)
    
    # Evaluate
    results = evaluate_model(model, test_x, test_y, num_classes)
    results.update({
        "dataset": dataset, "split": split, "experiment": "swapgt_mlp",
        "best_val_acc": best_val_acc
    })
    
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Test Macro-F1: {results['macro_f1']:.4f}")
    
    return results


def run_swapgt_llm(dataset: str, split: str, dry_run: bool = False,
                   endpoint: str = "http://localhost:9931/v1", max_test_nodes: int = 100) -> Dict:
    """Experiment 2: SwapGT embeddings + Projector + LLM."""
    print(f"\n[Experiment 2] SwapGT + Projector + LLM on {dataset} ({split})")
    
    data = data_utils.load_graph(dataset, split)
    train_mask, val_mask, test_mask = data_utils.get_split_masks(data, split)
    num_classes = data_utils.get_num_classes(data)
    
    if dry_run:
        return {"dataset": dataset, "split": split, "experiment": "swapgt_llm", "dry_run": True}
    
    # Compute SwapGT embeddings
    print("  Computing SwapGT embeddings...")
    embeddings = swapgt_encoder.get_swapgt_embeddings(
        dataset, split, num_nodes=data.x.shape[0],
        hidden_dim=128, use_precomputed=False, device="cpu"
    )
    
    # Project embeddings
    print("  Projecting embeddings...")
    projector = Projector(input_dim=128, output_dim=16)
    projected = projector.project_batch(embeddings)
    
    # Build prompts and classify
    print(f"  Classifying with LLM (max {max_test_nodes} test nodes)...")
    prompt_builder = PromptBuilder(num_classes)
    llm_client = LLMServerClient(endpoint)
    
    test_indices = torch.where(test_mask)[0]
    # Limit to max_test_nodes for practical runtime
    test_indices = test_indices[:max_test_nodes]
    
    predictions = []
    true_labels = []
    
    for i, idx in enumerate(test_indices):
        if i % 25 == 0:
            print(f"    Progress: {i}/{len(test_indices)}")
        prompt = prompt_builder.build_embedding_prompt(
            projected[idx].tolist(), top_k=5
        )
        response = llm_client.generate(prompt, max_tokens=256, temperature=0.0)
        # Parse response to extract class
        pred = 0
        for j in range(num_classes):
            if f"class_{j}" in response.lower():
                pred = j
                break
        predictions.append(pred)
        true_labels.append(data.y[idx].item())
    
    llm_client.close()
    
    # Evaluate
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    results = {
        "dataset": dataset, "split": split, "experiment": "swapgt_llm",
        "accuracy": acc, "macro_f1": f1,
        "num_test_nodes": len(predictions)
    }
    
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test Macro-F1: {f1:.4f}")
    
    return results


def run_shuffled_embeddings(dataset: str, split: str, dry_run: bool = False,
                            max_test_nodes: int = 100) -> Dict:
    """Experiment 3: Shuffled SwapGT embeddings + Projector + LLM."""
    print(f"\n[Experiment 3] Shuffled Embeddings + LLM on {dataset} ({split})")
    
    data = data_utils.load_graph(dataset, split)
    train_mask, val_mask, test_mask = data_utils.get_split_masks(data, split)
    num_classes = data_utils.get_num_classes(data)
    
    if dry_run:
        return {"dataset": dataset, "split": split, "experiment": "shuffled_llm", "dry_run": True}
    
    # Compute and shuffle embeddings
    print("  Computing and shuffling embeddings...")
    embeddings = swapgt_encoder.get_swapgt_embeddings(
        dataset, split, num_nodes=data.x.shape[0],
        hidden_dim=128, use_precomputed=False, device="cpu"
    )
    
    # Shuffle rows (break node-embedding correspondence)
    perm = torch.randperm(embeddings.shape[0])
    shuffled = embeddings[perm]
    
    # Project and classify
    print(f"  Classifying with LLM (max {max_test_nodes} test nodes)...")
    projector = Projector(input_dim=128, output_dim=16)
    projected = projector.project_batch(shuffled)
    
    prompt_builder = PromptBuilder(num_classes)
    llm_client = LLMServerClient()
    
    test_indices = torch.where(test_mask)[0][:max_test_nodes]
    predictions = []
    true_labels = []
    
    for i, idx in enumerate(test_indices):
        if i % 25 == 0:
            print(f"    Progress: {i}/{len(test_indices)}")
        prompt = prompt_builder.build_embedding_prompt(
            projected[idx].tolist(), top_k=5
        )
        response = llm_client.generate(prompt, max_tokens=256, temperature=0.0)
        pred = 0
        for j in range(num_classes):
            if f"class_{j}" in response.lower():
                pred = j
                break
        predictions.append(pred)
        true_labels.append(data.y[idx].item())
    
    llm_client.close()
    
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    results = {
        "dataset": dataset, "split": split, "experiment": "shuffled_llm",
        "accuracy": acc, "macro_f1": f1,
        "num_test_nodes": len(predictions)
    }
    
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test Macro-F1: {f1:.4f}")
    
    return results


def run_random_embeddings(dataset: str, split: str, dry_run: bool = False,
                          max_test_nodes: int = 100) -> Dict:
    """Experiment 4: Random embeddings + Projector + LLM."""
    print(f"\n[Experiment 4] Random Embeddings + LLM on {dataset} ({split})")
    
    data = data_utils.load_graph(dataset, split)
    train_mask, val_mask, test_mask = data_utils.get_split_masks(data, split)
    num_classes = data_utils.get_num_classes(data)
    
    if dry_run:
        return {"dataset": dataset, "split": split, "experiment": "random_llm", "dry_run": True}
    
    # Generate random embeddings
    print("  Generating random embeddings...")
    embeddings = torch.randn(data.x.shape[0], 128)
    
    # Project and classify
    print(f"  Classifying with LLM (max {max_test_nodes} test nodes)...")
    projector = Projector(input_dim=128, output_dim=16)
    projected = projector.project_batch(embeddings)
    
    prompt_builder = PromptBuilder(num_classes)
    llm_client = LLMServerClient()
    
    test_indices = torch.where(test_mask)[0][:max_test_nodes]
    predictions = []
    true_labels = []
    
    for i, idx in enumerate(test_indices):
        if i % 25 == 0:
            print(f"    Progress: {i}/{len(test_indices)}")
        prompt = prompt_builder.build_embedding_prompt(
            projected[idx].tolist(), top_k=5
        )
        response = llm_client.generate(prompt, max_tokens=256, temperature=0.0)
        pred = 0
        for j in range(num_classes):
            if f"class_{j}" in response.lower():
                pred = j
                break
        predictions.append(pred)
        true_labels.append(data.y[idx].item())
    
    llm_client.close()
    
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    results = {
        "dataset": dataset, "split": split, "experiment": "random_llm",
        "accuracy": acc, "macro_f1": f1,
        "num_test_nodes": len(predictions)
    }
    
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test Macro-F1: {f1:.4f}")
    
    return results


def run_stats_only_llm(dataset: str, split: str, dry_run: bool = False,
                       endpoint: str = "http://localhost:9931/v1",
                       max_test_nodes: int = 100) -> Dict:
    """Experiment 5: Stats-only prompt LLM (no embeddings)."""
    print(f"\n[Experiment 5] Stats-Only Prompt LLM on {dataset} ({split})")
    
    data = data_utils.load_graph(dataset, split)
    train_mask, val_mask, test_mask = data_utils.get_split_masks(data, split)
    num_classes = data_utils.get_num_classes(data)
    
    if dry_run:
        return {"dataset": dataset, "split": split, "experiment": "stats_llm", "dry_run": True}
    
    # Compute structural features
    print("  Computing structural features...")
    struct_features = features.compute_full_structure_features(
        data.edge_index, data.x.shape[0]
    )
    
    # Build stats-only prompts and classify
    print(f"  Classifying with LLM (max {max_test_nodes} test nodes)...")
    prompt_builder = PromptBuilder(num_classes)
    llm_client = LLMServerClient(endpoint)
    
    test_indices = torch.where(test_mask)[0][:max_test_nodes]
    predictions = []
    true_labels = []
    
    for i, idx in enumerate(test_indices):
        if i % 25 == 0:
            print(f"    Progress: {i}/{len(test_indices)}")
        node_stats = {
            "degree": struct_features[idx, 0].item(),
            "clustering": struct_features[idx, 1].item(),
            "pagerank": struct_features[idx, 2].item()
        }
        prompt = prompt_builder.build_stats_prompt(node_stats)
        response = llm_client.generate(prompt, max_tokens=256, temperature=0.0)
        pred = 0
        for j in range(num_classes):
            if f"class_{j}" in response.lower():
                pred = j
                break
        predictions.append(pred)
        true_labels.append(data.y[idx].item())
    
    llm_client.close()
    
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    results = {
        "dataset": dataset, "split": split, "experiment": "stats_llm",
        "accuracy": acc, "macro_f1": f1,
        "num_test_nodes": len(predictions)
    }
    
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test Macro-F1: {f1:.4f}")
    
    return results


def run_all_experiments(dataset: str, split: str, dry_run: bool = False,
                        endpoint: str = "http://localhost:9931/v1",
                        max_test_nodes: int = 100, output_dir: str = "results",
                        resume: bool = False) -> List[Dict]:
    """Run all experiment variants on a dataset/split with progress logging."""
    set_seed(42)
    
    # Setup logging
    log_path = os.path.join(output_dir, f"experiment_{dataset}_{split}.log")
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(log_path, "a" if resume else "w")
    
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    
    # Load existing results if resuming
    results = []
    if resume:
        json_path = os.path.join(output_dir, "results.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                results = json.load(f)
            log(f"Resuming with {len(results)} existing results")
    
    # Track which experiments are complete
    completed = {r["experiment"] for r in results}
    
    try:
        # Experiment 1: SwapGT + MLP
        if "swapgt_mlp" not in completed:
            result = run_swapgt_mlp(dataset, split, dry_run)
            results.append(result)
            write_results(results, output_dir)
            log(f"Experiment 1 complete: {result['accuracy']:.4f}")
        else:
            log("Experiment 1 already complete, skipping")
        
        if not dry_run:
            # Experiment 2: SwapGT + Projector + LLM
            if "swapgt_llm" not in completed:
                result = run_swapgt_llm(dataset, split, dry_run, endpoint, max_test_nodes)
                results.append(result)
                write_results(results, output_dir)
                log(f"Experiment 2 complete: {result['accuracy']:.4f}")
            else:
                log("Experiment 2 already complete, skipping")
            
            # Experiment 3: Shuffled Embeddings + LLM
            if "shuffled_llm" not in completed:
                result = run_shuffled_embeddings(dataset, split, dry_run, max_test_nodes)
                results.append(result)
                write_results(results, output_dir)
                log(f"Experiment 3 complete: {result['accuracy']:.4f}")
            else:
                log("Experiment 3 already complete, skipping")
            
            # Experiment 4: Random Embeddings + LLM
            if "random_llm" not in completed:
                result = run_random_embeddings(dataset, split, dry_run, max_test_nodes)
                results.append(result)
                write_results(results, output_dir)
                log(f"Experiment 4 complete: {result['accuracy']:.4f}")
            else:
                log("Experiment 4 already complete, skipping")
            
            # Experiment 5: Stats-Only Prompt LLM
            if "stats_llm" not in completed:
                result = run_stats_only_llm(dataset, split, dry_run, endpoint, max_test_nodes)
                results.append(result)
                write_results(results, output_dir)
                log(f"Experiment 5 complete: {result['accuracy']:.4f}")
            else:
                log("Experiment 5 already complete, skipping")
    finally:
        log_file.close()
    
    return results


def write_results(results: List[Dict], output_dir: str = "results"):
    """Write results to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON output
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {json_path}")
    
    # CSV output
    csv_path = os.path.join(output_dir, "results.csv")
    if results:
        import csv
        keys = ["dataset", "split", "experiment", "accuracy", "macro_f1"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for result in results:
                row = {k: result.get(k, "") for k in keys}
                writer.writerow(row)
    print(f"Results written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="P5 Experiment Runner")
    parser.add_argument("--dataset", default="acm", help="Dataset name")
    parser.add_argument("--split", default="de", choices=["de", "sp"], help="Split type")
    parser.add_argument("--dry_run", action="store_true", help="Dry run (no actual computation)")
    parser.add_argument("--endpoint", default="http://localhost:9931/v1", help="LLM server endpoint")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--max_test_nodes", type=int, default=100,
                        help="Max test nodes for LLM experiments (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed experiment (skip already completed)")
    parser.add_argument("--experiments", type=int, default=0,
                        help="Run specific experiments (bitmask: 1=swapgt_mlp, 2=swapgt_llm, 4=shuffled, 8=random, 16=stats)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("P5 Experiment Runner: Pure-Structure Graph Node Classification")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Dry run: {args.dry_run}")
    print(f"Endpoint: {args.endpoint}")
    print("=" * 60)
    
    # Verify CPU-only
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    # Run experiments (with progress logging)
    results = run_all_experiments(
        args.dataset, args.split, args.dry_run, args.endpoint, args.max_test_nodes,
        args.output_dir, args.resume
    )
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
