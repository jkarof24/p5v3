P5: Pure-Structure Graph Node Classification
==============================================

This project explores whether graph node classification can be performed using
only graph structure (no node features) by combining SwapGT graph transformer
embeddings with LLM-based classification.

Architecture
------------

The pipeline has three main stages:

1. STRUCTURAL EMBEDDING (swapgt_encoder.py)
   - Takes graph structure (edge_index) only
   - Uses SwapGT (Token-Swapping Graph Transformer) to learn node embeddings
   - Learnable node embeddings initialized randomly, refined through transformer layers
   - No node features (x) used - pure structure
   - Runs on CPU with 128-dimensional embeddings

2. FEATURE EXTRACTION (features.py)
   - Computes 12 unsupervised structural features per node:
     * Degree, clustering coefficient, PageRank
     * Mean/max/std neighbor degree
     * In-degree, out-degree
     * Eigenvector centrality
     * Degree assortativity (global, broadcast)
     * SP-GNN random walk return probabilities (1-step, 2-step)
   - Used for stats-only LLM prompting baseline

3. LLM CLASSIFICATION (llm_classifier.py)
   - Projector: Maps 128-dim embeddings to 16-dim (LLM-friendly)
   - PromptBuilder: Constructs classification prompts
   - LLMServerClient: HTTP client to llama-server at localhost:9931
   - Model: davidau-qwen38 (Q4_K_M GGUF) running on Tesla P40 GPU

Experiments
-----------

Five experiment variants are run to isolate the contribution of each component:

1. SwapGT + MLP (baseline)
   - SwapGT embeddings fed to a simple 2-layer MLP classifier
   - Trained with Adam, early stopping on validation set
   - Establishes the baseline for structural embedding quality

2. SwapGT + Projector + LLM
   - SwapGT embeddings projected to 16-dim, then classified by LLM
   - Tests whether LLM can leverage structural embeddings
   - Main experimental condition

3. Shuffled Embeddings + Projector + LLM
   - SwapGT embeddings computed but node-embedding correspondence broken
   - Tests whether LLM is using actual structural information or just pattern matching
   - If performance drops significantly, embeddings contain useful signal

4. Random Embeddings + Projector + LLM
   - Random Gaussian embeddings instead of SwapGT
   - Tests whether any embedding works or if SwapGT is necessary
   - Lower bound for embedding quality

5. Stats-Only Prompt LLM
   - No embeddings; only structural statistics in prompt
   - Tests whether raw structural features are sufficient
   - Isolates the value of learned embeddings vs. hand-crafted features

Why This Approach?
------------------

Traditional graph classification relies on node features (text, attributes).
This project asks: can structure alone be sufficient?

- SwapGT learns structural representations without features
- LLM provides powerful reasoning over the representations
- The shuffled/random embedding controls verify the signal is real
- The stats-only baseline shows what embeddings add over raw features

Data & Splits
-------------

Datasets: acm, BlogCatalog, citeseer, computer, flickr, photo, uai, wikics
Location: /home/jonathan/Desktop/P5/dataset/SwapGT-data/dataset/{de,sp}/

Raw format: tuple of (adj, features, y, train_mask, val_mask, test_mask)
We use only adj (structure) and y (labels), discarding original features
to ensure pure-structure classification. Edge indices are computed from
the adjacency matrix, and structural node features (degree, clustering,
PageRank) are computed from the graph structure.

Two split types:
- de (dense-embedding): 50% train / 25% val / 25% test
- sp (sparse): 2.5% train / 2.5% val / 95% test

Splits are stratified by label, deterministic (seed=42), with verified
no-overlap between train/val/test sets.

All SwapGT embeddings are computed from scratch (no pre-computed embeddings).

CPU-Only Design
---------------

All PyTorch computation runs on CPU (CUDA_VISIBLE_DEVICES=""). The only
GPU usage is llama-server running on the Tesla P40 (CUDA device 0) for
LLM inference via HTTP. This separation allows:
- Easy scaling of LLM inference independently
- No GPU memory contention between embedding computation and LLM
- Clear resource accounting

File Structure
--------------

data_utils.py       - Dataset loading, split management, leakage verification
features.py         - 12 unsupervised structural feature computations
swapgt_encoder.py   - SwapGT encoder for pure-structure node embeddings
llm_classifier.py   - Projector, prompt builder, LLM HTTP client
run_experiments.py  - Experiment runner (all 5 variants)
start_llama_server.sh - Start llama-server on P40 GPU (port 9931)
run_p5_experiments.sh - Run experiments on CPU

Usage
-----

1. Start llama-server:
   ./start_llama_server.sh

2. Run experiments:
   ./run_p5_experiments.sh --dataset acm --split de
   ./run_p5_experiments.sh --dry_run  # Quick test without computation

3. Results are written to results/results.json and results/results.csv

Dependencies
------------
- Python 3.x (venv: /home/jonathan/venv/bin/python)
- PyTorch, PyTorch Geometric
- scikit-learn (metrics)
- networkx (graph algorithms)
- requests (HTTP client for llama-server)
- llama-server with Q4_K_M GGUF model
