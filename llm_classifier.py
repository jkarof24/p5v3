"""
LLM-based node classifier using llama-server HTTP API.

Components:
- Projector: Maps high-dimensional embeddings to LLM-friendly dimension
- PromptBuilder: Constructs classification prompts from node features
- LLMServerClient: HTTP client for llama-server at localhost:9931
- LLMClassifier: End-to-end classifier combining all components
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_ENDPOINT = "http://localhost:9931/v1"
DEFAULT_MODEL = "davidau-qwen38"


class Projector(nn.Module):
    """
    Projects high-dimensional embeddings to a lower-dimensional space
    suitable for LLM prompting or downstream classification.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def project_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project a batch of embeddings."""
        with torch.no_grad():
            return self.network(embeddings)


class PromptBuilder:
    """
    Builds classification prompts from node features and class labels.
    
    Supports two modes:
    - Embedding mode: Uses projected embedding values in the prompt
    - Stats mode: Uses structural statistics directly in the prompt
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        if class_names is None:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            self.class_names = class_names
    
    def build_embedding_prompt(self, node_features: List[float], top_k: int = 5) -> str:
        """
        Build a prompt from projected embedding values.
        
        Args:
            node_features: Projected embedding values for a node
            top_k: Number of most important dimensions to include
        
        Returns:
            Classification prompt string
        """
        # Find top-k most informative dimensions (largest absolute values)
        values = [(i, abs(v), v) for i, v in enumerate(node_features)]
        values.sort(key=lambda x: x[1], reverse=True)
        top_dims = values[:top_k]
        
        # Build feature description
        feature_desc = ", ".join([
            f"dim{i}={v:.4f}" for i, _, v in top_dims
        ])
        
        classes_str = ", ".join(self.class_names)
        
        prompt = (
            f"Classify this graph node based on its structural embedding.\n\n"
            f"Node embedding (top {top_k} dimensions): {feature_desc}\n\n"
            f"Possible classes: {classes_str}\n\n"
            f"What is the most likely class for this node? Respond with only the class name."
        )
        
        return prompt
    
    def build_stats_prompt(self, node_stats: Dict[str, float]) -> str:
        """
        Build a prompt from structural statistics.
        
        Args:
            node_stats: Dictionary of structural statistics for a node
        
        Returns:
            Classification prompt string
        """
        stats_desc = "\n".join([
            f"  {key}: {value:.4f}" for key, value in node_stats.items()
        ])
        
        classes_str = ", ".join(self.class_names)
        
        prompt = (
            f"Classify this graph node based on its structural properties.\n\n"
            f"Node structural statistics:\n{stats_desc}\n\n"
            f"Possible classes: {classes_str}\n\n"
            f"What is the most likely class for this node? Respond with only the class name."
        )
        
        return prompt
    
    def build_few_shot_prompt(self, examples: List[Dict], query_features: List[float]) -> str:
        """
        Build a few-shot classification prompt.
        
        Args:
            examples: List of example dicts with 'features' and 'label' keys
            query_features: Features for the query node
        
        Returns:
            Few-shot classification prompt
        """
        prompt = "Classify graph nodes based on their structural embeddings.\n\n"
        
        for example in examples:
            features_str = ", ".join([f"{v:.4f}" for v in example['features'][:5]])
            prompt += (
                f"Embedding: [{features_str}...]\n"
                f"Class: {example['label']}\n\n"
            )
        
        query_str = ", ".join([f"{v:.4f}" for v in query_features[:5]])
        prompt += (
            f"Embedding: [{query_str}...]\n"
            f"Class: "
        )
        
        return prompt


class LLMServerClient:
    """
    HTTP client for llama-server API.
    
    Communicates with llama-server running at the specified endpoint
    to perform text generation for classification.
    """
    
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL,
                 timeout: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """
        Generate text completion from the LLM server.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
        
        Returns:
            Generated text
        """
        url = f"{self.endpoint}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to llama-server at {self.endpoint}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to llama-server timed out after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"Error calling llama-server: {e}")
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 256,
                       temperature: float = 0.0) -> List[str]:
        """
        Generate completions for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, max_tokens, temperature))
        return results
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()


class LLMClassifier:
    """
    End-to-end LLM-based node classifier.
    
    Combines embedding projection, prompt building, and LLM inference
    to classify graph nodes.
    """
    
    def __init__(self, num_classes: int, embedding_dim: int, projection_dim: int = 16,
                 endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL,
                 class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        # Projector: embedding_dim -> projection_dim
        self.projector = Projector(embedding_dim, projection_dim)
        
        # Prompt builder
        self.prompt_builder = PromptBuilder(num_classes, self.class_names)
        
        # LLM client
        self.llm_client = LLMServerClient(endpoint, model)
        
        # Projection dimension for prompts
        self.projection_dim = projection_dim
    
    def classify_node(self, embedding: torch.Tensor) -> str:
        """
        Classify a single node.
        
        Args:
            embedding: Node embedding tensor of shape (embedding_dim,)
        
        Returns:
            Predicted class name
        """
        # Project embedding
        projected = self.projector.project_batch(embedding.unsqueeze(0)).squeeze(0)
        
        # Build prompt
        prompt = self.prompt_builder.build_embedding_prompt(
            projected.tolist(), top_k=min(5, self.projection_dim)
        )
        
        # Get LLM prediction
        response = self.llm_client.generate(prompt, max_tokens=10, temperature=0.0)
        
        return self._parse_response(response)
    
    def classify_batch(self, embeddings: torch.Tensor) -> List[str]:
        """
        Classify a batch of nodes.
        
        Args:
            embeddings: Node embeddings tensor of shape (batch_size, embedding_dim)
        
        Returns:
            List of predicted class names
        """
        # Project all embeddings
        projected = self.projector.project_batch(embeddings)
        
        # Build prompts
        prompts = []
        for i in range(projected.shape[0]):
            prompt = self.prompt_builder.build_embedding_prompt(
                projected[i].tolist(), top_k=min(5, self.projection_dim)
            )
            prompts.append(prompt)
        
        # Get LLM predictions
        responses = self.llm_client.generate_batch(prompts, max_tokens=10, temperature=0.0)
        
        return [self._parse_response(r) for r in responses]
    
    def _parse_response(self, response: str) -> str:
        """
        Parse LLM response to extract class prediction.
        
        Args:
            response: Raw LLM response string
        
        Returns:
            Predicted class name
        """
        response = response.strip().lower()
        
        # Try to match exact class names
        for class_name in self.class_names:
            if class_name.lower() in response:
                return class_name
        
        # Try to extract class number
        import re
        match = re.search(r'class[_\s]*(\d+)', response)
        if match:
            class_idx = int(match.group(1))
            if 0 <= class_idx < self.num_classes:
                return self.class_names[class_idx]
        
        # Default to first class if parsing fails
        return self.class_names[0]
    
    def close(self):
        """Close the LLM client."""
        self.llm_client.close()
