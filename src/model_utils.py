"""
Model utilities for text classification pipeline.
Handles model loading, prediction, evaluation, and deployment utilities.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    pipeline
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, saving, and basic operations."""

    def __init__(self, model_name: str, num_labels: int, cache_dir: Optional[str] = None):
        """
        Initialize model manager.

        Args:
            model_name: Name of the pretrained model
            num_labels: Number of classification labels
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load model and tokenizer.

        Args:
            model_path: Path to saved model (optional, uses model_name if not provided)
        """
        model_source = model_path if model_path else self.model_name

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                num_labels=self.num_labels,
                cache_dir=self.cache_dir
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                cache_dir=self.cache_dir
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.to(self.device)
            logger.info(f"Model loaded successfully from {model_source}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def save_model(self, save_path: str) -> None:
        """
        Save model and tokenizer.

        Args:
            save_path: Directory to save model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")

        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        logger.info(f"Model saved to {save_path}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {}

        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "device": str(self.device),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }


class TextClassificationPredictor:
    """Handles prediction and inference for text classification."""

    def __init__(self, model_manager: ModelManager):
        """
        Initialize predictor.

        Args:
            model_manager: Initialized ModelManager instance
        """
        self.model_manager = model_manager
        self.pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize Hugging Face pipeline for easy inference."""
        if self.model_manager.model and self.model_manager.tokenizer:
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_manager.model,
                tokenizer=self.model_manager.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

    def predict_single(self, text: str, return_probabilities: bool = False) -> Union[str, Dict]:
        """
        Predict class for a single text.

        Args:
            text: Input text
            return_probabilities: Whether to return class probabilities

        Returns:
            Predicted class or dictionary with class and probabilities
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized. Load model first.")

        result = self.pipeline(text)

        if return_probabilities:
            # Get all class probabilities
            inputs = self.model_manager.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True
            ).to(self.model_manager.device)

            with torch.no_grad():
                outputs = self.model_manager.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probabilities = probabilities.cpu().numpy()[0]

            return {
                "predicted_class": result[0]["label"],
                "confidence": result[0]["score"],
                "all_probabilities": probabilities.tolist(),
                "class_names": [f"LABEL_{i}" for i in range(len(probabilities))]
            }

        return result[0]["label"]

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict classes for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized. Load model first.")

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.pipeline(batch)
            results.extend(batch_results)

        return results


def save_evaluation_results(results: Dict, filepath: str):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.floating):
            serializable_results[key] = float(value)
        elif isinstance(value, np.integer):
            serializable_results[key] = int(value)
        else:
            serializable_results[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Evaluation results saved to {filepath}")


def load_evaluation_results(filepath: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)

    logger.info(f"Evaluation results loaded from {filepath}")
    return results
