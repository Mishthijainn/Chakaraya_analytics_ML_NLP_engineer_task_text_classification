"""
Configuration file for text classification pipeline.
Contains model hyperparameters, training settings, and dataset configurations.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_name: str = "distilbert-base-uncased"
    multilingual_models: List[str] = None
    num_labels: int = 2
    max_length: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    def __post_init__(self):
        if self.multilingual_models is None:
            self.multilingual_models = [
                "distilbert-base-multilingual-cased",
                "xlm-roberta-base",
                "bert-base-multilingual-cased"
            ]


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    output_dir: str = "./models/trained_model"
    tokenizer_dir: str = "./models/tokenizer"
    reports_dir: str = "./reports"

    # Training hyperparameters
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_train_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Advanced training settings
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    dataloader_num_workers: int = 4
    save_total_limit: int = 3
    seed: int = 42

    # Evaluation settings
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    logging_steps: int = 100

    # Early stopping
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3


@dataclass
class DataConfig:
    """Data configuration parameters."""
    dataset_name: str = "imdb"
    text_column: str = "text"
    label_column: str = "label"
    train_split: str = "train"
    test_split: str = "test"
    validation_split: Optional[str] = None

    # Data preprocessing
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    preprocessing_num_workers: int = 4

    # Alternative datasets configuration
    alternative_datasets: Dict[str, Dict] = None

    def __post_init__(self):
        if self.alternative_datasets is None:
            self.alternative_datasets = {
                "ag_news": {
                    "dataset_name": "ag_news",
                    "text_column": "text",
                    "label_column": "label",
                    "num_labels": 4
                },
                "yelp_polarity": {
                    "dataset_name": "yelp_polarity",
                    "text_column": "text",
                    "label_column": "label",
                    "num_labels": 2
                },
                "amazon_polarity": {
                    "dataset_name": "amazon_polarity",
                    "text_column": "content",
                    "label_column": "label",
                    "num_labels": 2
                }
            }


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    metrics: List[str] = None
    compute_detailed_metrics: bool = True
    save_predictions: bool = True
    create_confusion_matrix: bool = True
    create_roc_curve: bool = True
    perform_error_analysis: bool = True

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1"]


@dataclass
class MultilingualConfig:
    """Multilingual configuration parameters."""
    enable_multilingual: bool = True
    target_languages: List[str] = None
    translation_model: str = "Helsinki-NLP/opus-mt-en-mul"
    cross_lingual_evaluation: bool = True
    zero_shot_languages: List[str] = None

    def __post_init__(self):
        if self.target_languages is None:
            self.target_languages = ["en", "es", "fr", "de", "zh"]
        if self.zero_shot_languages is None:
            self.zero_shot_languages = ["it", "pt", "ru"]


@dataclass
class HyperparameterConfig:
    """Hyperparameter tuning configuration."""
    enable_hyperparameter_tuning: bool = False
    tuning_method: str = "optuna"  # or "ray"
    n_trials: int = 20

    # Parameter search spaces
    learning_rate_range: tuple = (1e-6, 1e-3)
    batch_size_options: List[int] = None
    epochs_range: tuple = (2, 5)

    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [8, 16, 32]


@dataclass
class DeploymentConfig:
    """Deployment configuration parameters."""
    enable_deployment: bool = True
    deployment_type: str = "fastapi"  # or "streamlit"
    host: str = "0.0.0.0"
    port: int = 8000
    model_version: str = "v1.0"

    # API configuration
    max_prediction_length: int = 512
    batch_prediction_limit: int = 100
    enable_logging: bool = True


class Config:
    """Main configuration class that combines all config components."""

    def __init__(self, config_name: str = "default"):
        self.config_name = config_name
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
        self.multilingual = MultilingualConfig()
        self.hyperparameter = HyperparameterConfig()
        self.deployment = DeploymentConfig()

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for the project."""
        directories = [
            self.training.output_dir,
            self.training.tokenizer_dir,
            self.training.reports_dir,
            "./models/trained_model",
            "./models/tokenizer"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "evaluation": self.evaluation.__dict__,
            "multilingual": self.multilingual.__dict__,
            "hyperparameter": self.hyperparameter.__dict__,
            "deployment": self.deployment.__dict__
        }

    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        import json
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from JSON file."""
        import json
        with open(filepath, "r") as f:
            config_dict = json.load(f)

        config = cls()
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

        return config


# Default configuration instance
default_config = Config()


# Specialized configurations for different scenarios
class IMDBConfig(Config):
    """Configuration optimized for IMDB sentiment analysis."""

    def __init__(self):
        super().__init__("imdb")
        self.data.dataset_name = "imdb"
        self.model.num_labels = 2
        self.training.num_train_epochs = 3


class AGNewsConfig(Config):
    """Configuration optimized for AG News topic classification."""

    def __init__(self):
        super().__init__("ag_news")
        self.data.dataset_name = "ag_news"
        self.model.num_labels = 4
        self.training.num_train_epochs = 4
        self.training.learning_rate = 3e-5


class YelpConfig(Config):
    """Configuration optimized for Yelp sentiment analysis."""

    def __init__(self):
        super().__init__("yelp")
        self.data.dataset_name = "yelp_polarity"
        self.model.num_labels = 2
        self.training.num_train_epochs = 2
        self.training.train_batch_size = 32


class MultilingualPipelineConfig(Config):
    """Configuration optimized for multilingual text classification."""

    def __init__(self):
        super().__init__("multilingual")
        self.model.model_name = "distilbert-base-multilingual-cased"
        self.multilingual.enable_multilingual = True
        self.training.num_train_epochs = 4
        self.training.learning_rate = 2e-5


# Configuration factory
def get_config(config_type: str = "default") -> Config:
    """Get configuration based on type."""
    config_map = {
        "default": Config,
        "imdb": IMDBConfig,
        "ag_news": AGNewsConfig,
        "yelp": YelpConfig,
        "multilingual": MultilingualPipelineConfig
    }

    if config_type not in config_map:
        raise ValueError(f"Unknown config type: {config_type}")

    return config_map[config_type]()
