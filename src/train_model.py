"""
Main training script for text classification using Hugging Face Transformers.
Supports fine-tuning, evaluation, multilingual training, and hyperparameter tuning.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
print("TrainingArguments source:", TrainingArguments.__module__)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import wandb
import optuna

# Import local modules
from config import get_config, Config
from data_preprocessing import create_data_pipeline, analyze_dataset
from model_utils import ModelManager, TextClassificationPredictor, save_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextClassificationTrainer:
    """Main trainer class for text classification."""

    def __init__(self, config: Config):
        """
        Initialize trainer with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.trainer = None
        self.model_manager = None

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize wandb if tracking is enabled
        if hasattr(self.config.training, 'use_wandb') and self.config.training.use_wandb:
            wandb.init(
                project="text-classification",
                config=self.config.to_dict(),
                name=f"{self.config.model.model_name}_{self.config.data.dataset_name}"
            )

    def prepare_data(self, dataset_name: str = None, custom_data_path: str = None):
        """
        Prepare training, validation, and test datasets.

        Args:
            dataset_name: Name of dataset from Hub
            custom_data_path: Path to custom dataset
        """
        logger.info("Preparing datasets...")

        # Use config defaults if not provided
        if dataset_name is None:
            dataset_name = self.config.data.dataset_name

        # Create data pipeline
        try:
            dataset, tokenizer_wrapper, analysis = create_data_pipeline(
                self.config, dataset_name, custom_data_path
            )

            self.train_dataset = dataset['train']
            self.eval_dataset = dataset['validation'] if 'validation' in dataset else dataset['test']
            self.test_dataset = dataset['test']
            self.tokenizer = tokenizer_wrapper.tokenizer

            # Log dataset statistics
            logger.info("Dataset Analysis:")
            for split, stats in analysis.items():
                logger.info(f"{split}: {stats['num_samples']} samples, {stats['num_labels']} labels")

            # Save dataset analysis
            analysis_path = os.path.join(self.config.training.reports_dir, "dataset_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)

            logger.info("Data preparation completed successfully")

        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def setup_model(self):
        """Set up the model for training."""
        logger.info(f"Setting up model: {self.config.model.model_name}")

        try:
            # Initialize model manager
            self.model_manager = ModelManager(
                model_name=self.config.model.model_name,
                num_labels=self.config.model.num_labels
            )

            # Load model
            self.model_manager.load_model()
            self.model = self.model_manager.model

            # Update tokenizer if needed
            if self.tokenizer is None:
                self.tokenizer = self.model_manager.tokenizer

            logger.info("Model setup completed")

        except Exception as e:
            logger.error(f"Error in model setup: {str(e)}")
            raise

    def setup_trainer(self):
        """Set up Hugging Face Trainer."""
        logger.info("Setting up trainer...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.train_batch_size,
            per_device_eval_batch_size=self.config.training.eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            warmup_steps=self.config.training.warmup_steps,
            weight_decay=self.config.training.weight_decay,
            learning_rate=self.config.training.learning_rate,
            fp16=self.config.training.fp16,
            logging_dir=os.path.join(self.config.training.output_dir, "logs"),
            logging_steps=self.config.training.logging_steps,
            evaluation_strategy=self.config.training.evaluation_strategy,
            eval_steps=self.config.training.eval_steps,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            seed=self.config.training.seed,
            report_to="wandb" if hasattr(self.config.training, 'use_wandb') and self.config.training.use_wandb else None
        )

        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )

        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )

            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.training.early_stopping_patience
            )]
        )

        logger.info("Trainer setup completed")

    def train(self):
        """Train the model."""
        logger.info("Starting model training...")

        try:
            # Train the model
            train_result = self.trainer.train()

            # Save training metrics
            train_metrics = train_result.metrics
            train_metrics["train_samples"] = len(self.train_dataset)

            # Log training results
            logger.info("Training completed successfully")
            logger.info(f"Training loss: {train_metrics.get('train_loss', 'N/A')}")

            # Save model and tokenizer
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.training.output_dir)

            # Save training metrics
            metrics_path = os.path.join(self.config.training.reports_dir, "training_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(train_metrics, f, indent=2, default=str)

            return train_metrics

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def evaluate(self, dataset_name: str = "test") -> Dict:
        """
        Evaluate the model on test dataset.

        Args:
            dataset_name: Name of dataset split to evaluate on

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} dataset...")

        try:
            # Choose dataset
            if dataset_name == "test":
                eval_dataset = self.test_dataset
            elif dataset_name == "validation":
                eval_dataset = self.eval_dataset
            else:
                eval_dataset = self.test_dataset

            # Evaluate
            eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)

            # Add dataset info
            eval_results["eval_samples"] = len(eval_dataset)
            eval_results["dataset"] = dataset_name

            logger.info("Evaluation completed")
            logger.info(f"Evaluation results: {eval_results}")

            # Save evaluation results
            eval_path = os.path.join(self.config.training.reports_dir, f"evaluation_{dataset_name}.json")
            save_evaluation_results(eval_results, eval_path)

            return eval_results

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

    def run_full_pipeline(self, dataset_name: str = None, custom_data_path: str = None):
        """
        Run the complete training and evaluation pipeline.

        Args:
            dataset_name: Name of dataset from Hub
            custom_data_path: Path to custom dataset
        """
        try:
            # Step 1: Prepare data
            self.prepare_data(dataset_name, custom_data_path)

            # Step 2: Setup model
            self.setup_model()

            # Step 3: Setup trainer
            self.setup_trainer()

            # Step 4: Train model
            train_metrics = self.train()

            # Step 5: Evaluate model
            eval_metrics = self.evaluate()

            # Step 6: Save final results
            final_results = {
                "training_metrics": train_metrics,
                "evaluation_metrics": eval_metrics,
                "model_info": self.model_manager.get_model_info() if self.model_manager else {},
                "config": self.config.to_dict()
            }

            results_path = os.path.join(self.config.training.reports_dir, "final_results.json")
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)

            logger.info("Full pipeline completed successfully")
            return final_results

        except Exception as e:
            logger.error(f"Error in full pipeline: {str(e)}")
            raise


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna."""

    def __init__(self, config: Config, dataset_name: str = None):
        """
        Initialize hyperparameter tuner.

        Args:
            config: Base configuration
            dataset_name: Dataset to use for tuning
        """
        self.config = config
        self.dataset_name = dataset_name or config.data.dataset_name
        self.best_params = None
        self.best_score = None

    def objective(self, trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            F1 score to maximize
        """
        # Suggest hyperparameters
        learning_rate = trial.suggest_float(
            "learning_rate", 
            self.config.hyperparameter.learning_rate_range[0],
            self.config.hyperparameter.learning_rate_range[1],
            log=True
        )
        batch_size = trial.suggest_categorical(
            "batch_size", 
            self.config.hyperparameter.batch_size_options
        )
        num_epochs = trial.suggest_int(
            "num_epochs",
            self.config.hyperparameter.epochs_range[0],
            self.config.hyperparameter.epochs_range[1]
        )

        # Update config with suggested parameters
        trial_config = Config()
        trial_config.__dict__.update(self.config.__dict__)
        trial_config.training.learning_rate = learning_rate
        trial_config.training.train_batch_size = batch_size
        trial_config.training.eval_batch_size = batch_size
        trial_config.training.num_train_epochs = num_epochs

        # Run training with trial parameters
        trainer = TextClassificationTrainer(trial_config)

        try:
            trainer.prepare_data(self.dataset_name)
            trainer.setup_model()
            trainer.setup_trainer()
            trainer.train()
            eval_results = trainer.evaluate()

            # Return F1 score for optimization
            return eval_results.get("eval_f1", 0.0)

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0

    def tune(self) -> Dict:
        """
        Run hyperparameter tuning.

        Returns:
            Best parameters and score
        """
        logger.info("Starting hyperparameter tuning...")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.config.hyperparameter.n_trials)

        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best F1 score: {self.best_score}")

        # Save tuning results
        tuning_results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(study.trials),
            "study_results": [
                {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params
                }
                for trial in study.trials
            ]
        }

        tuning_path = os.path.join(self.config.training.reports_dir, "hyperparameter_tuning.json")
        with open(tuning_path, 'w') as f:
            json.dump(tuning_results, f, indent=2)

        return tuning_results


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Text Classification Training")
    parser.add_argument("--config", type=str, default="default", 
                       help="Configuration type (default, imdb, ag_news, yelp, multilingual)")
    parser.add_argument("--dataset", type=str, help="Dataset name or path")
    parser.add_argument("--custom_data", type=str, help="Path to custom dataset")
    parser.add_argument("--tune_hyperparams", action="store_true", 
                       help="Run hyperparameter tuning")
    parser.add_argument("--eval_only", action="store_true", 
                       help="Only run evaluation on existing model")
    parser.add_argument("--output_dir", type=str, help="Custom output directory")

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Override config with command line arguments
    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.output_dir:
        config.training.output_dir = args.output_dir

    # Create output directories
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.training.reports_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(config.training.reports_dir, "config.json")
    config.save_config(config_path)

    try:
        if args.tune_hyperparams:
            # Run hyperparameter tuning
            tuner = HyperparameterTuner(config, args.dataset)
            tuning_results = tuner.tune()

            # Update config with best parameters and run final training
            config.training.learning_rate = tuning_results["best_params"]["learning_rate"]
            config.training.train_batch_size = tuning_results["best_params"]["batch_size"]
            config.training.eval_batch_size = tuning_results["best_params"]["batch_size"]
            config.training.num_train_epochs = tuning_results["best_params"]["num_epochs"]

            logger.info("Running final training with best parameters...")

        # Initialize trainer
        trainer = TextClassificationTrainer(config)

        if args.eval_only:
            # Load existing model and evaluate
            trainer.prepare_data(args.dataset, args.custom_data)
            trainer.model_manager = ModelManager(config.model.model_name, config.model.num_labels)
            trainer.model_manager.load_model(config.training.output_dir)
            trainer.model = trainer.model_manager.model
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")
        else:
            # Run full training pipeline
            results = trainer.run_full_pipeline(args.dataset, args.custom_data)
            logger.info("Training pipeline completed successfully")

            # Log final metrics
            if "evaluation_metrics" in results:
                eval_metrics = results["evaluation_metrics"]
                logger.info(f"Final F1 Score: {eval_metrics.get('eval_f1', 'N/A')}")
                logger.info(f"Final Accuracy: {eval_metrics.get('eval_accuracy', 'N/A')}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

    logger.info("Script completed successfully")


if __name__ == "__main__":
    main()
