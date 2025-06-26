# Text Classification Pipeline with Hugging Face Transformers

A complete text classification pipeline using Hugging Face Transformers, implementing preprocessing, model fine-tuning, evaluation, and deployment utilities. This project demonstrates best practices for NLP and showcases how to build an end-to-end pipeline for text classification tasks.

## Features

- **Comprehensive Pipeline**: Complete workflow from data preprocessing to model deployment
- **Modular Design**: Well-structured code organization with separated components
- **Multiple Datasets**: Support for IMDB, AG News, Yelp Reviews, and custom datasets
- **Multilingual Support**: Cross-lingual transfer learning and multilingual model fine-tuning
- **Advanced Evaluation**: F1, precision, recall metrics with visualizations and error analysis
- **Hyperparameter Tuning**: Automated tuning with Optuna
- **Model Comparison**: Tools to compare multiple model architectures
- **Deployment Ready**: Inference API code for model deployment

## Project Structure

```
/
├── notebooks/                         # Jupyter notebooks
│   ├── data_exploration.ipynb         # Dataset analysis
│   ├── model_training.ipynb           # Interactive training
│   └── evaluation_analysis.ipynb      # Results visualization
│
├── src/                               # Source code
│   ├── train_model.py                 # Main training script
│   ├── data_preprocessing.py          # Text preprocessing utilities
│   ├── model_utils.py                 # Model handling utilities
│   └── config.py                      # Configuration management
│
├── models/                            # Model storage
│   ├── trained_model/                 # Trained model weights
│   └── tokenizer/                     # Saved tokenizer files
│
├── reports/                           # Analysis reports
│   ├── model_report.md                # Model architecture report
│   ├── evaluation_metrics.json        # Detailed metrics
│   └── confusion_matrix.png           # Visualization outputs
│
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── submission.md                      # Approach and key learnings
├── train.py                           # Training script entry point
└── .gitignore                         # Git ignore configuration
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository

   ```bash
   git clone
   cd text-classification-pipeline
   ```

2. Set up a virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

#### Data Exploration

Explore and analyze the dataset:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

#### Training a Model

Train a model using the default configuration (IMDB dataset with DistilBERT):

```bash
python train.py
```

With custom configuration:

```bash
python train.py --config imdb  # Predefined configs: imdb, ag_news, yelp, multilingual
```

With hyperparameter tuning:

```bash
python train.py --tune_hyperparams
```

#### Evaluating a Model

Run evaluation on a trained model:

```bash
python train.py --eval_only
```

Visualize and analyze results:

```bash
jupyter notebook notebooks/evaluation_analysis.ipynb
```

## Supported Datasets

The pipeline supports the following datasets out of the box:

- **IMDB Movie Reviews**: Binary sentiment classification (positive/negative)
- **AG News**: Topic classification with 4 classes
- **Yelp Reviews**: Binary sentiment classification
- **Custom Datasets**: Support for custom datasets in CSV, JSON, or TSV formats

## 🔧 Multilingual Extension

The pipeline includes multilingual support with:

- Cross-lingual transfer learning
- Language detection and preprocessing
- Multilingual model training with XLM-RoBERTa or mBERT
- Zero-shot evaluation on new languages

## Advanced Evaluation

The evaluation framework includes:

- Precision, recall, and F1 metrics
- Confusion matrix visualization
- ROC curve analysis for binary classification
- Detailed error analysis
- Per-class performance metrics
- Failure case studies

## Model Comparison

Compare multiple models:

```bash
python -m src.train_model --config imdb
python -m src.train_model --config imdb --model_name bert-base-uncased
jupyter notebook notebooks/evaluation_analysis.ipynb  # For comparison visualization
```

## Deployment

The project includes a deployment-ready inference pipeline:

- FastAPI-based REST API
- Streamlit interactive web application
- Batch prediction capabilities
- Production-ready model serving

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the Transformers library
- The creators of the datasets used in this project
- The open-source community for various tools and libraries
