"""
Data preprocessing module for text classification pipeline.
Handles text cleaning, tokenization, dataset preparation, and multilingual processing.
"""

import re
import string
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from sklearn.model_selection import train_test_split
import spacy
from langdetect import detect, DetectorFactory
import warnings

# Set seed for reproducible language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities for cleaning and normalizing text data."""

    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False,
                 remove_extra_whitespace: bool = True,
                 expand_contractions: bool = True):
        """
        Initialize text preprocessor.

        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            remove_extra_whitespace: Whether to remove extra whitespace
            expand_contractions: Whether to expand contractions
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.expand_contractions = expand_contractions

        # Contraction mapping
        self.contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "i'm": "i am", "you're": "you are",
            "it's": "it is", "that's": "that is", "what's": "what is",
            "where's": "where is", "how's": "how is", "here's": "here is",
            "there's": "there is", "who's": "who is", "let's": "let us"
        }

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Expand contractions
        if self.expand_contractions:
            for contraction, expansion in self.contractions.items():
                text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts."""
        return [self.clean_text(text) for text in texts]


class MultilingualProcessor:
    """Utilities for handling multilingual text data."""

    def __init__(self):
        """Initialize multilingual processor."""
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ar']
        self.spacy_models = {}
        self._load_spacy_models()

    def _load_spacy_models(self):
        """Load available spaCy models for supported languages."""
        model_mapping = {
            'en': 'en_core_web_sm',
            'es': 'es_core_news_sm',
            'fr': 'fr_core_news_sm',
            'de': 'de_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm',
            'ru': 'ru_core_news_sm',
            'zh': 'zh_core_web_sm'
        }

        for lang, model_name in model_mapping.items():
            try:
                self.spacy_models[lang] = spacy.load(model_name)
            except OSError:
                logger.warning(f"spaCy model {model_name} not found for language {lang}")

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.

        Args:
            text: Input text

        Returns:
            Detected language code
        """
        try:
            return detect(text)
        except:
            return 'en'  # Default to English

    def detect_languages_batch(self, texts: List[str]) -> List[str]:
        """Detect languages for a batch of texts."""
        languages = []
        for text in texts:
            lang = self.detect_language(text)
            languages.append(lang)
        return languages

    def preprocess_multilingual_text(self, text: str, language: str = None) -> str:
        """
        Preprocess text based on detected or specified language.

        Args:
            text: Input text
            language: Language code (optional, will detect if not provided)

        Returns:
            Preprocessed text
        """
        if language is None:
            language = self.detect_language(text)

        if language in self.spacy_models:
            nlp = self.spacy_models[language]
            doc = nlp(text)
            # Basic preprocessing with spaCy
            tokens = [token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct and not token.is_space]
            return ' '.join(tokens)
        else:
            # Fallback to basic preprocessing
            preprocessor = TextPreprocessor()
            return preprocessor.clean_text(text)


class DatasetLoader:
    """Utility class for loading and preparing datasets for text classification."""

    def __init__(self, config):
        """
        Initialize dataset loader.

        Args:
            config: Configuration object
        """
        self.config = config
        self.text_preprocessor = TextPreprocessor()
        self.multilingual_processor = MultilingualProcessor()

    def load_dataset_from_hub(self, dataset_name: str, split: str = None) -> Union[Dataset, DatasetDict]:
        """
        Load dataset from Hugging Face Hub.

        Args:
            dataset_name: Name of the dataset
            split: Specific split to load (optional)

        Returns:
            Loaded dataset
        """
        try:
            if split:
                dataset = load_dataset(dataset_name, split=split)
            else:
                dataset = load_dataset(dataset_name)

            logger.info(f"Successfully loaded dataset: {dataset_name}")
            return dataset

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise

    def load_custom_dataset(self, 
                          filepath: str, 
                          text_column: str, 
                          label_column: str,
                          format: str = 'csv') -> Dataset:
        """
        Load custom dataset from file.

        Args:
            filepath: Path to dataset file
            text_column: Name of text column
            label_column: Name of label column
            format: File format ('csv', 'json', 'tsv')

        Returns:
            Dataset object
        """
        if format == 'csv':
            df = pd.read_csv(filepath)
        elif format == 'tsv':
            df = pd.read_csv(filepath, sep='\t')
        elif format == 'json':
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Validate required columns
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Required columns {text_column} or {label_column} not found")

        dataset = Dataset.from_pandas(df[[text_column, label_column]])
        return dataset

    def prepare_dataset(self, 
                       dataset: Union[Dataset, DatasetDict],
                       text_column: str,
                       label_column: str,
                       validation_split: float = 0.2,
                       test_split: float = 0.1) -> DatasetDict:
        """
        Prepare dataset with train/validation/test splits.

        Args:
            dataset: Input dataset
            text_column: Name of text column
            label_column: Name of label column
            validation_split: Validation split ratio
            test_split: Test split ratio

        Returns:
            DatasetDict with train/validation/test splits
        """
        if isinstance(dataset, DatasetDict):
            # Dataset already has splits
            if 'train' in dataset and 'test' in dataset:
                train_dataset = dataset['train']
                test_dataset = dataset['test']

                # Create validation split from training data if not exists
                if 'validation' not in dataset:
                    train_val = train_dataset.train_test_split(test_size=validation_split, seed=42)
                    train_dataset = train_val['train']
                    val_dataset = train_val['test']
                else:
                    val_dataset = dataset['validation']

                return DatasetDict({
                    'train': train_dataset,
                    'validation': val_dataset,
                    'test': test_dataset
                })

        # Single dataset - create splits
        df = dataset.to_pandas()

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_split, stratify=df[label_column], random_state=42
        )

        # Second split: separate validation set from training
        train_df, val_df = train_test_split(
            train_val_df, test_size=validation_split/(1-test_split), 
            stratify=train_val_df[label_column], random_state=42
        )

        return DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })

    def preprocess_dataset(self, 
                          dataset: DatasetDict,
                          text_column: str = 'text',
                          enable_multilingual: bool = False) -> DatasetDict:
        """
        Preprocess text data in dataset.

        Args:
            dataset: Input dataset
            text_column: Name of text column
            enable_multilingual: Whether to apply multilingual preprocessing

        Returns:
            Preprocessed dataset
        """
        def preprocess_function(examples):
            texts = examples[text_column]

            if enable_multilingual:
                # Detect languages and preprocess accordingly
                processed_texts = []
                for text in texts:
                    processed_text = self.multilingual_processor.preprocess_multilingual_text(text)
                    processed_texts.append(processed_text)
            else:
                # Standard preprocessing
                processed_texts = self.text_preprocessor.process_batch(texts)

            examples[text_column] = processed_texts
            return examples

        # Apply preprocessing to all splits
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.config.data.preprocessing_num_workers
        )

        return processed_dataset


class TokenizerWrapper:
    """Wrapper for Hugging Face tokenizers with additional functionality."""

    def __init__(self, model_name: str, max_length: int = 512):
        """
        Initialize tokenizer wrapper.

        Args:
            model_name: Name of the model/tokenizer
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_dataset(self, 
                        dataset: DatasetDict,
                        text_column: str = 'text',
                        label_column: str = 'label') -> DatasetDict:
        """
        Tokenize dataset for model training.

        Args:
            dataset: Input dataset
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            # Tokenize texts
            tokenized = self.tokenizer(
                examples[text_column],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors=None
            )

            # Ensure labels are properly formatted
            tokenized['labels'] = examples[label_column]

            return tokenized

        # Apply tokenization to all splits
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        return tokenized_dataset

    def save_tokenizer(self, save_path: str):
        """Save tokenizer to disk."""
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Tokenizer saved to {save_path}")

    @classmethod
    def load_tokenizer(cls, load_path: str, max_length: int = 512):
        """Load tokenizer from disk."""
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        wrapper = cls.__new__(cls)
        wrapper.tokenizer = tokenizer
        wrapper.max_length = max_length
        wrapper.model_name = load_path
        return wrapper


def analyze_dataset(dataset: Union[Dataset, DatasetDict], 
                   text_column: str = 'text',
                   label_column: str = 'label') -> Dict:
    """
    Analyze dataset characteristics and statistics.

    Args:
        dataset: Input dataset
        text_column: Name of text column
        label_column: Name of label column

    Returns:
        Dictionary containing dataset analysis
    """
    analysis = {}

    if isinstance(dataset, DatasetDict):
        for split_name, split_data in dataset.items():
            analysis[split_name] = _analyze_single_dataset(split_data, text_column, label_column)
    else:
        analysis['dataset'] = _analyze_single_dataset(dataset, text_column, label_column)

    return analysis


def _analyze_single_dataset(dataset: Dataset, 
                           text_column: str,
                           label_column: str) -> Dict:
    """Analyze a single dataset split."""
    df = dataset.to_pandas()

    analysis = {
        'num_samples': len(df),
        'num_labels': df[label_column].nunique(),
        'label_distribution': df[label_column].value_counts().to_dict(),
        'avg_text_length': df[text_column].str.len().mean(),
        'max_text_length': df[text_column].str.len().max(),
        'min_text_length': df[text_column].str.len().min(),
        'avg_word_count': df[text_column].str.split().str.len().mean(),
        'max_word_count': df[text_column].str.split().str.len().max(),
        'min_word_count': df[text_column].str.split().str.len().min()
    }

    return analysis


# Helper function to create data loaders
def create_data_pipeline(config, dataset_name: str = None, custom_data_path: str = None):
    """
    Create complete data processing pipeline.

    Args:
        config: Configuration object
        dataset_name: Name of dataset to load from Hub
        custom_data_path: Path to custom dataset file

    Returns:
        Tuple of (processed_dataset, tokenizer_wrapper, analysis)
    """
    loader = DatasetLoader(config)

    # Load dataset
    if dataset_name:
        raw_dataset = loader.load_dataset_from_hub(dataset_name)
    elif custom_data_path:
        raw_dataset = loader.load_custom_dataset(
            custom_data_path, 
            config.data.text_column, 
            config.data.label_column
        )
    else:
        raise ValueError("Either dataset_name or custom_data_path must be provided")

    # Prepare dataset splits
    dataset = loader.prepare_dataset(
        raw_dataset,
        config.data.text_column,
        config.data.label_column
    )

    # Preprocess text
    dataset = loader.preprocess_dataset(
        dataset,
        config.data.text_column,
        config.multilingual.enable_multilingual
    )

    # Initialize tokenizer
    tokenizer_wrapper = TokenizerWrapper(config.model.model_name, config.model.max_length)

    # Tokenize dataset
    tokenized_dataset = tokenizer_wrapper.tokenize_dataset(
        dataset,
        config.data.text_column,
        config.data.label_column
    )

    # Analyze dataset
    analysis = analyze_dataset(dataset, config.data.text_column, config.data.label_column)

    return tokenized_dataset, tokenizer_wrapper, analysis
