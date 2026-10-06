# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "keras",
#     "keras-hub",
#     "torch",
#     "datasets",
#     "numpy",
# ]
# ///



"""
A script to fine-tune T5-Gemma model to generate fixed docids using Keras.
"""

# Set Keras backend
import os
os.environ["KERAS_BACKEND"] = "torch"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import logging
import random
import numpy as np
import keras
from functools import partial

from keras_hub.models import T5GemmaSeq2SeqLM
from keras.utils import Sequence
from datasets import load_from_disk
from utils import download_corpus
import torch
from torch.utils.data import Dataset, DataLoader

# Configure logging
logger = logging.getLogger(__name__)


# Configure mixed precision
mixed_precision = "bfloat16"  # or "mixed_bfloat16" or "float32"
logger.info(f"Configuring mixed precision: {mixed_precision}")
keras.mixed_precision.set_global_policy(mixed_precision)
logger.info(f"Mixed precision policy set: {keras.mixed_precision.global_policy()}")

if torch.cuda.is_available():
    # Enable cudnn benchmarking for optimized kernels
    torch.backends.cudnn.benchmark = True
    # Faster bfloat16
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # NEW!


class CustomT5Gemma(T5GemmaSeq2SeqLM):
    """Custom T5-Gemma model with full control over training loop."""
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        
        # Clear gradients from previous step
        self.zero_grad()
        
        # Compute loss
        y_pred = self(x, training=True)
        # loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        
        # Backward pass
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        
        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """Validation step (no gradient updates)."""
        # Unpack the data
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        
        # Forward pass (no training mode)
        y_pred = self(x, training=False)
        
        # Compute loss
        # loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}


class PyTorchDocIDDataset(Dataset):
    """PyTorch Dataset for docid generation - for memory comparison testing.
    
    Uses memory-mapped HuggingFace datasets to load data on-demand.
    
    Args:
        corpus: Memory-mapped HuggingFace dataset with passages
        docids: Memory-mapped HuggingFace dataset with docids
        indices: List of indices to use for this split
        preprocessor: Model's preprocessor/tokenizer
        max_length: Maximum length of input sequences
    """
    
    def __init__(self, corpus, docids, indices, preprocessor, max_length=200):
        self.corpus = corpus
        self.docids = docids
        self.indices = indices
        self.preprocessor = preprocessor
        self.max_length = max_length
    
    def __len__(self):
        """Total number of samples"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get one sample.
        
        Returns:
            dict: Preprocessed sample with encoder/decoder inputs
        """
        actual_idx = self.indices[idx]
        corpus_item = self.corpus[actual_idx]
        docid_item = self.docids[actual_idx]
        
        # Verify IDs match (sanity check)
        if int(docid_item['passage_id']) != int(corpus_item['passage_id']):
            logger.warning(f"Mismatch at index {actual_idx}: docid_id={docid_item['passage_id']}, corpus_id={corpus_item['passage_id']}")
        
        encoder_text = f"Passage: {corpus_item['passage'][:self.max_length]}"
        decoder_text = str(docid_item['docid']).strip()
        
        return {
            'encoder_text': encoder_text,
            'decoder_text': decoder_text
        }


def collate_fn(batch, preprocessor):
    """Custom collate function for PyTorch DataLoader.
    
    Args:
        batch: List of dicts with 'encoder_text' and 'decoder_text'
        preprocessor: Model's preprocessor/tokenizer
    
    Returns:
        dict: Preprocessed batch ready for training
    """
    encoder_texts = [item['encoder_text'] for item in batch]
    decoder_texts = [item['decoder_text'] for item in batch]
    
    preprocessed = preprocessor(
        x={
            "encoder_text": encoder_texts,
            "decoder_text": decoder_texts
        }
    )
    
    return preprocessed


class DocIDGenerator(keras.utils.PyDataset):
    """Memory-efficient data generator for T5-Gemma docid generation training.
    
    Uses memory-mapped HuggingFace datasets to load data on-demand, keeping only
    batch indices in RAM.
    
    Args:
        corpus: Memory-mapped HuggingFace dataset with passages
        docids: Memory-mapped HuggingFace dataset with docids
        indices: List of indices to use for this split
        preprocessor: Model's preprocessor/tokenizer
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle indices after each epoch
    """
    
    def __init__(self, corpus, docids, indices, preprocessor, batch_size=4, max_length=200, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.corpus = corpus
        self.docids = docids
        self.indices = indices.copy()
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle_data = shuffle
        
        if self.shuffle_data:
            random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, batch_idx):
        """Get one batch of preprocessed data.
        
        Returns:
            dict: Preprocessed batch with keys like 'encoder_token_ids', 
                  'encoder_padding_mask', 'decoder_token_ids', 'decoder_padding_mask'.
                  Must return dict directly (not wrapped in tuple) for Keras to handle correctly.
        """
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        encoder_texts = []
        decoder_texts = []
        
        for idx in batch_indices:
            corpus_item = self.corpus[idx]
            docid_item = self.docids[idx]
            
            # Verify IDs match (sanity check)
            if int(docid_item['passage_id']) != int(corpus_item['passage_id']):
                logger.warning(f"Mismatch at index {idx}: docid_id={docid_item['passage_id']}, corpus_id={corpus_item['passage_id']}")
                continue
            
            encoder_texts.append(f"Passage: {corpus_item['passage'][:self.max_length]}")
            decoder_texts.append(str(docid_item['docid']).strip())
        
        # Tokenize with the model's preprocessor
        preprocessed = self.preprocessor(
            x={
                "encoder_text": encoder_texts,
                "decoder_text": decoder_texts
            }
        )
        
        return preprocessed
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch for better training"""
        if self.shuffle_data:
            random.shuffle(self.indices)


class DocIDGeneratorSequence(Sequence):
    """Memory-efficient data generator for T5-Gemma docid generation training using keras.utils.Sequence.
    
    Args:
        corpus: Memory-mapped HuggingFace dataset with passages
        docids: Memory-mapped HuggingFace dataset with docids
        indices: List of indices to use for this split
        preprocessor: Model's preprocessor/tokenizer
        batch_size: Number of samples per batch
        max_length: Maximum length of input sequences
        shuffle: Whether to shuffle indices after each epoch
    """
    
    def __init__(self, corpus, docids, indices, preprocessor, batch_size=4, max_length=200, shuffle=True):
        self.corpus = corpus
        self.docids = docids
        self.indices = np.array(indices)
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, batch_idx):
        """Get one batch of preprocessed data.
        """
        # Calculate batch slice indices
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        encoder_texts = []
        decoder_texts = []
        
        for idx in batch_indices:
            corpus_item = self.corpus[int(idx)]
            docid_item = self.docids[int(idx)]
            
            # Verify IDs match (sanity check)
            if int(docid_item['passage_id']) != int(corpus_item['passage_id']):
                logger.warning(f"Mismatch at index {idx}: docid_id={docid_item['passage_id']}, corpus_id={corpus_item['passage_id']}")
                continue
            
            encoder_texts.append(f"Passage: {corpus_item['passage'][:self.max_length]}")
            decoder_texts.append(str(docid_item['docid']).strip())
        
        preprocessed = self.preprocessor(
            x={
                "encoder_text": encoder_texts,
                "decoder_text": decoder_texts
            }
        )
        
        return preprocessed
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_and_create_datasets(docid_path: str, dataset_name: str, preprocessor,
                             subset: str = None, split: str = "train", 
                             max_samples: int = None, max_length: int = 200, batch_size: int = 4, 
                             train_val_split: float = 0.8, use_pytorch_dataloader: bool = False,
                             use_sequence: bool = False):
    """Load document-to-docid mappings and create memory-efficient data loaders.
    
    Uses memory-mapped HuggingFace datasets - data is NOT loaded into RAM.
    Only batch indices are stored in memory.
    
    Args:
        docid_path: Path to docids HuggingFace dataset (saved with save_to_disk)
        dataset_name: Name of the corpus dataset (e.g., 'mteb/msmarco')
        preprocessor: Model's preprocessor/tokenizer for text tokenization
        subset: Subset of the dataset (e.g., 'corpus')
        split: Split of the dataset (e.g., 'train', 'corpus')
        max_samples: Maximum number of samples to load (None = all)
        max_length: Maximum length of input sequences
        batch_size: Number of samples per batch
        train_val_split: Fraction of data for training (rest split between val/test)
        use_pytorch_dataloader: If True, use PyTorch DataLoader instead of Keras generators
        use_sequence: If True, use DocIDGeneratorSequence (keras.utils.Sequence) instead of DocIDGenerator (keras.utils.PyDataset)
    
    Returns:
        tuple: (train_loader, val_loader, test_examples)
            - train_loader: Data loader for training (DocIDGenerator, DocIDGeneratorSequence, or DataLoader)
            - val_loader: Data loader for validation (DocIDGenerator, DocIDGeneratorSequence, or DataLoader)
            - test_examples: List of dicts for evaluation
    """
    
    logger.info("Loading corpus (memory-mapped)...")
    corpus = download_corpus(dataset_name, subset, split, streaming=False)
    logger.info(f"Corpus size: {len(corpus):,} documents (memory-mapped)")

    logger.info("Loading docids dataset (memory-mapped)...")
    docids = load_from_disk(docid_path)
    logger.info(f"Docids size: {len(docids):,} entries (memory-mapped)")
    
    if len(docids) != len(corpus):
        logger.warning(f"Size mismatch - docids: {len(docids):,}, corpus: {len(corpus):,}")

    total_samples = min(len(docids), max_samples) if max_samples else len(docids)
    logger.info(f"Using {total_samples:,} samples")
    
    # Calculate split indices: 72% train, 18% val, 10% test
    train_end = int(total_samples * train_val_split * 0.9)
    val_end = int(total_samples * 0.9)
    
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, total_samples))
    
    logger.info(f"Dataset split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")
    
    # Create data loaders based on selected backend
    if use_pytorch_dataloader:
        logger.info("Creating PyTorch DataLoader (for memory testing)...")
        
        # Create PyTorch datasets
        train_dataset = PyTorchDocIDDataset(
            corpus, docids, train_idx,
            preprocessor=preprocessor,
            max_length=max_length
        )
        
        val_dataset = PyTorchDocIDDataset(
            corpus, docids, val_idx,
            preprocessor=preprocessor,
            max_length=max_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, preprocessor=preprocessor),
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with Keras
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, preprocessor=preprocessor),
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"PyTorch DataLoaders created")
        logger.info(f"Training batches per epoch: {len(train_loader):,}")
        logger.info(f"Validation batches per epoch: {len(val_loader):,}")
        
    elif use_sequence:
        logger.info("Creating Keras Sequence generators (memory-efficient)...")
        
        # Create Keras Sequence generators (only keeps current batch in memory)
        train_loader = DocIDGeneratorSequence(
            corpus, docids, train_idx, 
            preprocessor=preprocessor,
            batch_size=batch_size, 
            max_length=max_length,
            shuffle=True
        )
        
        val_loader = DocIDGeneratorSequence(
            corpus, docids, val_idx,
            preprocessor=preprocessor,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False
        )
        
        logger.info(f"Keras Sequence generators created")
        logger.info(f"Training batches per epoch: {len(train_loader):,}")
        logger.info(f"Validation batches per epoch: {len(val_loader):,}")
        
    else:
        logger.info("Creating Keras PyDataset generators...")
        
        # Create Keras PyDataset generators
        train_loader = DocIDGenerator(
            corpus, docids, train_idx, 
            preprocessor=preprocessor,
            batch_size=batch_size, 
            max_length=max_length,
            shuffle=True
        )
        
        val_loader = DocIDGenerator(
            corpus, docids, val_idx,
            preprocessor=preprocessor,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False
        )
        
        logger.info(f"Keras PyDataset generators created")
        logger.info(f"Training batches per epoch: {len(train_loader):,}")
        logger.info(f"Validation batches per epoch: {len(val_loader):,}")
    
    # Collect test examples for evaluation
    logger.info("Collecting test samples...")
    test_examples = []
    for idx in test_idx:
        corpus_item = corpus[idx]
        docid_item = docids[idx]
        test_examples.append({
            "input_text": f"Passage: {corpus_item['passage'][:max_length]}",
            "target_text": str(docid_item['docid']).strip()
        })
    
    logger.info(f"Created {len(test_examples):,} test examples")
    logger.info(f"Memory: Only indices (~40KB) + current batch in RAM")
    
    return train_loader, val_loader, test_examples


def docid_accuracy(y_true, y_pred):
    """Custom metric for exact docid match"""
    return keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def evaluate_retrieval(model, test_examples, verbose=False):
    """
    Evaluate exact match retrieval
    Will compute Hits@K and MRR metrics in the future when model supports beam search.
    """
    exact_matches = 0

    for idx, example in enumerate(test_examples):
        if idx>=5:
            break
        input_text = example["input_text"]
        expected_docid = example["target_text"]
        
        # Generate predicted docid
        predicted_docid = model.generate(input_text, max_length=40)

        if verbose and idx < 5:
            logger.info(f"Test Example {idx+1}:")
            logger.info(f"  Input: {input_text[:80]}...")
            logger.info(f"  Expected: {expected_docid}")
            logger.info(f"  Predicted: {predicted_docid}")
        
        if predicted_docid == expected_docid:
            exact_matches += 1
    
    # Calculate final metrics
    total_examples = len(test_examples)
    accuracy = exact_matches / total_examples if total_examples > 0 else 0.0

    logger.info("Retrieval Evaluation Results:")
    logger.info(f"  Total Examples: {total_examples}")
    logger.info(f"  Exact Match Accuracy: {accuracy:.4f}")

    return accuracy


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune T5-Gemma model for autoregressive docid generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model-preset",
        type=str,
        default="t5gemma_b_b_prefixlm_it",
        help="T5-Gemma model preset to use"
    )
    parser.add_argument(
        "--use-custom-fit",
        action="store_true",
        help="Use custom training loop instead of Keras fit"
    )
    
    # Data arguments
    parser.add_argument(
        "--docid-path",
        type=str,
        default="docids/mteb_msmarco_uuid_dataset",
        help="Path to docids HuggingFace dataset (saved with save_to_disk)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mteb/msmarco",
        help="Name of the corpus dataset to load"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="corpus",
        help="Subset of the dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="corpus",
        help="Split of the dataset (train/test/corpus)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (None = all)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum length of input sequences"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (rest split between val/test)"
    )
    
    # Mixed precision arguments
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="mixed_bfloat16",
        choices=["float32",  "float16", "mixed_float16", "bfloat16", "mixed_bfloat16"],
        help="Enable mixed precision training (None = disabled, mixed_float16 = float16, mixed_bfloat16 = bfloat16)"
    )
    
    # Data loader arguments
    parser.add_argument(
        "--use-pytorch-dataloader",
        action="store_true",
        help="Use PyTorch DataLoader instead of Keras generators (for memory testing)"
    )
    parser.add_argument(
        "--use-sequence",
        action="store_true",
        help="Use keras.utils.Sequence instead of keras.utils.PyDataset (memory-efficient, keeps only current batch in RAM)"
    )
    
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    
    # Configure logging for script execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("T5-Gemma Docid Generation Fine-tuning")
    logger.info(f"Keras backend: {keras.backend.backend()}")
    logger.info(f"Configuration: {vars(args)}")
    
    # Configure mixed precision
    logger.info(f"Configuring mixed precision: {args.mixed_precision}")
    keras.mixed_precision.set_global_policy(args.mixed_precision)
    logger.info(f"Mixed precision policy set: {keras.mixed_precision.global_policy()}")
    
    logger.info("Loading T5-Gemma model...")
    try:
        available_presets = list(T5GemmaSeq2SeqLM.presets.keys())
        logger.info(f"Available presets: {available_presets[:5]}")
        
        logger.info(f"Using preset: {args.model_preset}")
        
        # Load model with custom training loop
        if args.use_custom_fit:
            logger.info("Loading model with custom train_step()...")
            model = CustomT5Gemma.from_preset(args.model_preset)
        else:
            logger.info("Loading model with standard train_step()...")
            model = T5GemmaSeq2SeqLM.from_preset(args.model_preset)
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Preprocessor: {type(model.preprocessor).__name__}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Exiting...")
        exit(1)
    
    logger.info("Loading docid data and creating data loaders...")
    train_loader, val_loader, test_examples = load_and_create_datasets(
        docid_path=args.docid_path,
        dataset_name=args.dataset_name,
        preprocessor=model.preprocessor,
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_val_split=args.train_val_split,
        use_pytorch_dataloader=args.use_pytorch_dataloader,
        use_sequence=args.use_sequence
    )

    # logger.info("Evaluating before fine-tuning...")
    # evaluate_retrieval(model, test_examples, verbose=True)

    logger.info("Compiling model for fine-tuning...")
    optimizer = keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=1.0
        )
    
    if args.mixed_precision:
        logger.info(f"Optimizer will use compute dtype: {keras.mixed_precision.global_policy().compute_dtype}")
        logger.info(f"Optimizer will use variable dtype: {keras.mixed_precision.global_policy().variable_dtype}")
    
    model.compile(
        optimizer=optimizer,
        metrics=[docid_accuracy],
        jit_compile=True
    )

    logger.info("Starting fine-tuning...")
    try:
        # CRITICAL: Do NOT pass validation_data to fit()!
        # 
        # KerasHub's evaluate() override tries to convert data to tf.data.Dataset,
        # which fails when using PyTorch backend, because the preprocessor returns 
        # PyTorch tensors that cannot be converted to TensorFlow datasets.
        # 
        # Instead, use a custom callback to manually run validation after each epoch.
        history = keras.Model.fit(
            model,
            train_loader,
            # validation_data=val_loader,
            epochs=args.epochs,
            verbose=1
        )
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Evaluating fine-tuned model on test data...")
    evaluate_retrieval(model, test_examples, verbose=True)
    
    logger.info("Script completed!")