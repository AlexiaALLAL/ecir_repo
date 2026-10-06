# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "keras",
#     "keras-hub",
#     "torch",
#     "datasets",
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

import keras
from keras_hub.models import T5GemmaSeq2SeqLM
import torch
from torch.utils.data import Dataset, DataLoader
from generate_docids import download_corpus
from datasets import load_from_disk


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )


# Configure mixed precision by default (can be updated afterwards)
precision = "mixed_bfloat16"  # or "mixed_bfloat16" or "float32"
logger.info(f"Configuring mixed precision: {precision}")
keras.mixed_precision.set_global_policy(precision)
logger.info(f"Mixed precision policy set: {keras.mixed_precision.global_policy()}")


class PyTorchDocIDDataset(Dataset):
    """PyTorch Dataset for DocID generation task"""
    
    def __init__(self, docid_dataset, corpus_dataset, max_samples: int = None, max_length: int = 200):
        """
        Args:
            docid_dataset: HuggingFace dataset containing docids
            corpus_dataset: HuggingFace dataset containing passages
            max_samples: Maximum number of samples to use (None for all)
            max_length: Maximum length for passage text
        """
        self.docid_dataset = docid_dataset
        self.corpus_dataset = corpus_dataset
        self.max_length = max_length

        assert len(docid_dataset) == len(corpus_dataset), "DocID and corpus datasets must have the same length."
        
        # Determine the actual length
        self.length = min(len(docid_dataset), max_samples) if max_samples else len(docid_dataset)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Get a single example"""
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.length}")
        
        # Sanity check: verify passage_id matches
        docid_item = self.docid_dataset[idx]
        corpus_item = self.corpus_dataset[idx]
        
        if int(docid_item['passage_id']) != int(corpus_item['passage_id']):
            raise ValueError(f"Passage ID mismatch at index {idx}: {docid_item['passage_id']} vs {corpus_item['passage_id']}")
        
        text = corpus_item['passage']
        docid = docid_item['docid']
        
        return {
            "input_text": f"Passage: {text[:self.max_length]}",  # truncate long documents
            "target_text": str(docid).strip()
        }


def load_docid_data(docid_path: str, dataset_name: str, subset: str = None, split: str = "train", 
                    max_samples: int = 200, max_length: int = 200, batch_size: int = 4,
                    train_val_split: float = 0.9):
    """
    Load docid data and create PyTorch DataLoaders
    
    Args:
        docid_path: Path to the docid dataset
        dataset_name: Name of the corpus dataset
        subset: Subset of the corpus dataset
        split: Split of the corpus dataset
        max_samples: Maximum number of samples to load
        max_length: Maximum length for passage text
        batch_size: Batch size for DataLoader
        
    Returns:
        train_loader: PyTorch DataLoader for training
        test_loader: PyTorch DataLoader for testing
        train_size: Number of training examples
        test_size: Number of test examples
    """
    docids = load_from_disk(docid_path)
    dataset = download_corpus(dataset_name, subset, split)

    # Create full dataset
    full_dataset = PyTorchDocIDDataset(docids, dataset, max_samples=max_samples, max_length=max_length)
    
    # Split into train and val sets
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility with Keras
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Loaded {train_size} training docids")
    print(f"Loaded {val_size} validation docids")
    
    return train_loader, val_loader, train_size, val_size


def prepare_data(dataloader):
    """Prepare input and target texts from DataLoader for T5-Gemma"""
    input_texts = []
    target_texts = []
    
    for batch in dataloader:
        input_texts.extend(batch["input_text"])
        target_texts.extend(batch["target_text"])
        
    return input_texts, target_texts


def docid_accuracy(y_true, y_pred):
    """Custom metric for exact docid match"""
    return keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def evaluate_retrieval(model, test_loader, verbose=False, max_batches=5):
    """
    Evaluate exact match retrieval
    Will compute Hits@K and MRR metrics in the future when model supports beam search.
    
    Args:
        model: The trained model
        test_loader: PyTorch DataLoader for test data
        verbose: Whether to print detailed results
        max_batches: Maximum number of batches to evaluate (for speed)
    """
    exact_matches = 0
    total_examples = 0
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
            
        input_texts = batch["input_text"]
        expected_docids = batch["target_text"]
        
        for idx, (input_text, expected_docid) in enumerate(zip(input_texts, expected_docids)):
            # Generate predicted docid
            predicted_docid = model.generate(input_text, max_length=40)

            if verbose and total_examples < 5:
                print(f"Test Example {total_examples+1}:")
                print(f"Input Text: {input_text[:80]}...")
                print(f"Expected DocID: {expected_docid}")
                print(f"Predicted DocID: {predicted_docid}")
                print("-" * 40)
            
            if predicted_docid == expected_docid:
                exact_matches += 1
            
            total_examples += 1
    
    # Calculate final metrics
    accuracy = exact_matches / total_examples if total_examples > 0 else 0.0

    print("Retrieval Evaluation Results:")
    print(f"Total Examples: {total_examples}")
    print(f"Exact Match Accuracy: {accuracy:.4f}")

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
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    logger.info("T5-Gemma Sentiment Analysis Fine-tuning Script")
    logger.info("Backend: %s", keras.backend.backend())

    # Configure mixed precision
    logger.info(f"Configuring mixed precision: {args.mixed_precision}")
    keras.mixed_precision.set_global_policy(args.mixed_precision)
    logger.info(f"Mixed precision policy set: {keras.mixed_precision.global_policy()}")
    
    logger.info("Loading sentiment data...")
    train_loader, test_loader, _, _ = load_docid_data(
        docid_path=args.docid_path,
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_val_split=args.train_val_split
    )
    
    logger.info("Preparing training data...")
    input_texts, target_texts = prepare_data(train_loader)

    logger.info("Loading Gemma model...")        
    try:
        logger.info(f"T5GemmaSeq2SeqLM presets: {list(T5GemmaSeq2SeqLM.presets.keys())[:5]}")
        available_presets = list(T5GemmaSeq2SeqLM.presets.keys())
        if available_presets:
            preset = args.model_preset
            logger.info(f"Using preset: {preset}")
            model = T5GemmaSeq2SeqLM.from_preset(preset)
        else:
            raise Exception("No T5GemmaSeq2SeqLM presets available.")
    
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Exiting...")
        exit(1)
    

    logger.info("Testing model inference before training...")
    response = model.generate({
        "encoder_text": "I want to say",
        "decoder_text": ""
        }
        , max_length=30)
    logger.info(f"Generated text: {response}")
    
    
    
    # print("\nEvaluating before fine-tuning:")
    # evaluate_retrieval(model, test_loader, verbose=True)

    logger.info("Compiling model for fine-tuning...")
    optimizer = keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        metrics=[docid_accuracy],
        jit_compile=True
    )
    
    logger.info("Starting fine-tuning...")
    try:
        # Create proper dictionary format for T5GemmaSeq2SeqLM
        logger.info("Formatting data for seq2seq training...")
        
        # Manual train/validation split (80/20)
        split_idx = int(0.8 * len(input_texts))
        training_inputs = input_texts[:split_idx]
        training_targets = target_texts[:split_idx]
        val_inputs = input_texts[split_idx:]
        val_targets = target_texts[split_idx:]
        
        # Format data as dictionary with encoder_text and decoder_text keys
        training_data = {
            "encoder_text": training_inputs,
            "decoder_text": training_targets
        }
        
        validation_data = {
            "encoder_text": val_inputs,
            "decoder_text": val_targets
        } if val_inputs else None
        
        logger.info(f"Training on {len(training_inputs)} examples...")
        if validation_data:
            logger.info(f"Validation on {len(val_inputs)} examples...")
        logger.info("Sample encoder text: " + training_inputs[0][:50] + "...")
        logger.info("Sample decoder text: " + training_targets[0])
        
        # Train the model with properly formatted data
        history = model.fit(
            x=training_data,
            validation_data=validation_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1
        )
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training with dictionary format: {e}")
    
    logger.info("Evaluating fine-tuned model on test data...")
    evaluate_retrieval(model, test_loader, verbose=True)
    
    logger.info("Script completed!")