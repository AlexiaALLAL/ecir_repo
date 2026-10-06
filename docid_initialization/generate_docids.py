# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "pandas",
#     "sentence-transformers",
# ]
# ///

"""Generate unique document IDs from a corpus."""


from datasets import load_dataset
import uuid
from typing import Callable, Dict, Any
import os
import argparse
import logging
from collections import Counter

from docid_generator import (
    DocIDGenerator, UUID4DocIDGenerator, FirstWordsDocIDGenerator,
    EmbeddingDocIDGenerator, ResidualQuantizationDocIDGenerator,
)

from utils import download_corpus


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DOCID_GENERATORS: Dict[str, DocIDGenerator] = {
    'uuid': UUID4DocIDGenerator,
    'first_words': FirstWordsDocIDGenerator,
    'embedding': EmbeddingDocIDGenerator,
    'residual_quantization': ResidualQuantizationDocIDGenerator,
    # Add more generators here as needed
}


def generate_all_docids(
        dataset_name:str, subset:str=None, split:str="train", max_samples:int=None,
        docid_method:str='uuid', generator_params:Dict[str,Any]=None, batch_size:int=None,
        ensure_unique:bool=True):
    """Generate document IDs for all documents in the dataset.
    Returns a dataset with columns 'passage_id' and 'docid', 
    and saves it to disk in HuggingFace Arrow dataset format.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        subset: Subset of the dataset
        split: Split to use
        max_samples: Maximum number of samples to process
        docid_method: Method for generating docids ('uuid', 'first_words', 'embedding', 'residual_quantization')
        generator_params: Parameters to pass to the generator
        batch_size: Batch size for processing (default: None for single doc processing, 
                   recommended: 32+ for residual_quantization)
        ensure_unique: Whether to add duplicate counters to ensure uniqueness (default: True)
    """
    dataset = download_corpus(dataset_name, subset=subset, split=split)
    if max_samples:
        dataset = dataset.select(range(max_samples))
    
    logger.info(f"Generating DocIDs using method: {docid_method}")
    generator = DOCID_GENERATORS.get(docid_method)
    if generator is None:
        raise ValueError(f"Unknown docid_method: {docid_method}")   
    docid_generator = generator(**(generator_params or {}))  # Pass codebook or other params if needed
    
    # Use batched processing if batch_size is specified
    if batch_size and batch_size > 1:
        def generate_docids_batch(batch):
            return {'docid': docid_generator(batch['passage'])}
        
        docid_dataset = dataset.map(generate_docids_batch, batched=True, batch_size=batch_size)
    else:
        # Single document processing (backward compatible)
        docid_dataset = dataset.map(lambda x: {
            'docid': docid_generator(x['passage'])
        })
    
    if ensure_unique:
        logger.info("Adding duplicate counters to ensure uniqueness...")
        
        docid_counts = Counter(docid_dataset['docid']) # This loads all docids into memory, we can't avoid it to ensure uniqueness... :(
        duplicate_docids = {docid for docid, count in docid_counts.items() if count > 1}

        if len(duplicate_docids) > 0:
            logger.info(f"  Found {len(duplicate_docids)} duplicate DocIDs.")

            if docid_method == 'residual_quantization':
                # docid_generator is a ResidualQuantizationDocIDGenerator
                # docid_generator.codebook is a ResidualQuantizerKmeans
                codebook_size = docid_generator.codebook.codebook_size
            else:
                codebook_size = None
            
            # Build new docids with suffixes
            seen_counts = {docid: 0 for docid in duplicate_docids}
            new_docids = []
            codebook_size_warning_shown = False

            for docid in docid_dataset['docid']:
                if docid in duplicate_docids:
                    seen_counts[docid] += 1
                    if codebook_size:
                        suffix = seen_counts[docid] % codebook_size
                        # Warn the first time we're wrapping around (suffix resets to 0)
                        if not codebook_size_warning_shown and seen_counts[docid] >= codebook_size:
                            logger.warning(f"  ⚠️  DocID '{docid}' has at least {seen_counts[docid]} duplicates, exceeding codebook_size={codebook_size}.")
                            logger.warning(f"Docids won't be uniques... Consider increasing codebook size or number of codebooks.")
                            codebook_size_warning_shown = True
                    else:
                        suffix = seen_counts[docid]
                    
                    new_docid = f"{docid}-{suffix}"
                    new_docids.append(new_docid)
                else:
                    new_docids.append(docid)
            
            docid_dataset = docid_dataset.remove_columns(['docid'])
            docid_dataset = docid_dataset.add_column('docid', new_docids)
            logger.info("  Duplicate DocIDs have been modified to ensure uniqueness.")            

    docid_dataset = docid_dataset.select_columns(['passage_id', 'docid'])
    logger.info(f"Dataset size: {len(docid_dataset)}")

    # Save as a HuggingFace dataset (memory-mapped Arrow format)
    os.makedirs("docids", exist_ok=True)
    
    # Extract name from codebook path if provided, otherwise use method name
    if generator_params and 'codebook_path' in generator_params:
        codebook_path = generator_params['codebook_path']
        # Extract filename without extension from codebook path
        codebook_name = os.path.splitext(os.path.basename(codebook_path))[0]
        samples_str = f"_n{max_samples//1000}k" if max_samples else "all"
        dataset_path = f"docids/{codebook_name}{samples_str}_dataset"
    elif generator_params and 'embedding_model' in generator_params:
        # For embedding method, include model name in path
        model_slug = generator_params['embedding_model'].replace('/', '_').replace('-', '_')
        dataset_slug = dataset_name.replace('/', '_')
        samples_str = f"_n{max_samples//1000}k" if max_samples else ""
        dataset_path = f"docids/{docid_method}_{dataset_slug}_{model_slug}{samples_str}_dataset"
    else:
        dataset_path = f"docids/{dataset_name.replace('/', '_')}_{docid_method}_dataset"
    
    docid_dataset.save_to_disk(dataset_path)
    logger.info(f"Document IDs saved to HuggingFace dataset at {dataset_path}")

    # if ensure_unique:
    #     logger.info("First non unique DocID that was modified: ")
    #     nb_prints = 0
    #     for docid, count in docid_counts.items():
    #         if count > 1:
    #             logger.info(f"  Original DocID: {docid} | Occurrences: {count}")
    #             logger.info(f"New lines with modified DocIDs:")
    #             logger.info(
    #                 docid_dataset.filter(
    #                         lambda x: x['docid'].startswith(docid + '_')
    #                 ).select_columns(['passage_id', 'docid']).to_pandas().to_string(index=False)
    #             )
    #             nb_prints += 1
    #         if nb_prints >= 5:
    #             break

    return docid_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Document IDs for a given corpus.")
    parser.add_argument("--dataset_name", type=str, default="sentence-transformers/msmarco", help="Name of the dataset to download from HuggingFace (default: sentence-transformers/msmarco).")
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset, if applicable.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train).")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (default: all).")
    parser.add_argument("--docid_method", type=str, default="uuid", help="Method to generate DocIDs (default: uuid).")
    parser.add_argument("--generator_params", type=str, default=None, help="Additional parameters for the DocID generator in key=value format, separated by commas.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for processing documents (default: None for single doc, auto-set to 32 for residual_quantization).")
    parser.add_argument("--ensure_unique", action="store_true", default=True, 
                        help="Add duplicate counters to ensure all DocIDs are unique (default: True)." \
                        "To set to False, use --no-ensure_unique.")
    parser.add_argument("--no-ensure_unique", action="store_false", dest="ensure_unique", 
                        help="Do not add duplicate counters; DocIDs may not be unique.")
    args = parser.parse_args()


    logger.info("Starting DocID generation script with arguments: %s", args)
    generator_params = {}
    if args.generator_params:
        for param in args.generator_params.split(','):
            key, value = param.split('=')
            generator_params[key] = value

    generate_all_docids(
        dataset_name=args.dataset_name,
        subset=args.subset, 
        split=args.split, 
        max_samples=args.max_samples, 
        docid_method=args.docid_method, 
        generator_params=generator_params, 
        batch_size=args.batch_size,
        ensure_unique=args.ensure_unique
    )