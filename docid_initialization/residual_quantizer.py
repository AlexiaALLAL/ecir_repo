# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "torch",
#     "sentence-transformers",
#     "datasets",
# ]
# ///

"""
Utilities for training and using Residual Quantization codebooks for DocID generation.
Implements each codebook as a KMeans clustering model.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import logging
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
from sentence_transformers import SentenceTransformer
try:
    from .utils import download_corpus
except ImportError:
    from utils import download_corpus
import torch
import argparse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ResidualQuantizerKmeans:
    """
    Implements Residual Quantization for encoding vectors.
    """
    def __init__(self, embedding_dim: int, nb_codebooks: int, codebook_size: int):
        self.embedding_dim = embedding_dim
        self.nb_codebooks = nb_codebooks
        self.codebook_size = codebook_size
        self.codebooks = []
        self.is_trained = False
    
    def train(self, embeddings: np.ndarray) -> list[int]:
        """
        Encode the input vector using residual quantization.
        Args:
            embeddings: np.ndarray of shape (nb_vectors, embedding_dim)
        """
        residual = embeddings.copy()
        self.codebooks = []

        for i in range(self.nb_codebooks):
            logger.info(f"Training codebook {i+1}/{self.nb_codebooks}")

            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size, 
                random_state=42,
                batch_size=1024,
                n_init=3,
                max_iter=100,
                verbose=0
                )
            kmeans.fit(residual)
            self.codebooks.append(kmeans)

            codes = kmeans.predict(residual)

            quantized = kmeans.cluster_centers_[codes]
            residual -= quantized

        self.is_trained = True
        logger.info("Training completed.")

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode the input vector using the trained codebooks.
        Args:
            embeddings: np.ndarray of shape (batch_size, embedding_dim)
        Returns:
            codes: Array of shape (batch_size, nb_codebooks) with the code indices
        """
        if not self.codebooks or not self.is_trained:
            raise ValueError("Codebooks are not trained. Call train() before encode().")

        codes = np.zeros((embeddings.shape[0], self.nb_codebooks), dtype=int)
        residual = embeddings.copy()

        for i, codebook in enumerate(self.codebooks):
            codes[:, i] = codebook.predict(residual)
            quantized = codebook.cluster_centers_[codes[:, i]]
            residual -= quantized

        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode the codes back to approximate embeddings.
        Args:
            codes: np.ndarray of shape (batch_size, nb_codebooks)
        Returns:
            embeddings: np.ndarray of shape (batch_size, embedding_dim)
        """
        if not self.codebooks or not self.is_trained:
            raise ValueError("Codebooks are not trained. Call train() before decode().")

        embeddings = np.zeros((codes.shape[0], self.embedding_dim), dtype=float)

        for i, codebook in enumerate(self.codebooks):
            quantized = codebook.cluster_centers_[codes[:, i]]
            embeddings += quantized

        return embeddings
    
    def codes_to_docid(self, codes: np.ndarray) -> str:
        """
        Convert code indices to a string DocID.
        Args:
            codes: np.ndarray of shape (batch_size, nb_codebooks)
        Returns:
            List[str]: List of DocID strings
        """
        if codes.ndim == 1:
            return ' - '.join(map(str, codes))
        else:
            return [' - '.join(map(str, code)) for code in codes]
    
    def docid_to_codes(self, docid: str) -> np.ndarray:
        """
        Convert a DocID string back to code indices.
        Args:
            List[str]: List of DocID strings
        Returns:
            np.ndarray of shape (nb_codebooks,)
        """
        if isinstance(docid, str):
            return np.array([int(code) for code in docid.split(' - ')], dtype=int)
        else:
            return np.array([
                [int(code) for code in d.split(' - ')] 
                for d in docid
                ], dtype=int)
    
    def save(self, filepath: str):
        """
        Save the trained codebooks to disk.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embedding_dim': self.embedding_dim,
                'nb_codebooks': self.nb_codebooks,
                'codebook_size': self.codebook_size,
                'codebooks': self.codebooks,
                'is_trained': self.is_trained
            }, f)
        logger.info(f"Codebooks saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ResidualQuantizerKmeans':
        """
        Load trained codebooks from disk.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        rq = cls(
            embedding_dim=state['embedding_dim'],
            nb_codebooks=state['nb_codebooks'],
            codebook_size=state['codebook_size']
        )
        rq.codebooks = state['codebooks']
        rq.is_trained = state['is_trained']
        
        logger.info(f"Codebooks loaded from {filepath}")
        return rq
    

def train_residual_quantizer(
        dataset_name: str,
        subset: str,
        split: str,
        max_samples: int=None,
        embedding_model: str="all-MiniLM-L6-v2",
        num_codebooks: int=4,
        codebook_size: int=256,
        batch_size: int=32,
        save_path: str=None,
    ) -> ResidualQuantizerKmeans:
    """
    Train a Residual Quantizer using embeddings from the specified dataset.
    Args:
        dataset_name: Name of the dataset to use for training
        subset: Subset of the dataset (if applicable)
        split: Split of the dataset to use ('train', 'validation', etc.)
        max_samples: Maximum number of samples to use for training
        embedding_model: SentenceTransformer model name for embeddings
        num_codebooks: Number of codebooks to use in residual quantization
        codebook_size: Number of centroids per codebook
        batch_size: Batch size for encoding passages
        save_path: Path to save the trained codebook (auto-generated if None)
    Returns:
        Trained ResidualQuantizerKmeans instance
    """
    
    # Auto-generate save_path if not provided
    if save_path is None:
        dataset_slug = dataset_name.replace('/', '_')
        model_slug = embedding_model.replace('/', '_').replace('-', '_')
        samples_str = f"{max_samples//1000}k" if max_samples else "all"
        save_path = f"codebooks/rq_{dataset_slug}_{model_slug}_nc{num_codebooks}_cs{codebook_size}_n{samples_str}.pkl"
        logger.info(f"Auto-generated save path: {save_path}")

    dataset = download_corpus(dataset_name, subset=subset, split=split)
    if max_samples:
        dataset = dataset.select(range(max_samples))
    
    encoder = SentenceTransformer(
        embedding_model, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    logger.info(f"Encoding {len(dataset)} passages using model {embedding_model}...")
    embeddings = encoder.encode(
        dataset['passage'],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embedding_dim = embeddings.shape[1]
    rq = ResidualQuantizerKmeans(
        embedding_dim=embedding_dim,
        nb_codebooks=num_codebooks,
        codebook_size=codebook_size
    )
    rq.train(embeddings)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rq.save(save_path)


    # Test reconstruction quality
    test_codes = rq.encode(embeddings[:1000])
    test_decoded = rq.decode(test_codes)
    reconstruction_error = np.mean((embeddings[:1000] - test_decoded) ** 2)
    logger.info(f"Reconstruction MSE on 1000 samples: {reconstruction_error:.6f}")
    
    return rq



def test_residual_quantizer():
    """
    Simple test of the ResidualQuantizerKmeans class.
    """
    batch_size = 512
    embedding_dim = 8
    nb_codebooks = 3
    codebook_size = 10
    embeddings = np.random.rand(batch_size, embedding_dim).astype(np.float32)
    rq = ResidualQuantizerKmeans(embedding_dim, nb_codebooks, codebook_size)
    rq.train(embeddings)
    codes = rq.encode(embeddings)
    # logging.info(f"Encoded codes:\n{codes}")
    decoded_embeddings = rq.decode(codes)
    # logging.info(f"Decoded embeddings:\n{decoded_embeddings}")
    reconstruction_error = np.mean((embeddings - decoded_embeddings) ** 2)
    logging.info(f"Reconstruction MSE: {reconstruction_error}")
    docids = rq.codes_to_docid(codes)
    # logging.info(f"Generated DocIDs:\n{docids}")
    recovered_codes = np.array([rq.docid_to_codes(docid) for docid in docids])
    # logging.info(f"Recovered codes:\n{recovered_codes}")
    assert np.array_equal(codes, recovered_codes), "DocID to codes conversion failed"
    logging.info("ResidualQuantizerKmeans test completed successfully.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Residual Quantization codebook for DocID generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="mteb/msmarco",
                       help="Name of the dataset to use for training")
    parser.add_argument("--subset", type=str, default=None,
                       help="Subset of the dataset")
    parser.add_argument("--split", type=str, default="corpus",
                       help="Split of the dataset to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use for training (None = all)")
    
    # Model arguments
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                       help="SentenceTransformer model name for embeddings")
    
    # RQ arguments
    parser.add_argument("--num_codebooks", type=int, default=5,
                       help="Number of codebooks to use in residual quantization")
    parser.add_argument("--codebook_size", type=int, default=256,
                       help="Number of centroids per codebook")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="Batch size for encoding passages")
    parser.add_argument("--save_path", type=str, default=None,
                       help="Path to save the trained codebook (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    logger.info("Starting Residual Quantizer training with arguments:")
    logger.info(f"  Dataset: {args.dataset_name}/{args.subset}/{args.split}")
    logger.info(f"  Max samples: {args.max_samples}")
    logger.info(f"  Embedding model: {args.embedding_model}")
    logger.info(f"  Num codebooks: {args.num_codebooks}")
    logger.info(f"  Codebook size: {args.codebook_size}")
    logger.info(f"  Batch size: {args.batch_size}")
    
    rq = train_residual_quantizer(
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
        embedding_model=args.embedding_model,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
    
    logger.info("Training completed successfully!")
