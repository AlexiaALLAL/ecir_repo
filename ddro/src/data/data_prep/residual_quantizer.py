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
                init="k-means++",
                n_init="auto",
                # n_init=3,
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
    
    def save(self, filepath: str, vocab_size: int = None, cluster_num: int = None):
        """
        Save the trained codebooks to disk.
        
        Args:
            filepath: Path to save the model
            vocab_size: Optional vocabulary size (for docid generation)
            cluster_num: Optional cluster number (for docid generation)
        """
        state = {
            'embedding_dim': self.embedding_dim,
            'nb_codebooks': self.nb_codebooks,
            'codebook_size': self.codebook_size,
            'codebooks': self.codebooks,
            'is_trained': self.is_trained
        }
        
        # Add optional parameters if provided
        if vocab_size is not None:
            state['vocab_size'] = vocab_size
        if cluster_num is not None:
            state['cluster_num'] = cluster_num
            
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Codebooks saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ResidualQuantizerKmeans':
        """
        Load trained codebooks from disk.
        
        Returns the loaded ResidualQuantizerKmeans object.
        The saved state may also contain 'vocab_size' and 'cluster_num' which are accessible via the pickle file.
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
    



class PQRQ:
    """
    Implements Residual Quantization for encoding vectors.
    """
    def __init__(self, embedding_dim: int, nb_subspaces: int, codebook_size: int, rq_steps: int):
        self.embedding_dim = embedding_dim
        self.nb_subspaces = nb_subspaces
        self.codebook_size = codebook_size
        self.rq_steps = rq_steps
        self.codebooks = [] # List of ResidualQuantizerKmeans objects, one per subspace
        self.is_trained = False
    
    def train(self, embeddings: np.ndarray) -> list[int]:
        """
        Encode the input vector using residual quantization.
        Args:
            embeddings: np.ndarray of shape (nb_vectors, embedding_dim)
        """
        self.codebooks = []
        subspace_dim = self.embedding_dim // self.nb_subspaces

        for i in range(self.nb_subspaces):
            logger.info(f"Training pq subspace {i+1}/{self.nb_subspaces}")

            # split the residual into subspaces
            subspace_residual = embeddings[:, i*subspace_dim:(i+1)*subspace_dim]

            # train RQ on the subspace residual
            rq = ResidualQuantizerKmeans(
                embedding_dim=subspace_dim, 
                nb_codebooks=self.rq_steps, 
                codebook_size=self.codebook_size
            )
            rq.train(subspace_residual)
            self.codebooks.append(rq)

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

        codes = np.zeros((embeddings.shape[0], self.rq_steps * self.nb_subspaces), dtype=int)
        subspace_dim = self.embedding_dim // self.nb_subspaces

        for i, rq in enumerate(self.codebooks): # range self.nb_subspaces
            subspace_residual = embeddings[:, i*subspace_dim:(i+1)*subspace_dim]
            subspace_codes = rq.encode(subspace_residual)
            # Place the subspace codes in the correct position in the final code array
            # start with all rq step = 0, then all rq step = 1, ... to keep hierarchical structure
            for j in range(self.rq_steps):
                codes[:, j*self.nb_subspaces + i] = subspace_codes[:, j]

        
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

        subspace_dim = self.embedding_dim // self.nb_subspaces
        for i, rq in enumerate(self.codebooks): # range self.nb_subspaces
            # Extract the codes for this subspace and reshape to (batch_size, rq_steps)
            subspace_codes = codes[:, i::self.nb_subspaces] # take every nb_subspaces-th code starting from i
            subspace_embeddings = rq.decode(subspace_codes)
            embeddings[:, i*subspace_dim:(i+1)*subspace_dim] = subspace_embeddings

        return embeddings
    
    def save(self, filepath: str, vocab_size: int = None):
        """
        Save the trained codebooks to disk.
        
        Args:
            filepath: Path to save the model
            vocab_size: Optional vocabulary size of t5 (for docid generation)
        """
        state = {
            'embedding_dim': self.embedding_dim,
            'nb_subspaces': self.nb_subspaces,
            'codebook_size': self.codebook_size,
            'codebooks': self.codebooks,
            'rq_steps': self.rq_steps,
            'nb_subspaces': self.nb_subspaces,
            'is_trained': self.is_trained
        }
        
        # Add optional parameters if provided
        if vocab_size is not None:
            state['vocab_size'] = vocab_size
            
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Codebooks saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PQRQ':
        """
        Load trained codebooks from disk.
        
        Returns the loaded PQRQ object.
        The saved state may also contain 'vocab_size' which are accessible via the pickle file.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        pq_rq = cls(
            embedding_dim=state['embedding_dim'],
            nb_subspaces=state['nb_subspaces'],
            codebook_size=state['codebook_size'],
            rq_steps=state['rq_steps']
        )
        pq_rq.codebooks = state['codebooks']
        pq_rq.is_trained = state['is_trained']
        
        logger.info(f"Codebooks loaded from {filepath}")
        return pq_rq
