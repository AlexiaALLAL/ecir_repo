# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "pandas",
#     "sentence-transformers",
#     "torch",
# ]
# ///

"""This implements the functions to generate DocIDs."""


from typing import Any, Union, List
import uuid
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from .residual_quantizer import ResidualQuantizerKmeans
import torch


class DocIDGenerator(ABC):
    """
    Base interface for all DocID generators. 
    New methods should inherit from this class.
    """
    @abstractmethod
    def __call__(self, passage_text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generates unique DocID(s) from the passage information.
        
        Args:
            passage_text: Single passage (str) or batch of passages (List[str])
            
        Returns:
            Single docid (str) if input is str, or list of docids (List[str]) if input is List[str]
            
        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method to generate a DocID.")



class UUID4DocIDGenerator(DocIDGenerator):
    """
    Use a simple uuid4 for quick and unique DocID generation.
    """
    def __call__(self, passage_text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(passage_text, list):
            return [str(uuid.uuid4()) for _ in passage_text]
        else:
            return str(uuid.uuid4())
    

class FirstWordsDocIDGenerator(DocIDGenerator):
    """
    Generate DocID based on the first N words of the passage.
    """
    def __init__(self, n_words: int=5):
        self.n_words = int(n_words)

    def __call__(self, passage_text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(passage_text, list):
            return [
                ' '.join(text.split()[:self.n_words]).strip() 
                for text in passage_text
                ]
        else:
            first_words = ' '.join(passage_text.split()[:self.n_words])
            return f"{first_words}".strip()
    

class EmbeddingDocIDGenerator(DocIDGenerator):
    """
    Generate DocID using a simple embedding model.
    Each embedding dimension is quantized to an integer for compact representation.
    
    Requires:
    - A SentenceTransformer model for encoding passages
    
    Args:
        embedding_model: SentenceTransformer model name or instance (default: all-MiniLM-L6-v2)
        quantization_bits: Number of bits for quantization (default: 8, range 0-255)
    """
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", quantization_bits: int = 8):
        self.embedding_model_name = embedding_model
        self.quantization_bits = int(quantization_bits)
        self.max_value = (2 ** self.quantization_bits) - 1
        
        self.encoder = SentenceTransformer(
            embedding_model, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
    def __call__(self, passage_text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generate docid(s) from passage text using the embedding model.
        Each embedding dimension is quantized to an integer in range [0, 2^quantization_bits - 1].
        """
        # Normalize input to list for processing
        is_single = isinstance(passage_text, str)
        passages = [passage_text] if is_single else passage_text
        
        # Encode all passages at once (batch encoding)
        embeddings = self.encoder.encode(
            passages,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Quantize embeddings: normalized embeddings are in [-1, 1]
        # Map [-1, 1] to [0, max_value]
        quantized_embeddings = ((embeddings + 1) / 2 * self.max_value).astype(int)
        
        # Convert to string docids with underscore separator
        docids = [' - '.join(map(str, emb)) for emb in quantized_embeddings]
        
        # Return single docid if input was single, otherwise return list
        return docids[0] if is_single else docids
    

class ResidualQuantizationDocIDGenerator(DocIDGenerator):
    """
    Generate DocID using residual quantization with a trained codebook.
    
    Requires:
    - A trained ResidualQuantizationCodebook (from train_rq_docids.py)
    - A SentenceTransformer model for encoding passages
    
    Args:
        codebook: Trained ResidualQuantizationCodebook instance
        embedding_model: SentenceTransformer model name or instance
    """
    def __init__(self, codebook_path: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        if codebook_path is None:
            raise ValueError("Codebook path is required for ResidualQuantizationDocIDGenerator")
        self.codebook = ResidualQuantizerKmeans.load(codebook_path)
        self.embedding_model_name = embedding_model
        
        self.encoder = SentenceTransformer(
            embedding_model, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
    def __call__(self, passage_text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generate docid(s) from passage text using residual quantization.
        
        Args:
            passage_text: Single passage (str) or batch of passages (List[str])
            
        Returns:
            Single docid (str) or list of docids (List[str]) matching input type
        """
        # Normalize input to list for processing
        is_single = isinstance(passage_text, str)
        passages = [passage_text] if is_single else passage_text
        
        # Encode all passages at once (batch encoding)
        embeddings = self.encoder.encode(
            passages,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        codes = self.codebook.encode(embeddings)
        docids = [self.codebook.codes_to_docid(code) for code in codes]
        
        # Return single docid if input was single, otherwise return list
        return docids[0] if is_single else docids