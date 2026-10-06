# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "torch",
#     "wandb",
#     "tqdm",
# ]
# ///

"""
Train RQ-VAE (Residual Quantized VAE) for hierarchical semantic ID generation.
This model learns to encode item embeddings into discrete hierarchical codes.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np

import wandb
import logging
import tqdm

# Enable TensorFloat32 for faster training on CUDA
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    print("TensorFloat32 enabled for faster float32 matrix multiplications")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class RQVAEConfig:
    """Configuration for RQ-VAE training."""

    # Data settings
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "rqvae")

    # Model parameters
    embedding_dim: int = 3  # Embeddings dimension (before encoder)
    codebook_quantization_levels: int = 3  # Number of hierarchical levels
    codebook_size: int = 16  # Number of codes per codebook
    use_rotation_trick: bool = True  # Use rotation trick for better gradient flow
    recon_weight: float = 1.0  # Weight for reconstruction loss
    vq_weight: float = 1.0  # Weight for VQ loss component
    ir_weight: float = 3e-4  # Weight for IR loss component (if using query/negative samples)
    ir_query_processing: bool = True # Whether to apply RQ on the query (and add it in the loss calculation) or just use the original query embedding for the IR loss
    ir_loss_on_latent: bool = False  # Whether to apply IR loss on the latent embeddings (after quantization) instead of the final reconstruction output
    ir_loss_type: str = "infonce"  # Type of IR loss: "infonce" or "triplet"
    ir_loss_margin: float = 1.0  # Margin (epsilon) for triplet loss
    
    # Encoder/Decoder parameters
    use_encoder_decoder: bool = False  # Whether to use trainable encoder/decoder
    encoder_hidden_dims: List[int] = field(default_factory=list)  # Hidden dimensions for encoder (e.g., [512, 256, 128])
    codebook_embedding_dim: int = 3  # Dimension of codebook vectors (after encoder, if used, otherwise same as embedding_dim)
    commitment_weight: float = 0.25  # Weight for commitment loss in VQ

    # Training parameters
    batch_size: int = 32768  # Batch size for training
    gradient_accumulation_steps: int = 1  # Number of gradient accumulation steps
    num_epochs: int = 200  # Number of training epochs
    scheduler_type: str = "cosine_with_warmup"  # Learning rate scheduler type ("cosine", "cosine_with_warmup")
    warmup_start_lr: float = 1e-8  # Starting learning rate for warmup (only for cosine_with_warmup)
    warmup_steps: int = 10  # Number of warmup steps (only for cosine_with_warmup)
    max_lr: float = 3e-4  # Maximum learning rate (start of cosine)
    min_lr: float = 1e-6  # Minimum learning rate (end of cosine)
    use_gradient_clipping: bool = True  # Enable gradient clipping
    gradient_clip_norm: float = 1.0  # Maximum gradient norm for clipping
    use_kmeans_init: bool = True  # Use k-means initialization for codebooks
    reset_unused_codes: bool = True  # Reset unused codebook codes during training
    steps_per_codebook_reset: int = 2  # Reset unused codebook codes every N steps (breaks if set to 1)
    codebook_usage_threshold: float = 1.0  # Only reset if usage falls below this proportion (0-1)
    val_split: float = 0.05  # Validation set split ratio

    # Logging and checkpointing
    steps_per_train_log: int = 5  # Log training progress every N steps
    steps_per_val_log: int = 20  # Validate and checkpoint every N steps

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== RQ-VAE Configuration ===")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        # Model parameters
        logger.info("Model Parameters:")
        logger.info(f"  embedding_dim: {self.embedding_dim}")
        logger.info(f"  codebook_quantization_levels: {self.codebook_quantization_levels}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  use_rotation_trick: {self.use_rotation_trick}")
        logger.info(f"  recon_weight: {self.recon_weight}")
        logger.info(f"  vq_weight: {self.vq_weight}")
        logger.info(f"  ir_weight: {self.ir_weight}")
        logger.info(f"  ir_query_processing: {self.ir_query_processing}")
        logger.info(f"  ir_loss_on_latent: {self.ir_loss_on_latent}")
        logger.info(f"  ir_loss_type: {self.ir_loss_type}")
        logger.info(f"  ir_loss_margin: {self.ir_loss_margin}")
        logger.info(f"  use_encoder_decoder: {self.use_encoder_decoder}")
        if self.use_encoder_decoder:
            logger.info(f"  encoder_hidden_dims: {self.encoder_hidden_dims}")
            logger.info(f"  codebook_embedding_dim: {self.codebook_embedding_dim}")
        logger.info(f"  commitment_weight: {self.commitment_weight}")
        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  warmup_steps: {self.warmup_steps}")
        logger.info(f"  max_lr: {self.max_lr}")
        logger.info(f"  min_lr: {self.min_lr}")
        logger.info(f"  use_gradient_clipping: {self.use_gradient_clipping}")
        logger.info(f"  gradient_clip_norm: {self.gradient_clip_norm}")
        logger.info(f"  use_kmeans_init: {self.use_kmeans_init}")
        logger.info(f"  reset_unused_codes: {self.reset_unused_codes}")
        logger.info(f"  steps_per_codebook_reset: {self.steps_per_codebook_reset}")
        logger.info(f"  codebook_usage_threshold: {self.codebook_usage_threshold}")
        logger.info(f"  val_split: {self.val_split}")

        # Logging and checkpointing
        logger.info("Logging and Checkpointing:")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info("===========================")


class EmbeddingDataset(Dataset):
    """Dataset for loading embeddings from numpy array."""

    def __init__(self, embeddings: np.ndarray, queries: Optional[np.ndarray] = None, negatives: Optional[np.ndarray] = None, limit: Optional[int] = None):
        """Load embeddings from numpy array.

        Args:
            embeddings: Numpy array of embeddings (size [num_items, embedding_dim])
            queries: Optional numpy array of query embeddings (size [num_items, embedding_dim])
            negatives: Optional numpy array of negative embeddings (size [num_items, embedding_dim])
            limit: Optional limit on number of items to load
        """

        if limit is not None:
            logger.info(f"Limiting to {limit} items")
            embeddings = embeddings[:limit]
            if queries is not None:
                queries = queries[:limit]
            if negatives is not None:
                negatives = negatives[:limit]
        
        if queries is not None and (embeddings.shape[1] != queries.shape[1] or queries.shape[1] != negatives.shape[1]):
            print(f"Resizing queries and docs embeddings to match dimensions: {embeddings.shape[1]} vs {queries.shape[1]} vs {negatives.shape[1]}")
            min_dim = min(embeddings.shape[1], queries.shape[1], negatives.shape[1])
            embeddings = embeddings[:, :min_dim]
            queries = queries[:, :min_dim]
            negatives = negatives[:, :min_dim]

        # Extract embeddings and convert to tensor
        embeddings_list = embeddings.tolist()
        self.embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
        logger.info(f"Loaded {len(self.embeddings):,} embeddings of dimension {self.embeddings.shape[1]}")

        if queries is not None:
            queries_list = queries.tolist()
            self.queries = torch.tensor(queries_list, dtype=torch.float32)
            logger.info(f"Loaded {len(self.queries):,} query embeddings of dimension {self.queries.shape[1]}")
        else:
            self.queries = None

        if negatives is not None:
            negatives_list = negatives.tolist()
            self.negatives = torch.tensor(negatives_list, dtype=torch.float32)
            logger.info(f"Loaded {len(self.negatives):,} negative embeddings of dimension {self.negatives.shape[1]}")
        else:
            self.negatives = None

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.queries is not None:
            return self.embeddings[idx], self.queries[idx], self.negatives[idx]
        else:
            return self.embeddings[idx]


class QuantizationOutput(NamedTuple):
    quantized_st: Tensor
    quantized: Tensor
    indices: Tensor
    loss: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor


class VectorQuantizer(nn.Module):
    """Base class for vector quantization with shared functionality."""

    def __init__(self, config: RQVAEConfig):
        super().__init__()
        # Use codebook_embedding_dim if encoder is used, otherwise use embedding_dim
        self.codebook_embedding_dim = config.codebook_embedding_dim if config.use_encoder_decoder else config.embedding_dim
        self.codebook_size = config.codebook_size
        self.use_rotation_trick = config.use_rotation_trick
        self.use_encoder_decoder = config.use_encoder_decoder
        self.commitment_weight = config.commitment_weight

        # Learnable codebook
        self.embedding = nn.Embedding(self.codebook_size, self.codebook_embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.codebook_size, 1 / self.codebook_size)

        # Track codebook usage
        self.register_buffer("usage_count", torch.zeros(self.codebook_size))
        self.register_buffer("update_count", torch.tensor(0))

    @staticmethod
    def l2norm(t: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
        """L2 normalize tensor along specified dimension."""
        return F.normalize(t, p=2, dim=dim, eps=eps)

    @staticmethod
    def safe_div(num: Tensor, den: Tensor, eps: float = 1e-6) -> Tensor:
        """Safe division to avoid numerical issues."""
        return num / den.clamp(min=eps)

    @staticmethod
    def rotation_trick(u: Tensor, q: Tensor, e: Tensor) -> Tensor:
        """
        Efficient rotation trick transform from Eq 4.2 in https://arxiv.org/abs/2410.06424

        Args:
            u: Unit vector from embeddings (normalized x)
            q: Unit vector from quantized output (normalized quantized)
            e: Original embeddings (x)

        Returns:
            Rotated embeddings
        """
        w = VectorQuantizer.l2norm(u + q, dim=-1).detach()

        # Reshape for batch matrix multiplication
        w_col = w.unsqueeze(-1)
        w_row = w.unsqueeze(-2)
        u_col = u.unsqueeze(-1).detach()
        q_row = q.unsqueeze(-2).detach()

        # For 2D input, add temporary batch dimension
        if e.ndim == 2:
            e_expanded = e.unsqueeze(1)  # [B, D] -> [B, 1, D]
            result = e_expanded - 2 * (e_expanded @ w_col @ w_row) + 2 * (e_expanded @ u_col @ q_row)
            return result.squeeze(1)  # [B, 1, D] -> [B, D]
        else:
            return e - 2 * (e @ w_col @ w_row).squeeze(-1) + 2 * (e @ u_col @ q_row).squeeze(-1)

    @staticmethod
    def rotate_to(src: Tensor, tgt: Tensor) -> Tensor:
        """
        Apply rotation trick STE from https://arxiv.org/abs/2410.06424
        to get gradients through VQ layer.

        Args:
            src: Source tensor (encoder output)
            tgt: Target tensor (quantized output)

        Returns:
            Rotated tensor that equals tgt in forward pass but has gradients
        """
        # Flatten to 2D for processing
        orig_shape = src.shape
        src_flat = src.reshape(-1, src.shape[-1])
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])

        # Get norms
        norm_src = src_flat.norm(dim=-1, keepdim=True)
        norm_tgt = tgt_flat.norm(dim=-1, keepdim=True)

        # Apply rotation in normalized space
        rotated_tgt = VectorQuantizer.rotation_trick(
            VectorQuantizer.safe_div(src_flat, norm_src), VectorQuantizer.safe_div(tgt_flat, norm_tgt), src_flat
        )

        # Scale to match target norm
        rotated = rotated_tgt * VectorQuantizer.safe_div(norm_tgt, norm_src).detach()
        # Reshape back
        return rotated.reshape(orig_shape)

    def find_nearest_codes(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Find nearest codebook entries for input tensor."""
        input_shape = x.shape
        flat_x = x.reshape(-1, self.codebook_embedding_dim)

        # Calculate distances to all codebook vectors
        distances = torch.cdist(flat_x, self.embedding.weight)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(input_shape)

        return indices.view(input_shape[:-1]), quantized

    def apply_gradient_estimator(self, x: Tensor, quantized: Tensor) -> Tensor:
        """Apply rotation trick or straight-through estimator."""
        if self.training and x.requires_grad:
            if self.use_rotation_trick:
                return self.rotate_to(x, quantized)
            else:
                return x + (quantized - x).detach()
        return quantized

    def update_usage(self, indices: Tensor):
        """Update codebook usage statistics."""
        indices_flat = indices.flatten()
        self.usage_count.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.float))
        self.update_count += 1

    def get_usage_rate(self) -> float:
        """Get proportion of codebook vectors that have been used."""
        if self.update_count == 0:
            return 0.0
        return (self.usage_count > 0).float().mean().item()

    def reset_usage_count(self):
        """Reset usage count (useful for periodic resets)."""
        self.usage_count.zero_()


    def forward(self, x: Tensor) -> QuantizationOutput:
        indices, quantized = self.find_nearest_codes(x)
        quantized_st = self.apply_gradient_estimator(x, quantized)

        # Compute VQ losses
        codebook_loss = F.mse_loss(x.detach(), quantized)
        
        # Only compute commitment loss when encoder is used
        if self.use_encoder_decoder:
            commitment_loss = F.mse_loss(x, quantized.detach())
            loss = codebook_loss + self.commitment_weight * commitment_loss
        else:
            commitment_loss = None
            loss = codebook_loss

        if self.training:
            self.update_usage(indices)

        return QuantizationOutput(quantized_st, quantized, indices, loss, codebook_loss, commitment_loss)

    def reset_unused_codes(self, batch_data: Tensor):
        """Reset unused codebook vectors to random samples from batch."""
        if self.update_count == 0:
            return

        # Find codes with zero usage
        unused_indices = (self.usage_count == 0).nonzero().squeeze(-1)

        if len(unused_indices) > 0 and batch_data.shape[0] >= len(unused_indices):
            # Sample random vectors from batch
            batch_flat = batch_data.reshape(-1, self.codebook_embedding_dim)
            random_indices = torch.randperm(batch_flat.shape[0], device=batch_flat.device)[: len(unused_indices)]
            self.embedding.weight.data[unused_indices] = batch_flat[random_indices].detach()

        # Reset usage count after replacement
        self.reset_usage_count()


class RQVAE(nn.Module):
    """Residual Quantized VAE for generating semantic IDs."""

    def __init__(self, config: RQVAEConfig):
        super().__init__()

        # Store the config
        self.config = config

        # Extract model parameters from config
        self.embedding_dim = config.embedding_dim
        self.codebook_quantization_levels = config.codebook_quantization_levels
        self.codebook_size = config.codebook_size
        self.use_encoder_decoder = config.use_encoder_decoder
        self.codebook_embedding_dim = config.codebook_embedding_dim if config.use_encoder_decoder else config.embedding_dim
        self.use_rotation_trick = config.use_rotation_trick

        # Build encoder and decoder if specified
        if self.use_encoder_decoder:
            self.encoder_hidden_dims = config.encoder_hidden_dims
            # Build encoder: item_embedding_dim -> encoder_hidden_dims -> codebook_embedding_dim
            encoder_layers = []
            dims = [config.embedding_dim] + config.encoder_hidden_dims + [config.codebook_embedding_dim]

            for i in range(len(dims) - 2):
                encoder_layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.SiLU()])
            encoder_layers.append(nn.Linear(dims[-2], dims[-1]))

            self.encoder = nn.Sequential(*encoder_layers)

            # Build decoder: codebook_embedding_dim -> encoder_hidden_dims (reversed) -> item_embedding_dim
            decoder_layers = []
            dims_reversed = [config.codebook_embedding_dim] + config.encoder_hidden_dims[::-1] + [config.embedding_dim]

            for i in range(len(dims_reversed) - 2):
                decoder_layers.extend([nn.Linear(dims_reversed[i], dims_reversed[i + 1]), nn.SiLU()])
            decoder_layers.append(nn.Linear(dims_reversed[-2], dims_reversed[-1]))

            self.decoder = nn.Sequential(*decoder_layers)
        else:
            self.encoder = None
            self.decoder = None

        # Create quantization layers
        self.vq_layers = nn.ModuleList([VectorQuantizer(config) for _ in range(config.codebook_quantization_levels)])


    def print_architecture(self):
        """Print model architecture and parameter counts."""
        # Encoder
        if self.use_encoder_decoder:
            logger.info("Encoder Architecture:")
            for i, layer in enumerate(self.encoder):
                if isinstance(layer, nn.Linear):
                    logger.info(f"  Layer {i}: Linear({layer.in_features} -> {layer.out_features})")
                else:
                    logger.info(f"  Layer {i}: {layer.__class__.__name__}")
            encoder_params = sum(p.numel() for p in self.encoder.parameters())
            logger.info(f"Total Encoder Parameters: {encoder_params:,}")
        else:
            logger.info("No encoder used (input embeddings are directly quantized)")
        # RQ
        logger.info("Residual Quantization Layers:")
        for i, vq_layer in enumerate(self.vq_layers):
            logger.info(f"  VQ Layer {i}: codebook_size={vq_layer.codebook_size}, embedding_dim={vq_layer.codebook_embedding_dim}, use_rotation_trick={vq_layer.use_rotation_trick}")
            vq_params = sum(p.numel() for p in vq_layer.parameters())
            logger.info(f"    Parameters: {vq_params:,}")
        # Decoder
        if self.use_encoder_decoder:
            logger.info("Decoder Architecture:")
            for i, layer in enumerate(self.decoder):
                if isinstance(layer, nn.Linear):
                    logger.info(f"  Layer {i}: Linear({layer.in_features} -> {layer.out_features})")
                else:
                    logger.info(f"  Layer {i}: {layer.__class__.__name__}")
            decoder_params = sum(p.numel() for p in self.decoder.parameters())
            logger.info(f"Total Decoder Parameters: {decoder_params:,}")
        else:
            logger.info("No decoder used (quantized embeddings are directly used as output)")
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total RQ-VAE Parameters: {total_params:,}")
        logger.info("===========================")


    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation."""
        if self.use_encoder_decoder:
            return self.encoder(x)
        return x

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to output."""
        if self.use_encoder_decoder:
            return self.decoder(z)
        return z

    def forward(self, x: Tensor, query: Tensor = None, neg: Tensor = None) -> Tuple[Tensor, List[Tensor], dict]:
        """Full forward pass through each quantization codebook."""
        z = self.encode(x)

        # Residual Quantization
        quantized_out = torch.zeros_like(z)
        residual = z

        all_indices = []
        vq_loss = 0
        codebook_losses = []
        commitment_losses = []

        for vq_layer in self.vq_layers:
            vq_output = vq_layer(residual)  # Quantize current residual
            residual = residual - vq_output.quantized.detach()  # Update residual for next level
            quantized_out = quantized_out + vq_output.quantized_st  # Accumulate quantized vectors
            all_indices.append(vq_output.indices)

            vq_loss = vq_loss + vq_output.loss  # Store indices and accumulate loss
            if vq_output.codebook_loss is not None:  # Track individual loss components
                codebook_losses.append(vq_output.codebook_loss)
            if vq_output.commitment_loss is not None:
                commitment_losses.append(vq_output.commitment_loss)

        x_recon = self.decode(quantized_out)
        recon_loss = F.mse_loss(x_recon, x)  # Reconstruction loss
        loss = self.config.recon_weight * recon_loss + self.config.vq_weight * vq_loss  # Total loss

        # residual quantization of query and negative samples if provided
        if query is not None:
            z_q = self.encode(query)
            if self.config.ir_query_processing or not self.config.ir_loss_on_latent:  # Only apply RQ to query if we are using it in the loss calculation, otherwise we can just use the original query embedding for the IR loss calculation
                quantized_q_out = torch.zeros_like(z_q)
                residual_q = z_q

                all_q_indices = []
                vq_loss_q = 0
                codebook_losses_q = []
                commitment_losses_q = []

                for vq_layer in self.vq_layers:
                    vq_output_q = vq_layer(residual_q)  # Quantize current residual
                    residual_q = residual_q - vq_output_q.quantized.detach()  # Update residual for next level
                    quantized_q_out = quantized_q_out + vq_output_q.quantized_st  # Accumulate quantized vectors
                    all_q_indices.append(vq_output_q.indices)

                    vq_loss_q = vq_loss_q + vq_output_q.loss  # Store indices and accumulate loss
                    if vq_output_q.codebook_loss is not None:  # Track individual loss components
                        codebook_losses_q.append(vq_output_q.codebook_loss)
                    if vq_output_q.commitment_loss is not None:
                        commitment_losses_q.append(vq_output_q.commitment_loss)
                
                q_recon = self.decode(quantized_q_out)
                recon_loss_q = F.mse_loss(q_recon, query)  # Reconstruction loss
                if self.config.ir_query_processing:
                    loss = loss + self.config.recon_weight * recon_loss_q + self.config.vq_weight * vq_loss_q  # Total loss
            else:
                quantized_q_out = z_q  # Use original query embedding for IR loss calculation
                # don't update loss here since we are not applying RQ on the query, but we will still need to calculate the IR loss later

        
        if neg is not None:
            z_neg = self.encode(neg)
            quantized_neg_out = torch.zeros_like(z_neg)
            residual_neg = z_neg

            all_neg_indices = []
            vq_loss_neg = 0
            codebook_losses_neg = []
            commitment_losses_neg = []

            for vq_layer in self.vq_layers:
                vq_output_neg = vq_layer(residual_neg)  # Quantize current residual
                residual_neg = residual_neg - vq_output_neg.quantized.detach()  # Update residual for next level
                quantized_neg_out = quantized_neg_out + vq_output_neg.quantized_st  # Accumulate quantized vectors
                all_neg_indices.append(vq_output_neg.indices)

                vq_loss_neg = vq_loss_neg + vq_output_neg.loss  # Store indices and accumulate loss
                if vq_output_neg.codebook_loss is not None:  # Track individual loss components
                    codebook_losses_neg.append(vq_output_neg.codebook_loss)
                if vq_output_neg.commitment_loss is not None:
                    commitment_losses_neg.append(vq_output_neg.commitment_loss)
            
            neg_recon = self.decode(quantized_neg_out)
            recon_loss_neg = F.mse_loss(neg_recon, neg)  # Reconstruction loss
            loss = loss + self.config.recon_weight * recon_loss_neg + self.config.vq_weight * vq_loss_neg  # Total loss
        
        # calculate the ir loss based on recon(q), recon(x), recon(neg)
        if query is not None and neg is not None:
            if self.config.ir_loss_type == "infonce":
                # InfoNCE loss: -log(exp(sim_pos) / (exp(sim_pos) + exp(sim_neg)))
                if self.config.ir_loss_on_latent:
                    pos_sim = F.cosine_similarity(quantized_q_out, quantized_out, dim=-1)
                    neg_sim = F.cosine_similarity(quantized_q_out, quantized_neg_out, dim=-1)
                else:
                    pos_sim = F.cosine_similarity(q_recon, x_recon, dim=-1)
                    neg_sim = F.cosine_similarity(q_recon, neg_recon, dim=-1)
                
                pos_sim = torch.exp(pos_sim)
                neg_sim = torch.exp(neg_sim)

                ir_loss = pos_sim / (pos_sim + neg_sim + 1e-8)
                ir_loss = -torch.log(ir_loss).mean()
            elif self.config.ir_loss_type == "triplet":
                # Triplet loss: max(0, ||q_recon - x_recon||^2 - ||q_recon - neg_recon||^2 + margin)
                if self.config.ir_loss_on_latent:
                    pos_dist = torch.norm(quantized_q_out - quantized_out, p=2, dim=-1) ** 2
                    neg_dist = torch.norm(quantized_q_out - quantized_neg_out, p=2, dim=-1) ** 2
                else:
                    pos_dist = torch.norm(q_recon - x_recon, p=2, dim=-1) ** 2
                    neg_dist = torch.norm(q_recon - neg_recon, p=2, dim=-1) ** 2
                
                ir_loss = torch.clamp(pos_dist - neg_dist + self.config.ir_loss_margin, min=0).mean()
            else:
                raise ValueError(f"Unknown IR loss type: {self.config.ir_loss_type}")
            
            loss = loss + ir_loss * self.config.ir_weight

        loss_dict = {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "codebook_losses": codebook_losses,  # List of losses per level
            "commitment_losses": commitment_losses,  # List of commitment losses per level
            "recon_loss_q": recon_loss_q if (query is not None and self.config.ir_query_processing) else None,
            "vq_loss_q": vq_loss_q if (query is not None and self.config.ir_query_processing) else None,
            "codebook_losses_q": codebook_losses_q if (query is not None and self.config.ir_query_processing) else None,
            "commitment_losses_q": commitment_losses_q if (query is not None and self.config.ir_query_processing) else None,
            "recon_loss_neg": recon_loss_neg if neg is not None else None,
            "vq_loss_neg": vq_loss_neg if neg is not None else None,
            "codebook_losses_neg": codebook_losses_neg if neg is not None else None,
            "commitment_losses_neg": commitment_losses_neg if neg is not None else None,
            "ir_loss": ir_loss if query is not None and neg is not None else None,
            "indices": all_indices,  # Store for metric computation
            "residual": residual,  # Store for residual norm calculation
        }

        return x_recon, all_indices, loss_dict

    def encode_to_semantic_ids(self, x: Tensor) -> Tensor:
        """Extract semantic IDs for input batch."""
        with torch.no_grad():
            z = self.encode(x)
            residual = z
            indices_list = []

            for vq_layer in self.vq_layers:
                vq_output = vq_layer(residual)
                indices_list.append(vq_output.indices)
                residual = residual - vq_output.quantized

            # Stack indices from all levels
            semantic_ids = torch.stack(indices_list, dim=-1)
        return semantic_ids

    def decode_from_semantic_ids(self, semantic_ids: Tensor) -> Tensor:
        """Decode from semantic IDs."""
        with torch.no_grad():
            # semantic_ids shape: [batch, codebook_quantization_levels]
            quantized_sum = torch.zeros(semantic_ids.shape[0], self.codebook_embedding_dim, device=semantic_ids.device)

            for level, indices in enumerate(semantic_ids.unbind(dim=-1)):
                codes = self.vq_layers[level].embedding(indices)
                quantized_sum += codes

            return self.decode(quantized_sum)

    def calculate_unique_ids_proportion(self, semantic_ids: Tensor) -> float:
        """Calculate proportion of unique semantic IDs in a batch.

        Args:
            semantic_ids: Tensor of shape [batch_size, codebook_quantization_levels]

        Returns:
            Proportion of items with unique semantic IDs (0 to 1)
        """
        batch_size = semantic_ids.shape[0]
        if batch_size <= 1:
            return 1.0

        # More memory-efficient approach: convert each ID sequence to a unique hash
        # Instead of creating a [B, B] comparison matrix, we hash each sequence
        batch_size = semantic_ids.shape[0]
        
        # For small batches, use the old method
        if batch_size <= 1024:
            # Compare all pairs of semantic IDs
            ids_expanded_1 = semantic_ids.unsqueeze(1)  # [B, 1, L]
            ids_expanded_2 = semantic_ids.unsqueeze(0)  # [1, B, L]
            matches = (ids_expanded_1 == ids_expanded_2).all(dim=-1)  # [B, B]
            upper_tri_matches = torch.triu(matches, diagonal=1)
            has_duplicate = upper_tri_matches.any(dim=1)
            n_unique = (~has_duplicate).sum().item()
        else:
            # For large batches, use hashing approach (more memory efficient)
            # Convert each sequence to a tuple and use set to count unique
            ids_list = semantic_ids.cpu().tolist()
            unique_ids = set(tuple(seq) for seq in ids_list)
            n_unique = len(unique_ids)

        return n_unique / batch_size

    def calculate_codebook_usage(self) -> List[float]:
        """Get codebook usage rate for each level.

        Returns:
            List of usage percentages for each quantization level
        """
        return [vq_layer.get_usage_rate() for vq_layer in self.vq_layers]

    def calculate_avg_residual_norm(self, residual: Tensor) -> float:
        """Calculate average residual norm after quantization.

        Args:
            residual: Final residual tensor after all quantization levels

        Returns:
            Average L2 norm of the residual
        """
        return residual.norm(dim=-1).mean().item()

    def kmeans_init(self, data_loader, device):
        """Initialize codebooks using k-means on first batch."""
        logger.info("Initializing codebooks with k-means clustering...")
        # Get first batch
        first_batch = next(iter(data_loader))
        if isinstance(first_batch, (list, tuple)):
            first_batch = first_batch[0]
        first_batch = first_batch.to(device)

        # Encode to latent space
        with torch.no_grad():
            z = self.encode(first_batch)
                        
            # Initialize each level's codebook
            residual = z
            for level, vq_layer in enumerate(self.vq_layers):
                residual_np = residual.cpu().numpy().reshape(-1, self.codebook_embedding_dim)  # Flatten for k-means
                # kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, random_state=0)
                kmeans = MiniBatchKMeans(
                    n_clusters=self.codebook_size,
                    random_state=42,
                    batch_size=1024,
                    init="k-means++",
                    n_init="auto",
                    max_iter=100,
                    verbose=0
                )
                kmeans.fit(residual_np)  # Run k-means

                vq_layer.embedding.weight.data = torch.from_numpy(kmeans.cluster_centers_).to(device)  # Update codebook
                logger.info(f"  Level {level}: initialized {self.codebook_size} codes")

                if level < self.codebook_quantization_levels - 1:  # Compute next residual
                    vq_output = vq_layer(residual)
                    residual = residual - vq_output.quantized

    def train_rqvae(
        self,
        data_loader: torch.utils.data.DataLoader,
        config: RQVAEConfig,
        device: str = "cpu",
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        """Train RQVAE model with simplified logging.

        Args:
            self: RQVAE model to train
            data_loader: Training data loader
            config: RQVAEConfig object containing all training parameters
            device: Device to train on
            val_loader: Optional validation data loader
        """
        self = self.to(device)
        if config.use_kmeans_init:
            self.kmeans_init(data_loader, device)

        # Apply torch.compile for faster training (CUDA only, not MPS)
        if device == "cuda" and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile for faster training...")
            self = torch.compile(self)
        else:
            logger.info("torch.compile not available or not using CUDA, skipping compilation")

        # Better optimizer choice for RQ-VAE with fused support (PyTorch 2.0+)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        
        if use_fused:
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.max_lr, weight_decay=0.01, fused=True)
            logger.info("Using fused AdamW optimizer for faster training")
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.max_lr, weight_decay=0.01)
            if device == "cuda" and not fused_available:
                logger.info("Fused AdamW not available (requires PyTorch 2.0+)")

        # Calculate total training steps
        steps_per_epoch = len(data_loader) // config.gradient_accumulation_steps
        total_steps = steps_per_epoch * config.num_epochs
        logger.info(f"Total training steps: {total_steps:,} ({steps_per_epoch} steps/epoch x {config.num_epochs} epochs)")

        # Learning rate scheduler
        if config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.min_lr)
            logger.info(f"Cosine annealing: {config.max_lr:.1e} -> {config.min_lr:.1e} for {total_steps:,} steps")
        elif config.scheduler_type == "cosine_with_warmup":
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.warmup_start_lr / config.max_lr,
                total_iters=config.warmup_steps,
            )
            logger.info(f"Warmup: {config.warmup_start_lr:.1e} -> {config.max_lr:.1e} for {config.warmup_steps:,} steps")

            cosine_steps = total_steps - config.warmup_steps
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=config.min_lr)
            logger.info(f"Cosine annealing: {config.max_lr:.1e} -> {config.min_lr:.1e} for {cosine_steps:,} steps")

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
            )
        else:
            scheduler = None

        # Track best model
        best_loss = float("inf")
        global_step = 0

        for epoch in tqdm.tqdm(range(config.num_epochs), desc="Training Epochs"):
            self.train()

            for batch_idx, data in enumerate(data_loader):
                # Only perform optimizer step every gradient_accumulation_steps
                if batch_idx % config.gradient_accumulation_steps == 0:
                    t0 = time.time()
                    optimizer.zero_grad()
                    loss_accum = 0.0

                # Unpack data (handle both single and triple tensor cases)
                if isinstance(data, (list, tuple)) and len(data) == 3:
                    doc, query, neg = data
                    doc = doc.to(device)
                    query = query.to(device)
                    neg = neg.to(device)
                    x_recon, indices, loss_dict = self(doc, query, neg)
                    batch_size = doc.shape[0]
                else:
                    data = data.to(device)
                    x_recon, indices, loss_dict = self(data)
                    batch_size = data.shape[0]

                loss = loss_dict["loss"] / config.gradient_accumulation_steps
                loss_accum += loss_dict["loss"].detach()  # Accumulate unscaled loss for logging

                loss.backward()

                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    # Get gradient norm before clipping
                    grad_norm_before = self.get_gradient_norm()

                    # Clip gradients to prevent explosion
                    if config.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=config.gradient_clip_norm)
                        grad_norm_after = self.get_gradient_norm()
                    else:
                        grad_norm_after = grad_norm_before

                    optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    t1 = time.time()
                    batch_time_ms = (t1 - t0) * 1000
                    samples_per_second = (batch_size * config.gradient_accumulation_steps) / (t1 - t0)
                    global_step += 1

                    avg_loss = loss_accum / config.gradient_accumulation_steps

                    # Log progress every N steps
                    if global_step == 1 or global_step % config.steps_per_train_log == 0:
                        current_lr = optimizer.param_groups[0]["lr"]

                        # Compute codebook usage and unique IDs for current batch
                        codebook_usage = self.calculate_codebook_usage()
                        semantic_ids = torch.stack(loss_dict["indices"], dim=-1)
                        unique_ids_proportion = self.calculate_unique_ids_proportion(semantic_ids)
                        usage_str = "/".join([f"{u:.2f}" for u in codebook_usage])

                        # get total codebook loss for logging
                        total_codebook_loss = sum(loss_dict["codebook_losses"]).item() if loss_dict["codebook_losses"] else 0.0

                        log_str = (
                            f"Step {global_step:05d} | Epoch {epoch + 1:05d} | lr: {current_lr:.2e} | "
                            f"loss: {avg_loss:.2e} | recon: {loss_dict['recon_loss'].item():.2e} | "
                            f"vq: {loss_dict['vq_loss'].item():.2e} | codebook usage: {usage_str} | "
                            f"unique ids: {unique_ids_proportion:.1%} | time: {batch_time_ms:.0f}ms | "
                            f"samples/s: {samples_per_second:,.0f}"
                        )
                        logger.info(log_str)

                        wandb_log_dict = {
                            "train/loss": avg_loss,
                            "train/reconstruction_loss": loss_dict["recon_loss"].item(),
                            "train/vq_loss": loss_dict["vq_loss"].item(),
                            "train/total_codebook_loss": total_codebook_loss,
                            "train/learning_rate": current_lr,
                            "train/gradient_norm": grad_norm_before,
                            "train/gradient_norm_clipped": grad_norm_after,
                            "train/batch_time_ms": batch_time_ms,
                            "train/samples_per_second": samples_per_second,
                            "train/epoch": epoch + 1,
                            "train/step": global_step,
                        }
                        if loss_dict.get("recon_loss_q") is not None:
                            wandb_log_dict["train/reconstruction_loss_query"] = loss_dict["recon_loss_q"].item()
                        if loss_dict.get("vq_loss_q") is not None:
                            wandb_log_dict["train/vq_loss_query"] = loss_dict["vq_loss_q"].item()
                        if loss_dict.get("recon_loss_neg") is not None:
                            wandb_log_dict["train/reconstruction_loss_negative"] = loss_dict["recon_loss_neg"].item()
                        if loss_dict.get("vq_loss_neg") is not None:
                            wandb_log_dict["train/vq_loss_negative"] = loss_dict["vq_loss_neg"].item()
                        if loss_dict.get("ir_loss") is not None:
                            wandb_log_dict["train/ir_loss"] = loss_dict["ir_loss"].item()

                        # Add per-level losses
                        # if loss_dict["codebook_losses"]:
                        #     for level, codebook_loss in enumerate(loss_dict["codebook_losses"]):
                        #         wandb_log_dict[f"train/codebook_level_{level}_loss"] = codebook_loss.item()
                        # if loss_dict.get("codebook_losses_q"):
                        #     for level, codebook_loss_q in enumerate(loss_dict["codebook_losses_q"]):
                        #         wandb_log_dict[f"train/codebook_level_{level}_loss_query"] = codebook_loss_q.item()
                        # if loss_dict.get("codebook_losses_neg"):
                        #     for level, codebook_loss_neg in enumerate(loss_dict["codebook_losses_neg"]):
                        #         wandb_log_dict[f"train/codebook_level_{level}_loss_negative"] = codebook_loss_neg.item()
                        
                        # Add commitment losses when encoder is used
                        if self.use_encoder_decoder:
                            if loss_dict.get("commitment_losses"):
                                total_commitment = sum(c.item() for c in loss_dict["commitment_losses"])
                                wandb_log_dict["train/total_commitment_loss"] = total_commitment
                                # for level, commitment_loss in enumerate(loss_dict["commitment_losses"]):
                                #     wandb_log_dict[f"train/commitment_level_{level}_loss"] = commitment_loss.item()
                            if loss_dict.get("commitment_losses_q"):
                                total_commitment_q = sum(c.item() for c in loss_dict["commitment_losses_q"])
                                wandb_log_dict["train/total_commitment_loss_query"] = total_commitment_q
                                # for level, commitment_loss_q in enumerate(loss_dict["commitment_losses_q"]):
                                #     wandb_log_dict[f"train/commitment_level_{level}_loss_query"] = commitment_loss_q.item()
                            if loss_dict.get("commitment_losses_neg"):
                                total_commitment_neg = sum(c.item() for c in loss_dict["commitment_losses_neg"])
                                wandb_log_dict["train/total_commitment_loss_negative"] = total_commitment_neg
                                # for level, commitment_loss_neg in enumerate(loss_dict["commitment_losses_neg"]):
                                #     wandb_log_dict[f"train/commitment_level_{level}_loss_negative"] = commitment_loss_neg.item()

                        # Add codebook metrics
                        wandb_log_dict["train/unique_ids_proportion"] = unique_ids_proportion
                        for level, usage in enumerate(codebook_usage):
                            wandb_log_dict[f"train/codebook_usage_train_level_{level}"] = usage

                        wandb.log(wandb_log_dict, step=global_step)

                    # Validation (without checkpointing)
                    # if global_step % config.steps_per_val_log == 0 and val_loader is not None:
                    #     # Run validation and compute all metrics
                    #     metrics = self.evaluate(val_loader, data_loader, device, global_step, epoch + 1)
                    #     self.train()

                    # Codebook reset
                    if config.reset_unused_codes and global_step % config.steps_per_codebook_reset == 0:
                        if config.scheduler_type == "cosine_with_warmup" and global_step < config.warmup_steps:
                            logger.debug(f"Step {global_step:05d} - Skipping codebook reset during warmup")
                        else:
                            self.eval()

                            # Calculate current codebook usage
                            codebook_usage = self.calculate_codebook_usage()
                            usage_str = "/".join([f"{u:.2f}" for u in codebook_usage])
                            reset_performed = []

                            # Get a sample batch for reset
                            reset_batch = next(iter(data_loader))

                            # Reset unused codes for each VQ layer
                            for level, vq_layer in enumerate(self.vq_layers):
                                if codebook_usage[level] < config.codebook_usage_threshold:
                                    with torch.no_grad():
                                        # Unpack reset_batch
                                        if isinstance(reset_batch, (list, tuple)) and len(reset_batch) == 3:
                                            z = reset_batch[0].to(device)
                                        else:
                                            z = reset_batch.to(device)
                                        
                                        # Encode the batch before processing
                                        z_encoded = self.encode(z)
                                        residual = z_encoded
                                        
                                        for i in range(level):
                                            vq_out = self.vq_layers[i](residual)
                                            residual = residual - vq_out.quantized

                                        vq_layer.reset_unused_codes(residual)
                                        reset_performed.append(f"{level}({codebook_usage[level]:.2f})")

                                vq_layer.reset_usage_count()

                            self.train()

            # Handle incomplete gradient accumulation at end of epoch
            if (batch_idx + 1) % config.gradient_accumulation_steps != 0:
                grad_norm = self.get_gradient_norm()

                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=config.gradient_clip_norm)

                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                logger.debug(
                    f"Step {global_step:05d} | Epoch {epoch + 1:03d} - Applied remaining gradients at epoch end (grad norm: {grad_norm:.2f})"
                )

            # End of epoch: run validation and save checkpoint
            if val_loader is not None:
                logger.info(f"End of epoch {epoch + 1} - Running validation and saving checkpoint")
                metrics = self.evaluate(val_loader, data_loader, device, global_step, epoch + 1)
                
                # Save checkpoint and update best model if improved
                best_loss = self.save_checkpoint(
                    optimizer, scheduler, metrics, config, global_step, epoch, best_loss
                )
                
                self.train()


    def get_gradient_norm(self) -> float:
        """Calculate the L2 norm of gradients across all model parameters."""
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
        return total_norm.item()


    def get_loss(self, val_loader: torch.utils.data.DataLoader, device: str) -> float:
        """Validate model on validation set."""
        self.eval()
        total_loss = 0
        batch_count = 0

        with torch.no_grad():
            for data in val_loader:
                # Unpack data (handle both single and triple tensor cases)
                if isinstance(data, (list, tuple)) and len(data) == 3:
                    doc, query, neg = data
                    doc = doc.to(device)
                    query = query.to(device)
                    neg = neg.to(device)
                    _, _, loss_dict = self(doc, query, neg)
                else:
                    if isinstance(data, (list, tuple)):
                        data = data[0]
                    data = data.to(device)                          
                    _, _, loss_dict = self(data)
                total_loss += loss_dict["loss"].item()
                batch_count += 1

        return total_loss / batch_count


    def evaluate(
        self,
        val_loader: DataLoader,
        train_loader: DataLoader,  # For sample batch metrics
        device: str,
        global_step: int,
        epoch: int,
    ) -> dict:
        """Run validation

        Args:
            self: RQVAE model to validate
            val_loader: Validation data loader
            train_loader: Training data loader (for sample batch metrics)
            device: Device to run on
            global_step: Current training step
            epoch: Current epoch (1-indexed)

        Returns:
            Dictionary containing all metrics including val_loss
        """
        self.eval()

        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            # Unpack data (handle both single and triple tensor cases)
            if isinstance(sample_batch, (list, tuple)) and len(sample_batch) == 3:
                doc, query, neg = sample_batch
                doc = doc.to(device)
                query = query.to(device)
                neg = neg.to(device)

                _ , indices, loss_dict_sample = self(doc, query, neg)
                # Log distance statistics for IR loss
                if self.config.ir_loss_type == "triplet":
                    pos_dist = torch.norm(query - doc, p=2, dim=-1) ** 2
                    neg_dist = torch.norm(query - neg, p=2, dim=-1) ** 2
                    logger.info(
                        f"Triplet distances - Pos: min={pos_dist.min()}, mean={pos_dist.mean()}, max={pos_dist.max()} | "
                        f"Neg: min={neg_dist.min()}, mean={neg_dist.mean()}, max={neg_dist.max()}"
                    )
            elif isinstance(sample_batch, (list, tuple)):
                sample_batch = sample_batch[0].to(device)
                _, indices, loss_dict_sample = self(sample_batch)

            else:
                sample_batch = sample_batch.to(device)
                _, indices, loss_dict_sample = self(sample_batch)

            # Compute all metrics
            codebook_usage = self.calculate_codebook_usage()
            avg_residual_norm = self.calculate_avg_residual_norm(loss_dict_sample["residual"])
            semantic_ids = torch.stack(indices, dim=-1)
            unique_ids_proportion = self.calculate_unique_ids_proportion(semantic_ids)

        # Compute validation loss
        val_loss = self.get_loss(val_loader, device)
        usage_str = "/".join([f"{u:.2f}" for u in codebook_usage])

        logger.info(
            f"Step {global_step:05d} | Epoch {epoch:05d} | Val loss: {val_loss:.2e} | "
            # f"Codebook usage: {usage_str} | "
            f"Avg residual norm: {avg_residual_norm:.3f} | "
            f"Unique ids: {unique_ids_proportion:.1%}"
        )

        wandb_dict = {
            "eval/loss": val_loss,
            "eval/avg_residual_norm": avg_residual_norm,
            "eval/unique_ids_proportion": unique_ids_proportion,
        }

        # for level, usage in enumerate(codebook_usage):
        #     wandb_dict[f"eval/codebook_usage_level_{level}"] = usage

        wandb.log(wandb_dict, step=global_step)

        return {
            "val_loss": val_loss,
            "codebook_usage": codebook_usage,
            "codebook_usage_str": usage_str,
            "avg_residual_norm": avg_residual_norm,
            "unique_ids_proportion": unique_ids_proportion,
        }


    def save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: dict,
        config: RQVAEConfig,
        global_step: int,
        epoch: int,
        best_loss: float,
    ) -> float:
        """Save checkpoint and handle best model tracking.

        Args:
            self: Model to save
            optimizer: Optimizer with state to save
            scheduler: Optional scheduler with state to save
            metrics: Dictionary of metrics from validation
            config: Training configuration
            global_step: Current training step
            epoch: Current epoch (0-indexed for checkpoint compatibility)
            best_loss: Current best validation loss

        Returns:
            Updated best_loss value
        """
        # Ensure checkpoint directory exists
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "epoch": epoch,
            "step": global_step,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "val_loss": metrics["val_loss"],
            "config": config.__dict__,
        }

        # Check if this is the best model
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            best_model_path = config.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint_data, best_model_path)
            logger.info(f"Saved best model with val_loss: {best_loss:.4e}")

        return best_loss


# ---------------------------------------------------------------------------
# Product Quantization + Residual Quantized VAE (PRQVAE)
# ---------------------------------------------------------------------------

@dataclass
class PRQVAEConfig:
    """Configuration for PRQVAE (Product Quantization + Residual Quantized VAE).

    Mirrors RQVAEConfig but without encoder/decoder or IR-loss fields.
    """

    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "prqvae")

    # Model parameters
    embedding_dim: int = 768          # Input embedding dimension (must be divisible by nb_subspaces)
    nb_subspaces: int = 4             # Number of PQ subspaces
    codebook_quantization_levels: int = 3  # Number of RQ levels per subspace
    codebook_size: int = 16           # Codebook size per level
    use_rotation_trick: bool = True
    recon_weight: float = 1.0
    vq_weight: float = 1.0

    # Training parameters
    batch_size: int = 32768
    gradient_accumulation_steps: int = 1
    num_epochs: int = 200
    scheduler_type: str = "cosine_with_warmup"
    warmup_start_lr: float = 1e-8
    warmup_steps: int = 10
    max_lr: float = 3e-4
    min_lr: float = 1e-6
    use_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    use_kmeans_init: bool = True
    reset_unused_codes: bool = True
    steps_per_codebook_reset: int = 2
    codebook_usage_threshold: float = 1.0
    val_split: float = 0.05

    # Logging and checkpointing
    steps_per_train_log: int = 5
    steps_per_val_log: int = 20

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== PRQVAE Configuration ===")
        logger.info("Data Settings:")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")
        logger.info("Model Parameters:")
        logger.info(f"  embedding_dim: {self.embedding_dim}")
        logger.info(f"  nb_subspaces: {self.nb_subspaces}")
        logger.info(f"  subspace_dim: {self.embedding_dim // self.nb_subspaces}")
        logger.info(f"  codebook_quantization_levels: {self.codebook_quantization_levels}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  total_codes_per_item: {self.nb_subspaces * self.codebook_quantization_levels}")
        logger.info(f"  use_rotation_trick: {self.use_rotation_trick}")
        logger.info(f"  recon_weight: {self.recon_weight}")
        logger.info(f"  vq_weight: {self.vq_weight}")
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  warmup_steps: {self.warmup_steps}")
        logger.info(f"  max_lr: {self.max_lr}")
        logger.info(f"  min_lr: {self.min_lr}")
        logger.info(f"  use_gradient_clipping: {self.use_gradient_clipping}")
        logger.info(f"  gradient_clip_norm: {self.gradient_clip_norm}")
        logger.info(f"  use_kmeans_init: {self.use_kmeans_init}")
        logger.info(f"  reset_unused_codes: {self.reset_unused_codes}")
        logger.info(f"  steps_per_codebook_reset: {self.steps_per_codebook_reset}")
        logger.info(f"  codebook_usage_threshold: {self.codebook_usage_threshold}")
        logger.info(f"  val_split: {self.val_split}")
        logger.info("Logging and Checkpointing:")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info("===========================")


class PRQVAE(nn.Module):
    """Product Quantization + Residual Quantized VAE.

    Divides input embeddings into ``nb_subspaces`` equal subspaces and trains
    an independent RQVAE (no encoder/decoder, no IR loss) on each subspace.

    Semantic IDs have ``nb_subspaces * codebook_quantization_levels`` codes,
    interleaved so that position ``j * nb_subspaces + i`` holds the level-j
    code of subspace i — the same convention used by PQRQ in
    residual_quantizer.py.
    """

    def __init__(self, config: PRQVAEConfig):
        super().__init__()
        assert config.embedding_dim % config.nb_subspaces == 0, (
            f"embedding_dim ({config.embedding_dim}) must be divisible by "
            f"nb_subspaces ({config.nb_subspaces})"
        )

        self.config = config
        self.nb_subspaces = config.nb_subspaces
        self.embedding_dim = config.embedding_dim
        self.subspace_dim = config.embedding_dim // config.nb_subspaces
        self.codebook_quantization_levels = config.codebook_quantization_levels
        self.codebook_size = config.codebook_size

        # One RQVAE per subspace (no encoder/decoder, no IR loss)
        subspace_config = RQVAEConfig(
            embedding_dim=self.subspace_dim,
            codebook_embedding_dim=self.subspace_dim,
            codebook_quantization_levels=config.codebook_quantization_levels,
            codebook_size=config.codebook_size,
            use_rotation_trick=config.use_rotation_trick,
            recon_weight=config.recon_weight,
            vq_weight=config.vq_weight,
            use_encoder_decoder=False,
        )
        self.subspaces = nn.ModuleList([RQVAE(subspace_config) for _ in range(config.nb_subspaces)])

    # ------------------------------------------------------------------
    # Core forward / encode / decode
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], dict]:
        """Quantize each subspace independently and aggregate losses.

        Returns the same (x_recon, all_indices, loss_dict) tuple as RQVAE,
        where all_indices is the interleaved list of length
        ``nb_subspaces * codebook_quantization_levels``.

        Losses are averaged over subspaces so that the scale stays
        comparable to a single RQVAE trained on the full embedding.
        """
        subspace_inputs = x.split(self.subspace_dim, dim=-1)

        sub_total_losses: List[Tensor] = []
        sub_recon_losses: List[Tensor] = []
        sub_vq_losses: List[Tensor] = []
        all_codebook_losses: List[Tensor] = []

        recon_parts: List[Tensor] = []
        residual_parts: List[Tensor] = []
        # all_subspace_indices[i] = list of codebook_quantization_levels tensors [batch]
        all_subspace_indices: List[List[Tensor]] = []

        for subspace, sub_x in zip(self.subspaces, subspace_inputs):
            sub_recon, sub_indices, sub_loss_dict = subspace(sub_x)
            recon_parts.append(sub_recon)
            all_subspace_indices.append(sub_indices)
            residual_parts.append(sub_loss_dict["residual"])
            sub_total_losses.append(sub_loss_dict["loss"])
            sub_recon_losses.append(sub_loss_dict["recon_loss"])
            sub_vq_losses.append(sub_loss_dict["vq_loss"])
            all_codebook_losses.extend(sub_loss_dict["codebook_losses"])

        x_recon = torch.cat(recon_parts, dim=-1)
        residual = torch.cat(residual_parts, dim=-1)

        # Average over subspaces to keep the loss scale comparable to RQVAE
        total_loss = sum(sub_total_losses) / self.nb_subspaces
        total_recon_loss = sum(sub_recon_losses) / self.nb_subspaces
        total_vq_loss = sum(sub_vq_losses) / self.nb_subspaces

        # Interleave indices: position j * nb_subspaces + i = level j of subspace i
        interleaved_indices: List[Tensor] = []
        for j in range(self.codebook_quantization_levels):
            for i in range(self.nb_subspaces):
                interleaved_indices.append(all_subspace_indices[i][j])

        loss_dict = {
            "loss": total_loss,
            "recon_loss": total_recon_loss,
            "vq_loss": total_vq_loss,
            "codebook_losses": all_codebook_losses,
            "commitment_losses": [],
            "indices": interleaved_indices,
            "residual": residual,
        }

        return x_recon, interleaved_indices, loss_dict

    def encode_to_semantic_ids(self, x: Tensor) -> Tensor:
        """Encode x to interleaved semantic IDs.

        Returns:
            Tensor of shape [batch, nb_subspaces * codebook_quantization_levels]
        """
        with torch.no_grad():
            subspace_inputs = x.split(self.subspace_dim, dim=-1)
            batch_size = x.shape[0]
            total_codes = self.nb_subspaces * self.codebook_quantization_levels
            semantic_ids = torch.zeros(batch_size, total_codes, dtype=torch.long, device=x.device)

            for i, (subspace, sub_x) in enumerate(zip(self.subspaces, subspace_inputs)):
                sub_ids = subspace.encode_to_semantic_ids(sub_x)  # [batch, codebook_quantization_levels]
                for j in range(self.codebook_quantization_levels):
                    semantic_ids[:, j * self.nb_subspaces + i] = sub_ids[:, j]

        return semantic_ids

    def decode_from_semantic_ids(self, semantic_ids: Tensor) -> Tensor:
        """Decode interleaved semantic IDs back to approximate embeddings.

        Args:
            semantic_ids: shape [batch, nb_subspaces * codebook_quantization_levels]
        """
        with torch.no_grad():
            recon_parts = []
            for i, subspace in enumerate(self.subspaces):
                # Codes for subspace i are at positions i, nb_subspaces+i, 2*nb_subspaces+i, ...
                sub_ids = semantic_ids[:, i :: self.nb_subspaces]  # [batch, codebook_quantization_levels]
                recon_parts.append(subspace.decode_from_semantic_ids(sub_ids))
            return torch.cat(recon_parts, dim=-1)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_unique_ids_proportion(self, semantic_ids: Tensor) -> float:
        """Calculate proportion of unique semantic IDs in a batch."""
        batch_size = semantic_ids.shape[0]
        if batch_size <= 1:
            return 1.0
        if batch_size <= 1024:
            ids_expanded_1 = semantic_ids.unsqueeze(1)
            ids_expanded_2 = semantic_ids.unsqueeze(0)
            matches = (ids_expanded_1 == ids_expanded_2).all(dim=-1)
            upper_tri_matches = torch.triu(matches, diagonal=1)
            has_duplicate = upper_tri_matches.any(dim=1)
            n_unique = (~has_duplicate).sum().item()
        else:
            ids_list = semantic_ids.cpu().tolist()
            unique_ids = set(tuple(seq) for seq in ids_list)
            n_unique = len(unique_ids)
        return n_unique / batch_size

    def calculate_codebook_usage(self) -> List[float]:
        """Return usage rate for each (level, subspace) pair, interleaved.

        Position j * nb_subspaces + i corresponds to level j of subspace i.
        """
        usage = []
        for j in range(self.codebook_quantization_levels):
            for i, subspace in enumerate(self.subspaces):
                usage.append(subspace.vq_layers[j].get_usage_rate())
        return usage

    def calculate_avg_residual_norm(self, residual: Tensor) -> float:
        """Calculate average residual norm after quantization."""
        return residual.norm(dim=-1).mean().item()

    def get_gradient_norm(self) -> float:
        """Calculate the L2 norm of gradients across all model parameters."""
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
        return total_norm.item()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def kmeans_init(self, data_loader, device):
        """Initialise codebooks with k-means using the first batch."""
        logger.info("Initialising PRQVAE codebooks with k-means clustering...")
        first_batch = next(iter(data_loader))
        if isinstance(first_batch, (list, tuple)):
            first_batch = first_batch[0]
        first_batch = first_batch.to(device)

        subspace_batches = first_batch.split(self.subspace_dim, dim=-1)

        with torch.no_grad():
            for i, (subspace, sub_z) in enumerate(zip(self.subspaces, subspace_batches)):
                logger.info(f"  Subspace {i + 1}/{self.nb_subspaces}:")
                residual = sub_z
                for level, vq_layer in enumerate(subspace.vq_layers):
                    residual_np = residual.cpu().numpy().reshape(-1, self.subspace_dim)
                    kmeans = MiniBatchKMeans(
                        n_clusters=self.codebook_size,
                        random_state=42,
                        batch_size=1024,
                        init="k-means++",
                        n_init="auto",
                        max_iter=100,
                        verbose=0,
                    )
                    kmeans.fit(residual_np)
                    vq_layer.embedding.weight.data = torch.from_numpy(kmeans.cluster_centers_).to(device)
                    logger.info(f"    Level {level}: initialised {self.codebook_size} codes")

                    if level < self.codebook_quantization_levels - 1:
                        vq_output = vq_layer(residual)
                        residual = residual - vq_output.quantized

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def get_loss(self, val_loader: DataLoader, device: str) -> float:
        """Compute average loss on a data loader."""
        self.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for data in val_loader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                data = data.to(device)
                _, _, loss_dict = self(data)
                total_loss += loss_dict["loss"].item()
                batch_count += 1
        return total_loss / batch_count

    def evaluate(
        self,
        val_loader: DataLoader,
        train_loader: DataLoader,
        device: str,
        global_step: int,
        epoch: int,
    ) -> dict:
        """Run validation and log metrics to wandb."""
        self.eval()
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, (list, tuple)):
                sample_batch = sample_batch[0]
            sample_batch = sample_batch.to(device)
            _, indices, loss_dict_sample = self(sample_batch)

            codebook_usage = self.calculate_codebook_usage()
            avg_residual_norm = self.calculate_avg_residual_norm(loss_dict_sample["residual"])
            semantic_ids = torch.stack(indices, dim=-1)
            unique_ids_proportion = self.calculate_unique_ids_proportion(semantic_ids)

        val_loss = self.get_loss(val_loader, device)
        usage_str = "/".join([f"{u:.2f}" for u in codebook_usage])

        logger.info(
            f"Step {global_step:05d} | Epoch {epoch:05d} | Val loss: {val_loss:.2e} | "
            f"Avg residual norm: {avg_residual_norm:.3f} | "
            f"Unique ids: {unique_ids_proportion:.1%}"
        )

        wandb_dict = {
            "eval/loss": val_loss,
            "eval/avg_residual_norm": avg_residual_norm,
            "eval/unique_ids_proportion": unique_ids_proportion,
        }
        wandb.log(wandb_dict, step=global_step)

        return {
            "val_loss": val_loss,
            "codebook_usage": codebook_usage,
            "codebook_usage_str": usage_str,
            "avg_residual_norm": avg_residual_norm,
            "unique_ids_proportion": unique_ids_proportion,
        }

    def save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: dict,
        config: "PRQVAEConfig",
        global_step: int,
        epoch: int,
        best_loss: float,
    ) -> float:
        """Save a checkpoint and update best model if validation loss improved."""
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            "epoch": epoch,
            "step": global_step,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "val_loss": metrics["val_loss"],
            "config": config.__dict__,
        }
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            best_model_path = config.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint_data, best_model_path)
            logger.info(f"Saved best model with val_loss: {best_loss:.4e}")
        return best_loss

    def train_rqvae(
        self,
        data_loader: DataLoader,
        config: "PRQVAEConfig",
        device: str = "cpu",
        val_loader: Optional[DataLoader] = None,
    ):
        """Train PRQVAE model.

        Args:
            data_loader: Training data loader. Each batch may be a single tensor
                or a tuple/list whose first element is the document embedding.
            config: PRQVAEConfig controlling all training hyper-parameters.
            device: Device string ("cpu", "cuda", "mps", …).
            val_loader: Optional validation data loader.
        """
        self = self.to(device)
        if config.use_kmeans_init:
            self.kmeans_init(data_loader, device)

        if device == "cuda" and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile for faster training...")
            self = torch.compile(self)
        else:
            logger.info("torch.compile not available or not using CUDA, skipping compilation")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        if use_fused:
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.max_lr, weight_decay=0.01, fused=True)
            logger.info("Using fused AdamW optimizer for faster training")
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.max_lr, weight_decay=0.01)

        steps_per_epoch = len(data_loader) // config.gradient_accumulation_steps
        total_steps = steps_per_epoch * config.num_epochs
        logger.info(f"Total training steps: {total_steps:,} ({steps_per_epoch} steps/epoch x {config.num_epochs} epochs)")

        if config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.min_lr)
            logger.info(f"Cosine annealing: {config.max_lr:.1e} -> {config.min_lr:.1e} for {total_steps:,} steps")
        elif config.scheduler_type == "cosine_with_warmup":
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.warmup_start_lr / config.max_lr,
                total_iters=config.warmup_steps,
            )
            cosine_steps = total_steps - config.warmup_steps
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=config.min_lr)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
            )
            logger.info(f"Warmup {config.warmup_steps} steps then cosine: {config.max_lr:.1e} -> {config.min_lr:.1e}")
        else:
            scheduler = None

        best_loss = float("inf")
        global_step = 0

        for epoch in tqdm.tqdm(range(config.num_epochs), desc="Training Epochs"):
            self.train()

            for batch_idx, data in enumerate(data_loader):
                if batch_idx % config.gradient_accumulation_steps == 0:
                    t0 = time.time()
                    optimizer.zero_grad()
                    loss_accum = 0.0

                if isinstance(data, (list, tuple)):
                    data = data[0]
                data = data.to(device)
                batch_size = data.shape[0]

                _, indices, loss_dict = self(data)
                loss = loss_dict["loss"] / config.gradient_accumulation_steps
                loss_accum += loss_dict["loss"].detach()
                loss.backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    grad_norm_before = self.get_gradient_norm()
                    if config.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=config.gradient_clip_norm)
                        grad_norm_after = self.get_gradient_norm()
                    else:
                        grad_norm_after = grad_norm_before

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    t1 = time.time()
                    batch_time_ms = (t1 - t0) * 1000
                    samples_per_second = (batch_size * config.gradient_accumulation_steps) / (t1 - t0)
                    global_step += 1
                    avg_loss = loss_accum / config.gradient_accumulation_steps

                    if global_step == 1 or global_step % config.steps_per_train_log == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        codebook_usage = self.calculate_codebook_usage()
                        semantic_ids = torch.stack(indices, dim=-1)
                        unique_ids_proportion = self.calculate_unique_ids_proportion(semantic_ids)
                        usage_str = "/".join([f"{u:.2f}" for u in codebook_usage])

                        log_str = (
                            f"Step {global_step:05d} | Epoch {epoch + 1:05d} | lr: {current_lr:.2e} | "
                            f"loss: {avg_loss:.2e} | recon: {loss_dict['recon_loss'].item():.2e} | "
                            f"vq: {loss_dict['vq_loss'].item():.2e} | codebook usage: {usage_str} | "
                            f"unique ids: {unique_ids_proportion:.1%} | time: {batch_time_ms:.0f}ms | "
                            f"samples/s: {samples_per_second:,.0f}"
                        )
                        logger.info(log_str)

                        wandb_log_dict = {
                            "train/loss": avg_loss,
                            "train/reconstruction_loss": loss_dict["recon_loss"].item(),
                            "train/vq_loss": loss_dict["vq_loss"].item(),
                            "train/learning_rate": current_lr,
                            "train/gradient_norm": grad_norm_before,
                            "train/gradient_norm_clipped": grad_norm_after,
                            "train/batch_time_ms": batch_time_ms,
                            "train/samples_per_second": samples_per_second,
                            "train/epoch": epoch + 1,
                            "train/step": global_step,
                            "train/unique_ids_proportion": unique_ids_proportion,
                        }
                        for idx, usage in enumerate(codebook_usage):
                            wandb_log_dict[f"train/codebook_usage_{idx}"] = usage
                        wandb.log(wandb_log_dict, step=global_step)

                    # Codebook reset
                    if config.reset_unused_codes and global_step % config.steps_per_codebook_reset == 0:
                        if config.scheduler_type == "cosine_with_warmup" and global_step < config.warmup_steps:
                            logger.debug(f"Step {global_step:05d} - Skipping codebook reset during warmup")
                        else:
                            self.eval()
                            codebook_usage = self.calculate_codebook_usage()

                            reset_batch = next(iter(data_loader))
                            if isinstance(reset_batch, (list, tuple)):
                                reset_batch = reset_batch[0]
                            reset_batch = reset_batch.to(device)
                            subspace_batches = reset_batch.split(self.subspace_dim, dim=-1)

                            for i, (subspace, sub_z) in enumerate(zip(self.subspaces, subspace_batches)):
                                for level, vq_layer in enumerate(subspace.vq_layers):
                                    usage_idx = level * self.nb_subspaces + i
                                    if codebook_usage[usage_idx] < config.codebook_usage_threshold:
                                        with torch.no_grad():
                                            residual = sub_z
                                            for prev_level in range(level):
                                                vq_out = subspace.vq_layers[prev_level](residual)
                                                residual = residual - vq_out.quantized
                                            vq_layer.reset_unused_codes(residual)
                                    vq_layer.reset_usage_count()

                            self.train()

            # Handle incomplete gradient accumulation at end of epoch
            if (batch_idx + 1) % config.gradient_accumulation_steps != 0:
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=config.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            # End of epoch: validate and checkpoint
            if val_loader is not None:
                logger.info(f"End of epoch {epoch + 1} - Running validation and saving checkpoint")
                metrics = self.evaluate(val_loader, data_loader, device, global_step, epoch + 1)
                best_loss = self.save_checkpoint(
                    optimizer, scheduler, metrics, config, global_step, epoch, best_loss
                )
                self.train()


if __name__ == "__main__":
    config = RQVAEConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = f"rqvae-L{config.codebook_quantization_levels}-C{config.codebook_size}-D{config.embedding_dim}"
    run = wandb.init(project="ddro-training-alexia", name=run_name, config=config.__dict__)
    config.log_config()

    embeddings_test = np.random.rand(50, config.embedding_dim).astype(np.float32)
    print(embeddings_test, embeddings_test.shape)

    dataset = EmbeddingDataset(embeddings=embeddings_test, limit=None)

    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train size: {len(train_dataset):,}, Val size: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced to avoid iterator issues
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,  # Pre-fetch next batches
        persistent_workers=False,  # Disabled to fix infinite loop
        drop_last=False,  # Include partial batches to avoid losing data
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=False,
    )

    model = RQVAE(config)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train_rqvae(data_loader=train_loader, val_loader=val_loader, config=config, device=device)

    final_path = config.checkpoint_dir / "final_model.pth"
    logger.info(f"Saving final model to {final_path}")
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, final_path)

    logger.info("Training complete!")

    # get codes for all embeddings 
    model = model.to(device)
    model.eval()
    all_semantic_ids = []
    with torch.no_grad():
        for embedding in embeddings_test[:5]:
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
            semantic_ids = model.encode_to_semantic_ids(embedding_tensor)
            all_semantic_ids.append(semantic_ids.cpu().numpy())
    logger.info(f"Sample semantic IDs: {all_semantic_ids}")

    # get reconstructed embeddings from semantic ids
    reconstructed_embeddings = []
    with torch.no_grad():
        for semantic_id in all_semantic_ids:
            semantic_id_tensor = torch.tensor(semantic_id, dtype=torch.long).to(device)
            recon_embedding = model.decode_from_semantic_ids(semantic_id_tensor)
            reconstructed_embeddings.append(recon_embedding.cpu().numpy())
    logger.info(f"Original embeddings: {embeddings_test[:5]}")
    logger.info(f"Sample reconstructed embeddings: {reconstructed_embeddings}")