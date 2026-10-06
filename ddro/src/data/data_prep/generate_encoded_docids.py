import os
import json
import argparse
import collections
import random
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import gzip
import wandb
from pathlib import Path
import time

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser(description="Generate document IDs using encoding methods")
parser.add_argument("--encoding", default="pq", type=str, help="docid method: atomic/pq/rq-kmeans/rq-module/rq-ir/pq-rq/prq-module/url/summary")
parser.add_argument("--scale", default="top_300k", type=str, help="Scale of the dataset.")
parser.add_argument("--top_or_rand", default="top", type=str, help="Top or random selection.")
parser.add_argument("--sub_space", default=24, type=int, help="Sub-spaces for 768-dim vector.")
parser.add_argument("--subspace_rq", default=4, type=int, help="Number of recursive quantization steps for PQ-RQ methods.")
parser.add_argument("--cluster_num", default=256, type=int, help="Clusters per sub-space.")
parser.add_argument("--output_path", default="output/encoded_docid.txt", type=str, help="Output path")
parser.add_argument("--pretrain_model_path", default="t5-base", type=str, help="Path to pre-trained model")
parser.add_argument("--input_doc_path", type=str, required=True, help="Path to input document file")
parser.add_argument("--input_embed_path", type=str, required=True, help="Path to input doc embedding file")
parser.add_argument("--input_query_path", type=str, required=False, help="Path to input query embedding file (for rq-module with IR loss)")
parser.add_argument("--input_qrel_path", type=str, required=False, help="Path to input qrel file (for rq-module with IR loss)")
parser.add_argument("--recon_weight", type=float, default=1.0, help="Weight for reconstruction loss in RQ module-based docid generation")
parser.add_argument("--vq_weight", type=float, default=1.0, help="Weight for VQ loss in RQ module-based docid generation")
parser.add_argument("--ir_weight", default=1e-2, type=float, help="Weight for IR loss in RQ module-based docid generation")
parser.add_argument("--summary_path", default="data/summaries.json", type=str, help="Path to summaries JSON")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for encoding")

# Encoder/Decoder parameters
parser.add_argument("--use_encoder_decoder", action="store_true", help="Use trainable encoder/decoder in RQVAE")
parser.add_argument("--encoder_hidden_dims", type=int, nargs="+", default=[], help="Hidden dimensions for encoder (e.g., 512 256 128)")
parser.add_argument("--codebook_embedding_dim", type=int, default=None, help="Dimension of codebook vectors (after encoder, if used)")
parser.add_argument("--commitment_weight", type=float, default=0.25, help="Weight for commitment loss in VQ")

# IR Loss parameters
parser.add_argument("--ir_query_processing", action="store_true", help="Whether to apply recon and vq loss on query embeddings")
parser.add_argument("--ir_loss_on_latent", action="store_true", help="Whether to apply IR loss on latent embeddings instead of recon output")
parser.add_argument("--ir_loss_type", type=str, default="infonce", choices=["infonce", "triplet"], help="Type of IR loss: 'infonce' or 'triplet'")
parser.add_argument("--ir_loss_margin", type=float, default=1.0, help="Margin (epsilon) for triplet loss")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


args = parser.parse_args()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(args.seed)

def load_doc_vec(input_path):
    docid_2_idx, idx_2_docid = {}, {}
    doc_embeddings = []
    open_func = gzip.open if input_path.endswith('.gz') else open
    with open_func(input_path, 'rt', encoding='utf-8') as fr:
        for line in tqdm(fr, desc="Loading document vectors"):
            did, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]
            docid_2_idx[did] = len(docid_2_idx)
            idx_2_docid[docid_2_idx[did]] = did
            doc_embeddings.append(d_embedding)
    return docid_2_idx, idx_2_docid, np.array(doc_embeddings, dtype=np.float32)


def load_doc_and_query_vec(input_doc_path, input_query_path, input_qrel_path):
    """
    Load bath documents embeddings, query embeddings and qrels.
    Useful for the IR loss we want to implements, that needs triplets (q, d+, d-)
    d- will be sampled randomly from all documents (shuffled)

    Returns :
        docid_2_idx: dict mapping docid to index
        idx_2_docid: dict mapping index to docid
        doc_embeddings: np.array of shape (num_docs, embedding_dim)
        query_embeddings: np.array of shape (num_queries, embedding_dim) (same order as doc_embeddings)
    """
    # create doc dict 
    doc_emb_dict = {} # will be docid -> embedding
    open_func = gzip.open if input_doc_path.endswith('.gz') else open
    with open_func(input_doc_path, 'rt', encoding='utf-8') as fr:
        for line in tqdm(fr, desc="Loading document vectors"):
            did, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]
            doc_emb_dict[did] = d_embedding

    # create query dict
    query_emb_dict = {} # will be qid -> embedding
    open_func = gzip.open if input_query_path.endswith('.gz') else open
    with open_func(input_query_path, 'rt', encoding='utf-8') as fr:
        for line in tqdm(fr, desc="Loading query vectors"):
            qid, qemb = line.strip().split('\t')
            q_embedding = [float(x) for x in qemb.split(',')]
            query_emb_dict[qid] = q_embedding

    # read qrels, it is a tsv file with format 3 0 D312959 1
    docid_2_idx, idx_2_docid = {}, {}
    doc_embeddings = []
    query_embeddings = []
    open_func = gzip.open if input_qrel_path.endswith('.gz') else open
    counter = 0
    with open_func(input_qrel_path, 'rt', encoding='utf-8') as fr:
        for line in tqdm(fr, desc="Loading qrels and aligning embeddings"):
            counter +=1
            qid, _, docid, _ = line.strip().split()
            qid = f"[{qid.lower()}]"
            docid = f"[{docid.lower()}]"
            if docid not in doc_emb_dict or qid not in query_emb_dict:
                continue
            # Skip if we've already added this document
            if docid in docid_2_idx:
                continue
            docid_2_idx[docid] = len(docid_2_idx)
            idx_2_docid[docid_2_idx[docid]] = docid
            doc_embeddings.append(doc_emb_dict[docid])
            query_embeddings.append(query_emb_dict[qid])
    
    negative_doc_embeddings = doc_embeddings.copy()
    np.random.shuffle(negative_doc_embeddings)

    return (docid_2_idx, idx_2_docid, 
            np.array(query_embeddings, dtype=np.float32), 
            np.array(doc_embeddings, dtype=np.float32), 
            np.array(negative_doc_embeddings, dtype=np.float32)
    )


def atomic_docid(input_path, output_path):
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    encoded_docids = {}
    open_func = gzip.open if input_path.endswith('.gz') else open
    with open_func(input_path, 'rt', encoding='utf-8') as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='Processing atomic docids'):
            doc_item = json.loads(line)
            docid = f"[{doc_item['docid'].lower()}]"
            encoded_docids[docid] = vocab_size + doc_index
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            fw.write(f"{docid}\t{code}\n")

def product_quantization_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path):
    import nanopq
    import pickle
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    pq = nanopq.PQ(M=args.sub_space, Ks=args.cluster_num) # TODO: #28 use OPQ and test if we stil have empty clusters
    start_time = time.time()
    # pq.fit(doc_embeddings)
    pq.fit(doc_embeddings, minit='++')
    # pq.fit(doc_embeddings, iter=50, minit='++')
    print(f"PQ training completed in {(time.time() - start_time)/3600:.2f} hours")
    
    # Save the PQ model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_save_path = output_path.replace('.txt', '_model.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump({'pq': pq, 'vocab_size': vocab_size, 'cluster_num': args.cluster_num}, f)
    print(f"Saved PQ model to {model_save_path}")
    
    # Encode all embeddings at once
    print("Encoding all documents...")
    X_codes = pq.encode(doc_embeddings)
    print(f"Encoded {len(X_codes)} documents")
    
    with open(output_path, "w") as fw:
        for idx, doc_code in enumerate(X_codes):
            docid = idx_2_docid[idx]
            new_doc_code = [int(x) + n_codebook * args.cluster_num for n_codebook, x in enumerate(doc_code)]
            code = ','.join(str(x + vocab_size) for x in new_doc_code)
            fw.write(f"{docid}\t{code}\n")


def residual_quantization_kmeans_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path):
    """
    Docstring for residual_quantization_docid
    Use a recursive kmeans clustering.
    """
    from residual_quantizer import ResidualQuantizerKmeans
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size

    rq = ResidualQuantizerKmeans(
        embedding_dim = doc_embeddings.shape[1], 
        nb_codebooks = args.subspace_rq, 
        codebook_size = args.cluster_num
    )
    start_time = time.time()
    rq.train(doc_embeddings)
    print(f"RQ training completed in {(time.time() - start_time)/3600:.2f} hours")
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # rq.save(output_path)
    
    # Save the RQ-kmeans model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_save_path = output_path.replace('.txt', '_model.pkl')
    rq.save(model_save_path, vocab_size=vocab_size, cluster_num=args.cluster_num)
    print(f"Saved RQ-kmeans model to {model_save_path}")

    # Test reconstruction quality
    test_codes = rq.encode(doc_embeddings[:1000])
    test_decoded = rq.decode(test_codes)
    reconstruction_error = np.mean((doc_embeddings[:1000] - test_decoded) ** 2)
    print(f"Reconstruction MSE on 1000 samples: {reconstruction_error:.6f}")

    # Save encoded docids
    with open(output_path, "w") as fw:
        for i in range(0, len(doc_embeddings), args.batch_size):
            batch_embeddings = doc_embeddings[i:i + args.batch_size]
            X_code = rq.encode(batch_embeddings)
            for idx, doc_code in enumerate(X_code, start=i):
                docid = idx_2_docid[idx]
                new_doc_code = [int(x) + n_codebook * args.cluster_num for n_codebook, x in enumerate(doc_code)]
                code = ','.join(str(x + vocab_size) for x in new_doc_code)
                fw.write(f"{docid}\t{code}\n")

def residual_quantization_module_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path):
    """
    Residual quantization docid with a module-based implementation.
    """
    from residual_quantizer_module import RQVAEConfig, RQVAE, EmbeddingDataset
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = RQVAEConfig(
        checkpoint_dir= Path(output_path).parent / "checkpoints" / Path(output_path).stem,
        embedding_dim = doc_embeddings.shape[1], 
        codebook_quantization_levels = args.subspace_rq, 
        codebook_size = args.cluster_num,
        use_rotation_trick=True,
        use_kmeans_init=True,
        use_encoder_decoder=args.use_encoder_decoder,
        encoder_hidden_dims=args.encoder_hidden_dims,
        codebook_embedding_dim=args.codebook_embedding_dim if args.codebook_embedding_dim is not None else doc_embeddings.shape[1],
        commitment_weight=args.commitment_weight,
    )

    run_name = f"rqvae-L{config.codebook_quantization_levels}-C{config.codebook_size}-D{config.embedding_dim}"
    run = wandb.init(project="ddro-training-alexia", name=run_name, config=config.__dict__)
    config.log_config()

    dataset = EmbeddingDataset(embeddings=doc_embeddings, limit=None)
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train size: {len(train_dataset):,}, Val size: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Disabled to prevent segmentation faults
        pin_memory=False,  # Disabled to prevent memory issues
        drop_last=False,  # Include partial batches to avoid losing data
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Disabled to prevent segmentation faults
        pin_memory=False,  # Disabled to prevent memory issues
    )

    model = RQVAE(config)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    model.train_rqvae(data_loader=train_loader, val_loader=val_loader, config=config, device=device)
    print(f"RQ-VAE training completed in {(time.time() - start_time)/3600:.2f} hours")
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_path = config.checkpoint_dir / "final_model.pth"
    print(f"Saving final model to {final_path}")
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, final_path)

    print("Training complete!")
    # get best model
    best_model_path = config.checkpoint_dir / "best_model.pth"
    print(f"Loading best model from {best_model_path}")
    best_model_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # Reconstruct config from checkpoint
    saved_config_dict = best_model_checkpoint["config"]
    saved_config = RQVAEConfig(**saved_config_dict)
    
    # Create new model with saved config
    model = RQVAE(saved_config)
    model.load_state_dict(best_model_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Test reconstruction quality
    with torch.no_grad():
        sample_embeddings = torch.tensor(doc_embeddings[:1000], dtype=torch.float32).to(device)
        reconstructed, indices, loss_dict = model(sample_embeddings)
        print(f"Sample original embeddings (first 2):\n{sample_embeddings[:5].cpu().numpy()}")
        print(f"Sample reconstructed embeddings (first 2):\n{reconstructed[:5].cpu().numpy()}")
        reconstruction_error = torch.mean((sample_embeddings - reconstructed) ** 2).item()
        print(f"Reconstruction MSE on 1000 samples: {reconstruction_error:.6f}")

    # Save encoded docids
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fw:
        for i in range(0, len(doc_embeddings), args.batch_size):
            batch_embeddings = doc_embeddings[i:i + args.batch_size]
            embedding_tensor = torch.tensor(batch_embeddings, dtype=torch.float32).to(device)
            with torch.no_grad():
                reconstructed, indices, loss_dict = model(embedding_tensor)
            # indices is a list of tensors, stack them to get [actual_batch_size, num_codebooks]
            X_code = torch.stack(indices, dim=-1).cpu().numpy()
            
            for idx, doc_code in enumerate(X_code, start=i):
                docid = idx_2_docid[idx]
                new_doc_code = [int(x) + n_codebook * args.cluster_num for n_codebook, x in enumerate(doc_code)]
                code = ','.join(str(x + vocab_size) for x in new_doc_code)
                fw.write(f"{docid}\t{code}\n")
    print(f"Encoded docids saved to {output_path}")


def residual_quantization_ir_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, query_embeddings, negative_doc_embeddings, output_path):
    """
    Residual quantization docid with a module-based implementation.
    Uses a custom IR loss based on query embeddings and qrels.
    Important: doc_embeddings and query_embeddings must be aligned based on qrels!
    """
    from residual_quantizer_module import RQVAEConfig, RQVAE, EmbeddingDataset
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = RQVAEConfig(
        checkpoint_dir= Path(output_path).parent / "checkpoints" / Path(output_path).stem,
        embedding_dim = doc_embeddings.shape[1], 
        codebook_quantization_levels = args.subspace_rq, 
        codebook_size = args.cluster_num,
        use_rotation_trick=True,
        use_kmeans_init=True,
        recon_weight = args.recon_weight,
        vq_weight = args.vq_weight,
        ir_weight = args.ir_weight,
        use_encoder_decoder=args.use_encoder_decoder,
        encoder_hidden_dims=args.encoder_hidden_dims,
        codebook_embedding_dim=args.codebook_embedding_dim if args.codebook_embedding_dim is not None else doc_embeddings.shape[1],
        commitment_weight=args.commitment_weight,
        ir_query_processing=args.ir_query_processing,
        ir_loss_on_latent=args.ir_loss_on_latent,
        ir_loss_type=args.ir_loss_type,
        ir_loss_margin=args.ir_loss_margin
    )

    run_name = f"rqvae-L{config.codebook_quantization_levels}-C{config.codebook_size}-D{config.embedding_dim}"
    run = wandb.init(project="ddro-training-alexia", name=run_name, config=config.__dict__)
    config.log_config()

    dataset = EmbeddingDataset(embeddings=doc_embeddings, queries=query_embeddings, negatives=negative_doc_embeddings, limit=None)
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train size: {len(train_dataset):,}, Val size: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Disabled to prevent segmentation faults
        pin_memory=False,  # Disabled to prevent memory issues
        drop_last=False,  # Include partial batches to avoid losing data
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Disabled to prevent segmentation faults
        pin_memory=False,  # Disabled to prevent memory issues
    )

    model = RQVAE(config)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    model.train_rqvae(data_loader=train_loader, val_loader=val_loader, config=config, device=device)
    print(f"RQ-VAE training completed in {(time.time() - start_time)/3600:.2f} hours")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_path = config.checkpoint_dir / "final_model.pth"
    print(f"Saving final model to {final_path}")
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, final_path)

    print("Training complete!")
    # get best model
    best_model_path = config.checkpoint_dir / "best_model.pth"
    print(f"Loading best model from {best_model_path}")
    best_model_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # Reconstruct config from checkpoint
    saved_config_dict = best_model_checkpoint["config"]
    saved_config = RQVAEConfig(**saved_config_dict)
    
    # Create new model with saved config
    model = RQVAE(saved_config)
    model.load_state_dict(best_model_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Test reconstruction quality
    with torch.no_grad():
        sample_embeddings = torch.tensor(doc_embeddings[:1000], dtype=torch.float32).to(device)
        reconstructed, indices, loss_dict = model(sample_embeddings)
        print(f"Sample original embeddings (first 2):\n{sample_embeddings[:5].cpu().numpy()}")
        print(f"Sample reconstructed embeddings (first 2):\n{reconstructed[:5].cpu().numpy()}")
        reconstruction_error = torch.mean((sample_embeddings - reconstructed) ** 2).item()
        print(f"Reconstruction MSE on 1000 samples: {reconstruction_error:.6f}")

    # Save encoded docids
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fw:
        for i in range(0, len(doc_embeddings), args.batch_size):
            batch_embeddings = doc_embeddings[i:i + args.batch_size]
            embedding_tensor = torch.tensor(batch_embeddings, dtype=torch.float32).to(device)
            with torch.no_grad():
                reconstructed, indices, loss_dict = model(embedding_tensor)
            # indices is a list of tensors, stack them to get [actual_batch_size, num_codebooks]
            X_code = torch.stack(indices, dim=-1).cpu().numpy()
            
            for idx, doc_code in enumerate(X_code, start=i):
                docid = idx_2_docid[idx]
                new_doc_code = [int(x) + n_codebook * args.cluster_num for n_codebook, x in enumerate(doc_code)]
                code = ','.join(str(x + vocab_size) for x in new_doc_code)
                fw.write(f"{docid}\t{code}\n")
    print(f"Encoded docids saved to {output_path}")


def pq_rq_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path):
    """
    Docstring for residual_quantization_docid
    Use a recursive kmeans clustering.
    """
    from residual_quantizer import PQRQ
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size

    pq_rq = PQRQ(
        embedding_dim = doc_embeddings.shape[1], 
        nb_subspaces = args.sub_space, 
        codebook_size = args.cluster_num,
        rq_steps = args.subspace_rq
    )
    start_time = time.time()
    pq_rq.train(doc_embeddings)
    print(f"PQ-RQ training completed in {(time.time() - start_time)/3600:.2f} hours")

    # Save the PQ-RQ model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_save_path = output_path.replace('.txt', '_model.pkl')
    pq_rq.save(model_save_path, vocab_size=vocab_size)
    print(f"Saved PQ-RQ model to {model_save_path}")

    # Test reconstruction quality
    test_codes = pq_rq.encode(doc_embeddings[:1000])
    test_decoded = pq_rq.decode(test_codes)
    reconstruction_error = np.mean((doc_embeddings[:1000] - test_decoded) ** 2)
    print(f"Reconstruction MSE on 1000 samples: {reconstruction_error:.6f}")

    # Save encoded docids
    with open(output_path, "w") as fw:
        for i in range(0, len(doc_embeddings), args.batch_size):
            batch_embeddings = doc_embeddings[i:i + args.batch_size]
            X_code = pq_rq.encode(batch_embeddings)
            for idx, doc_code in enumerate(X_code, start=i):
                docid = idx_2_docid[idx]
                new_doc_code = [int(x) + n_codebook * args.cluster_num for n_codebook, x in enumerate(doc_code)]
                code = ','.join(str(x + vocab_size) for x in new_doc_code)
                fw.write(f"{docid}\t{code}\n")


def prq_module_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path):
    """
    PRQVAE (Product Quantization + Residual Quantized VAE) docid generation.
    Splits embeddings into nb_subspaces subspaces and trains an independent
    RQVAE on each subspace. No encoder/decoder, no IR loss.
    """
    from residual_quantizer_module import PRQVAEConfig, PRQVAE, EmbeddingDataset
    t5_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = t5_model.config.vocab_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = PRQVAEConfig(
        checkpoint_dir=Path(output_path).parent / "checkpoints" / Path(output_path).stem,
        embedding_dim=doc_embeddings.shape[1],
        nb_subspaces=args.sub_space,
        codebook_quantization_levels=args.subspace_rq,
        codebook_size=args.cluster_num,
        use_rotation_trick=True,
        use_kmeans_init=True,
        recon_weight=args.recon_weight,
        vq_weight=args.vq_weight,
    )

    run_name = f"prqvae-C{config.nb_subspaces}-L{config.codebook_quantization_levels}-V{config.codebook_size}-D{config.embedding_dim}"
    wandb.init(project="ddro-training-alexia", name=run_name, config=config.__dict__)
    config.log_config()

    dataset = EmbeddingDataset(embeddings=doc_embeddings, limit=None)
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train size: {len(train_dataset):,}, Val size: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    prqvae_model = PRQVAE(config)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    prqvae_model.train_rqvae(data_loader=train_loader, val_loader=val_loader, config=config, device=device)
    print(f"PRQVAE training completed in {(time.time() - start_time)/3600:.2f} hours")

    final_path = config.checkpoint_dir / "final_model.pth"
    print(f"Saving final model to {final_path}")
    torch.save({"model_state_dict": prqvae_model.state_dict(), "config": config.__dict__}, final_path)

    # Load best model
    best_model_path = config.checkpoint_dir / "best_model.pth"
    print(f"Loading best model from {best_model_path}")
    best_model_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    saved_config = PRQVAEConfig(**best_model_checkpoint["config"])
    prqvae_model = PRQVAE(saved_config)
    prqvae_model.load_state_dict(best_model_checkpoint["model_state_dict"])
    prqvae_model.to(device)
    prqvae_model.eval()

    # Test reconstruction quality
    with torch.no_grad():
        sample_embeddings = torch.tensor(doc_embeddings[:1000], dtype=torch.float32).to(device)
        reconstructed, _, _ = prqvae_model(sample_embeddings)
        reconstruction_error = torch.mean((sample_embeddings - reconstructed) ** 2).item()
        print(f"Reconstruction MSE on 1000 samples: {reconstruction_error:.6f}")

    # Save encoded docids
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fw:
        for i in range(0, len(doc_embeddings), args.batch_size):
            batch_embeddings = doc_embeddings[i:i + args.batch_size]
            embedding_tensor = torch.tensor(batch_embeddings, dtype=torch.float32).to(device)
            with torch.no_grad():
                semantic_ids = prqvae_model.encode_to_semantic_ids(embedding_tensor)
            X_code = semantic_ids.cpu().numpy()
            for idx, doc_code in enumerate(X_code, start=i):
                docid = idx_2_docid[idx]
                new_doc_code = [int(x) + n_codebook * args.cluster_num for n_codebook, x in enumerate(doc_code)]
                code = ','.join(str(x + vocab_size) for x in new_doc_code)
                fw.write(f"{docid}\t{code}\n")
    print(f"Encoded docids saved to {output_path}")


# def is_url_semantically_rich(url_segments):
#     generic_terms = {"index", "page", "item", "view", "default", "home"}
#     descriptive_count = sum(
#         1 for segment in url_segments
#         if not segment.isnumeric() and segment not in generic_terms and len(segment) > 2
#     )
#     return descriptive_count > len(url_segments) / 2


def is_url_semantically_rich(url_segments):
    """
    Determines if a URL is semantically rich based on its segments.
    A URL is considered semantically rich if the majority of its segments
    are descriptive and not generic or numeric.
    """
    generic_terms = {"index", "page", "item", "view", "default", "home"}
    descriptive_count = 0
    total_segments = len(url_segments)

    for segment in url_segments:
        if not segment.isnumeric() and segment not in generic_terms and len(segment) > 2:
            descriptive_count += 1

    return descriptive_count > total_segments / 2

def url_docid(input_data, output_path, max_docid_len=99, pretrain_model_path="t5-base"):
    tokenizer = T5Tokenizer.from_pretrained(pretrain_model_path)
    results = {}
    skipped_docs = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for doc_item in tqdm(input_data, desc="Processing URL docids"):
        try:
            docid = doc_item.get('docid', '') or doc_item.get('id', '')
            docid = str(docid).strip().lower()
            url = doc_item.get('url', '')
            title = doc_item.get('title', '')
            if not docid:
                skipped_docs.append({"reason": "Missing or empty docid", "doc_item": doc_item})
                continue
            url = url.strip().lower() if isinstance(url, str) else ""
            title = title.strip().lower() if isinstance(title, str) else ""
            if not url and not title:
                skipped_docs.append({"reason": "Missing both URL and title", "docid": docid})
                continue
            url = url.replace("http://", "").replace("https://", "").replace("-", " ")
            url_segments = [segment for segment in url.split('/') if segment]
            domain = url_segments[0] if url_segments else ""
            reversed_url = " ".join(reversed(url_segments[1:])) if len(url_segments) > 1 else ""
            if url_segments and is_url_semantically_rich(url_segments):
                final_string = f"{reversed_url} {domain}".strip()
            elif title and domain:
                final_string = f"{title} {domain}".strip()
            elif title:
                final_string = title
            elif domain:
                final_string = domain
            else:
                skipped_docs.append({"reason": "Unable to determine final string", "docid": docid, "url": url, "title": title})
                continue
            tokenized_ids = tokenizer(final_string, truncation=True, max_length=max_docid_len).input_ids
            tokenized_ids = tokenized_ids[:-1][:max_docid_len] + [1]
            results[docid] = {"final_string": final_string, "token_ids": tokenized_ids}
        except Exception as e:
            skipped_docs.append({"reason": f"Error: {str(e)}", "docid": doc_item.get('docid', 'unknown')})
    with open(output_path, "w") as fw:
        for docid, val in results.items():
            fw.write(f"[{docid}]\t{','.join(map(str, val['token_ids']))}\n")
    if skipped_docs:
        print(f"Skipped documents count: {len(skipped_docs)}")
        for skipped in skipped_docs:
            print(skipped)

def summary_based_docid(summary_path, output_path, max_docid_len=128):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    with open(summary_path, "r") as f:
        summaries = json.load(f)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fw:
        for summary_item in tqdm(summaries, desc="Encoding summaries"):
            if 'id' not in summary_item or 'summary' not in summary_item:
                continue
            docid = f"[{summary_item['id'].lower()}]"
            summary = summary_item['summary']
            tokenized_summary = tokenizer(summary, truncation=True, max_length=max_docid_len).input_ids
            doc_code = ','.join(map(str, tokenized_summary))
            fw.write(f"{docid}\t{doc_code}\n")

if __name__ == "__main__":
    if args.encoding == "atomic":
        atomic_docid(args.input_doc_path, args.output_path)

    elif args.encoding == "pq":
        docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec(args.input_embed_path)
        product_quantization_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, args.output_path)
    
    elif args.encoding == "rq-kmeans":
        docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec(args.input_embed_path)
        residual_quantization_kmeans_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, args.output_path)

    elif args.encoding == "rq-module":
        docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec(args.input_embed_path)
        residual_quantization_module_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, args.output_path)
    
    elif args.encoding == "rq-ir":
        docid_2_idx, idx_2_docid, query_embeddings, doc_embeddings, negative_doc_embeddings = load_doc_and_query_vec(
            args.input_embed_path, args.input_query_path, args.input_qrel_path
        )
        residual_quantization_ir_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, query_embeddings, negative_doc_embeddings, args.output_path)

    elif args.encoding == "pq-rq":
        docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec(args.input_embed_path)
        pq_rq_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, args.output_path)

    elif args.encoding == "prq-module":
        docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec(args.input_embed_path)
        prq_module_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, args.output_path)

    elif args.encoding == "url":
        open_func = gzip.open if args.input_doc_path.endswith(".gz") else open
        input_data = []
        with open_func(args.input_doc_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    input_data.append(json.loads(line.strip()))
        url_docid(input_data, args.output_path, pretrain_model_path=args.pretrain_model_path)

    elif args.encoding == "summary":
        summary_based_docid(args.summary_path, args.output_path)
    else:
        raise ValueError("Invalid encoding method. Choose from atomic/pq/url/summary/rq-kmeans/rq-module/rq-ir/pq-rq/prq-module.")