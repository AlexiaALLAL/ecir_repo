import os
import sys
import torch
import random
import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time

# Fix for "Too many open files" error with PyTorch DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
# Add src to Python path for imports
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.utils import load_model
from utils.trie import Trie
from utils.evaluate import evaluator
from pretrain.T5ForPretrain import T5ForPretrain
from utils.pretrain_dataset import PretrainDataForT5


# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
### training settings
parser.add_argument("--epochs", default=2, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_batch_size", default=25, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="The ratio of warmup steps.")
parser.add_argument("--output_every_n_step", default=25, type=int, help="The steps to output training information.")
parser.add_argument("--save_every_n_epoch", default=25, type=int, help="The epochs to save the trained models.")
parser.add_argument("--operation", default="training", type=str, help="which operation to take, training/testing")
parser.add_argument("--use_docid_rank", default="False", type=str, help="whether to use docid for ranking, or only doc code.")
parser.add_argument("--load_ckpt", default="False", type=str, help="whether to load a trained model checkpoint.")
parser.add_argument("--debug", default="False", type=str, help="Enable detailed debugging output")

### path to load data and save models
parser.add_argument("--save_path", default="./model/", type=str, help="The path to save trained models.")
parser.add_argument("--log_path", default="./log/", type=str, help="The path to save log.")
parser.add_argument("--doc_file_path", default="/ivi/ilps/personal/kmekonn/projects/DPO-Enhanced-DSI/data/processed/msmarco-docs-sents.top.300k.json", type=str, help='path of origin sent data.')
parser.add_argument("--docid_path", default="None", type=str, help='path of the encoded docid.')
parser.add_argument("--train_file_path", type=str, help="the path/directory of the training file.")
parser.add_argument("--test_file_path", type=str, help="the path/directory of the testing file.")
parser.add_argument("--pretrain_model_path", type=str, help="path of the pretrained model checkpoint")
parser.add_argument("--load_ckpt_path", default="./model/", type=str, help="The path to load ckpt of a trained model.")
parser.add_argument("--dataset_script_dir", type=str, help="The path of dataset script.")
parser.add_argument("--dataset_cache_dir", type=str, help="The path of dataset cache.")

### hyper-parameters to control the model
parser.add_argument("--add_doc_num", type=int, help="the number of docid to be added.")
parser.add_argument("--max_seq_length", type=int, default=512, help="the max length of input sequences.")
parser.add_argument("--max_docid_length", type=int, default=1, help="the max length of docid sequences.")
parser.add_argument("--use_origin_head", default="False", type=str, help="whether to load the lm_head from the pretrained model.")
parser.add_argument("--num_beams", default=10, type=int, help="the number of beams.")
parser.add_argument("--results_csv_file", default=None, type=str, help="CSV file to append results to (optional)")
parser.add_argument("--encoding_method", default=None, type=str, help="Encoding method (pq, rq-kmeans, prq-module, etc.)")
parser.add_argument("--num_codebooks", default=None, type=int, help="Number of codebooks/subspaces")
parser.add_argument("--nb_subspaces", default=None, type=int, help="Number of subspaces (for prq-module)")
parser.add_argument("--codebook_size", default=None, type=int, help="Codebook size")

args = parser.parse_args()

DEBUG = args.debug == "True"

print("="*80)
print("CONFIGURATION SUMMARY")
print("="*80)
print(f"Operation: {args.operation}")
print(f"Debug mode: {DEBUG}")
print(f"Device: {device}")
print(f"Batch size: {args.per_gpu_batch_size * torch.cuda.device_count()}")
print(f"Num beams: {args.num_beams}")
print(f"Max docid length: {args.max_docid_length}")
print(f"Test file: {args.test_file_path}")
print(f"Docid path: {args.docid_path}")
print(f"Checkpoint: {args.save_path}")
if args.encoding_method:
    print(f"Encoding method: {args.encoding_method}")
    if args.num_codebooks:
        print(f"  - num_codebooks: {args.num_codebooks}")
    if args.nb_subspaces:
        print(f"  - nb_subspaces: {args.nb_subspaces}")
    if args.codebook_size:
        print(f"  - codebook_size: {args.codebook_size}")
if args.results_csv_file:
    print(f"Results CSV file: {args.results_csv_file}")
print("="*80)

if torch.cuda.is_available():
    args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
else:
    args.batch_size = args.per_gpu_batch_size

logger = open(args.log_path, "a")
logger.write("\n")
logger.write(f"start a new running with args: {args}\n")
tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

def load_data(file_path):
    """
        function: load data from the file_path
        args: file_path  -- a directory or a specific file
    """
    if os.path.isfile(file_path):
        fns = [file_path]
    else:
        data_dir = file_path
        fns = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]
    print("file path: ", fns)
    return fns


def load_encoded_docid(docid_path, dataset="msmarco"):
    """
        function: load encoded docid data from the docid_path
        return:
            encoded_docids: list of all encoded document identifiers.
            encode_2_docid: dict from encoded document identifiers to original unique id.
    """
    print("\n" + "="*80)
    print("LOADING ENCODED DOCIDS")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Docid path: {docid_path}")
    
    encode_2_docid = {}
    encoded_docids = []
    
    with open(docid_path, "r") as fr:
        for idx, line in enumerate(fr):
            docid, encode = line.strip().split("\t")
            docid = docid.lower()
            
            original_encode = encode
            
            # For NQ dataset, remove padding tokens (0 and 1)
            if dataset == "nq":
                encode_list = encode.split(",")
                encode = [int(x) for x in encode_list if x not in ["0", "1"]]
            else:
                # For MS MARCO, use as-is
                encode = [int(x) for x in encode.split(",")]
            
            # Debug: Print first 3 docids only
            if DEBUG and idx < 3:
                print(f"  Sample {idx}: docid={docid}")
                if dataset == "nq":
                    print(f"    Original: {original_encode}")
                    print(f"    After padding removal: {encode}")
                else:
                    print(f"    Encoded: {encode}")
            
            encoded_docids.append(encode)
            encode_str = ','.join([str(x) for x in encode])
            
            if encode_str not in encode_2_docid:
                encode_2_docid[encode_str] = [docid]
            else:
                encode_2_docid[encode_str].append(docid)
    
    print(f"Total encoded docids loaded: {len(encoded_docids)}")
    print(f"Total unique encodings: {len(encode_2_docid)}")
    
    if DEBUG:
        print(f"Sample encoding keys (first 3):")
        for i, key in enumerate(list(encode_2_docid.keys())[:3]):
            print(f"  {i+1}. {key} -> {encode_2_docid[key]}")
    print("="*80 + "\n")
    
    return encoded_docids, encode_2_docid


def order_docids(docid_list, target_docid, mode="random"):
            if mode == "random":
                random.shuffle(docid_list)
                return docid_list
            if target_docid in docid_list:
                others = [d for d in docid_list if d != target_docid]
                if mode == "best":
                    return [target_docid] + others
                if mode == "worst":
                    return others + [target_docid]
            return docid_list


def evaluate_beamsearch():
    '''
        function: Generate the document identifiers with constrained beam search, and evaluate the ranking results.
    '''
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)
    
    model = T5ForPretrain(pretrain_model, args)
    save_model = load_model(args.save_path)
    model.load_state_dict(save_model)
    model = model.to(device)
    model.eval()
    
    myevaluator = evaluator()
    
    # Detect dataset from docid_path
    dataset = "nq" if "nq" in args.docid_path.lower() else "msmarco"
    print(f"Detected dataset: {dataset}\n")
    
    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path, dataset=dataset)
    docid_trie = Trie([[0] + item for item in encoded_docid])
    print(f"Trie built with {len(encoded_docid)} docids\n")

    def prefix_allowed_tokens_fn(batch_id, sent): 
        outputs = docid_trie.get(sent.tolist())
        if len(outputs) == 0:
            return [tokenizer.pad_token_id]
        return outputs
    
    def docid2string(docid):
        x_list = []
        for x in docid:
            if x != 0:
                x_list.append(str(x))
            if x == 1:
                break
        return ",".join(x_list)

    if os.path.exists(args.test_file_path):
        localtime = time.asctime(time.localtime(time.time()))
        print(f"Evaluate on the {args.test_file_path}.")
        
        logger.write(f"{localtime} Evaluate on the {args.test_file_path}.\n")
        test_data = load_data(args.test_file_path)
        test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        truth, prediction, prediction_best, prediction_worst, inputs = [], [], [], [], []
        
        # Counters for debugging
        total_queries = 0
        successful_matches = 0
        failed_matches = 0
        not_found_docids = set()  # Track unique not-found docids
        
        print("Start evaluating... \n")
        

        for i, testing_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"):
            with torch.no_grad():
                for key in testing_data.keys():
                    if key in ["query_id", "doc_id"]:
                        continue
                    testing_data[key] = testing_data[key].to(device)
            
            input_ids = testing_data["input_ids"]
            
            if args.use_docid_rank == "False":
                labels = testing_data["docid_labels"]
                truth.extend([[docid2string(docid)] for docid in labels.cpu().numpy().tolist()])
            else:
                labels = testing_data["query_id"]
                truth.extend([[docid] for docid in labels])
            
            inputs.extend(input_ids)

            outputs = model.generate(input_ids, max_length=args.max_docid_length+1, num_return_sequences=args.num_beams, num_beams=args.num_beams, do_sample=False, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
            
            for j in range(input_ids.shape[0]):
                total_queries += 1
                doc_rank = []
                doc_rank_best = []
                doc_rank_worst = []
                batch_output = outputs[j*args.num_beams:(j+1)*args.num_beams].cpu().numpy().tolist()
                
                # ONLY print detailed debug for first 2 queries
                debug_this_query = DEBUG and total_queries <= 2
                
                if debug_this_query:
                    query_term = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                    print("\n" + "-"*80)
                    print(f"QUERY {total_queries} DEBUG")
                    print("-"*80)
                    print(f"Query: {query_term}")
                    print(f"Ground truth: {labels[j] if args.use_docid_rank == 'True' else docid2string(testing_data['docid_labels'][j].cpu().numpy().tolist())}")
                    print(f"\nGenerated {len(batch_output)} candidates:")
                
                batch_has_match = False
                for beam_idx, docid in enumerate(batch_output):
                    docid_string = docid2string(docid)
                    
                    if args.use_docid_rank == "False":
                        doc_rank.append(docid_string)
                        doc_rank_best.append(docid_string)
                        doc_rank_worst.append(docid_string)
                        if debug_this_query and beam_idx < 5:  # Only print first 5 beams
                            print(f"  Beam {beam_idx+1}: {docid_string}")
                    else:
                        # Check if this docid exists in our dictionary
                        if docid_string in encode_2_docid:
                            docid_list = encode_2_docid[docid_string]
                            if len(docid_list) > 1:
                                target_docid = labels[j]
                                doc_rank.extend(order_docids(docid_list.copy(), target_docid, mode="random"))
                                doc_rank_best.extend(order_docids(docid_list.copy(), target_docid, mode="best"))
                                doc_rank_worst.extend(order_docids(docid_list.copy(), target_docid, mode="worst"))
                            else:
                                doc_rank.extend(docid_list)
                                doc_rank_best.extend(docid_list)
                                doc_rank_worst.extend(docid_list)
                            
                            batch_has_match = True
                            if debug_this_query and beam_idx < 5:
                                print(f"  Beam {beam_idx+1}: {docid_string} -> FOUND: {docid_list}")
                        else:
                            if debug_this_query and beam_idx < 5:
                                print(f"  Beam {beam_idx+1}: {docid_string} -> NOT FOUND")
                            failed_matches += 1
                            not_found_docids.add(docid_string)
                
                if batch_has_match:
                    successful_matches += 1
                    
                if debug_this_query:
                    print(f"\nFinal ranking (top 5): {doc_rank[:5]}")
                    print("-"*80)
                
                prediction.append(doc_rank)
                prediction_best.append(doc_rank_best)
                prediction_worst.append(doc_rank_worst)

        print("\n" + "="*80)
        print("EVALUATION STATISTICS")
        print("="*80)
        print(f"Total queries processed: {total_queries}")
        print(f"Queries with at least one match: {successful_matches}")
        print(f"Match rate: {successful_matches/total_queries*100:.2f}%")
        print(f"Total failed docid lookups: {failed_matches}")
        print(f"Unique not-found docids: {len(not_found_docids)}")
        
        if not_found_docids and len(not_found_docids) <= 10:
            print(f"\nSample not-found docids:")
            for docid in list(not_found_docids)[:10]:
                print(f"  {docid}")
        print("="*80 + "\n")

        print("Computing metrics...")
        result_df = myevaluator.evaluate_ranking(truth, prediction)
        result_df_best = myevaluator.evaluate_ranking(truth, prediction_best)
        result_df_worst = myevaluator.evaluate_ranking(truth, prediction_worst)
        # Extracting metrics
        _mrr10 = result_df['MRR@10'].values.mean()
        _mrr = result_df['MRR'].values.mean()
        _ndcg10 = result_df['NDCG@10'].values.mean()
        _ndcg20 = result_df['NDCG@20'].values.mean()
        _ndcg100 = result_df['NDCG@100'].values.mean()
        _map20 = result_df['MAP@20'].values.mean()
        _p1 = result_df['P@1'].values.mean()
        _p10 = result_df['P@10'].values.mean()
        _p20 = result_df['P@20'].values.mean()
        _p100 = result_df['P@100'].values.mean()
        _r1 = result_df['R@1'].values.mean()
        _r10 = result_df['R@10'].values.mean()
        _r100 = result_df['R@100'].values.mean()
        _r1000 = result_df['R@1000'].values.mean()
        _hit1 = result_df['Hit@1'].values.mean()
        _hit5 = result_df['Hit@5'].values.mean()
        _hit10 = result_df['Hit@10'].values.mean()
        _hit100 = result_df['Hit@100'].values.mean()

        _mrr10_best = result_df_best['MRR@10'].values.mean()
        _mrr_best = result_df_best['MRR'].values.mean()
        _ndcg10_best = result_df_best['NDCG@10'].values.mean()
        _ndcg20_best = result_df_best['NDCG@20'].values.mean()
        _ndcg100_best = result_df_best['NDCG@100'].values.mean()
        _map20_best = result_df_best['MAP@20'].values.mean()
        _p1_best = result_df_best['P@1'].values.mean()
        _p10_best = result_df_best['P@10'].values.mean()
        _p20_best = result_df_best['P@20'].values.mean()
        _p100_best = result_df_best['P@100'].values.mean()
        _r1_best = result_df_best['R@1'].values.mean()
        _r10_best = result_df_best['R@10'].values.mean()
        _r100_best = result_df_best['R@100'].values.mean()
        _r1000_best = result_df_best['R@1000'].values.mean()
        _hit1_best = result_df_best['Hit@1'].values.mean()
        _hit5_best = result_df_best['Hit@5'].values.mean()
        _hit10_best = result_df_best['Hit@10'].values.mean()
        _hit100_best = result_df_best['Hit@100'].values.mean()

        _mrr10_worst = result_df_worst['MRR@10'].values.mean()
        _mrr_worst = result_df_worst['MRR'].values.mean()
        _ndcg10_worst = result_df_worst['NDCG@10'].values.mean()
        _ndcg20_worst = result_df_worst['NDCG@20'].values.mean()
        _ndcg100_worst = result_df_worst['NDCG@100'].values.mean()
        _map20_worst = result_df_worst['MAP@20'].values.mean()
        _p1_worst = result_df_worst['P@1'].values.mean()
        _p10_worst = result_df_worst['P@10'].values.mean()
        _p20_worst = result_df_worst['P@20'].values.mean()
        _p100_worst = result_df_worst['P@100'].values.mean()
        _r1_worst = result_df_worst['R@1'].values.mean()
        _r10_worst = result_df_worst['R@10'].values.mean()
        _r100_worst = result_df_worst['R@100'].values.mean()
        _r1000_worst = result_df_worst['R@1000'].values.mean()
        _hit1_worst = result_df_worst['Hit@1'].values.mean()
        _hit5_worst = result_df_worst['Hit@5'].values.mean()
        _hit10_worst = result_df_worst['Hit@10'].values.mean()
        _hit100_worst = result_df_worst['Hit@100'].values.mean()

        localtime = time.asctime(time.localtime(time.time()))
        
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"MRR@10: {_mrr10:.4f} | MRR: {_mrr:.4f}")
        print(f"P@1: {_p1:.4f} | P@10: {_p10:.4f} | P@20: {_p20:.4f}")
        print(f"R@1: {_r1:.4f} | R@10: {_r10:.4f} | R@100: {_r100:.4f} | R@1000: {_r1000:.4f}")
        print(f"Hit@1: {_hit1:.4f} | Hit@5: {_hit5:.4f} | Hit@10: {_hit10:.4f} | Hit@100: {_hit100:.4f}")
        print("="*80 + "\n")

        # Save TREC format (.run file)
        trec_output_path = args.save_path.replace(".pt", f"_beam{args.num_beams}.run")
        os.makedirs(os.path.dirname(trec_output_path), exist_ok=True)
        with open(trec_output_path, "w") as fw:
            for qid, docids in enumerate(prediction):
                for rank, docid in enumerate(docids, 1):
                    # Extract numeric ID from docid string (e.g., "[123456]" -> "123456")
                    docid_str = str(docid).strip("[]")
                    fw.write(f"{qid} Q0 {docid_str} {rank} {1.0/(rank)} beam{args.num_beams}\n")
        print(f"Saved TREC format results to: {trec_output_path}\n")

        # Save results to CSV if specified
        if args.results_csv_file:
            import csv
            from datetime import datetime
            
            # Use passed encoding arguments
            encoding_type = args.encoding_method
            num_codebooks = args.num_codebooks
            nb_subspaces = args.nb_subspaces
            codebook_size = args.codebook_size
            
            # Prepare results row
            results_row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "encoding": encoding_type,
                "num_codebooks": num_codebooks,
                "nb_subspaces": nb_subspaces,
                "codebook_size": codebook_size,
                "beam_size": args.num_beams,
                "MRR@10": _mrr10,
                "MRR": _mrr,
                "NDCG@10": _ndcg10,
                "NDCG@20": _ndcg20,
                "NDCG@100": _ndcg100,
                "MAP@20": _map20,
                "P@1": _p1,
                "P@10": _p10,
                "P@20": _p20,
                "P@100": _p100,
                "R@1": _r1,
                "R@10": _r10,
                "R@100": _r100,
                "R@1000": _r1000,
                "Hit@1": _hit1,
                "Hit@5": _hit5,
                "Hit@10": _hit10,
                "Hit@100": _hit100,
                "MRR@10_best": _mrr10_best,
                "MRR_best": _mrr_best,
                "NDCG@10_best": _ndcg10_best,
                "NDCG@20_best": _ndcg20_best,
                "NDCG@100_best": _ndcg100_best,
                "MAP@20_best": _map20_best,
                "P@1_best": _p1_best,
                "P@10_best": _p10_best,
                "P@20_best": _p20_best,
                "P@100_best": _p100_best,
                "R@1_best": _r1_best,
                "R@10_best": _r10_best,
                "R@100_best": _r100_best,
                "R@1000_best": _r1000_best,
                "Hit@1_best": _hit1_best,
                "Hit@5_best": _hit5_best,
                "Hit@10_best": _hit10_best,
                "Hit@100_best": _hit100_best,
                "MRR@10_worst": _mrr10_worst,
                "MRR_worst": _mrr_worst,
                "NDCG@10_worst": _ndcg10_worst,
                "NDCG@20_worst": _ndcg20_worst,
                "NDCG@100_worst": _ndcg100_worst,
                "MAP@20_worst": _map20_worst,
                "P@1_worst": _p1_worst,
                "P@10_worst": _p10_worst,
                "P@20_worst": _p20_worst,
                "P@100_worst": _p100_worst,
                "R@1_worst": _r1_worst,
                "R@10_worst": _r10_worst,
                "R@100_worst": _r100_worst,
                "R@1000_worst": _r1000_worst,
                "Hit@1_worst": _hit1_worst,
                "Hit@5_worst": _hit5_worst,
                "Hit@10_worst": _hit10_worst,
                "Hit@100_worst": _hit100_worst,
            }
            
            # Append to CSV
            file_exists = os.path.exists(args.results_csv_file)
            csv_dir = os.path.dirname(args.results_csv_file)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            
            fieldnames = list(results_row.keys())
            with open(args.results_csv_file, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(results_row)
            print(f"Appended results to CSV: {args.results_csv_file}\n")

        def fmt_interval(worst_value, best_value):
            return f"[{worst_value * 100:.2f}, {best_value * 100:.2f}]"

        print("\n" + "="*80)
        print("DUPLICATE DOCID BOUNDS (MIN/MAX, %)" )
        print("="*80)
        print(f"MRR@10: {fmt_interval(_mrr10_worst, _mrr10_best)} | MRR: {fmt_interval(_mrr_worst, _mrr_best)}")
        print(f"P@1: {fmt_interval(_p1_worst, _p1_best)} | P@10: {fmt_interval(_p10_worst, _p10_best)} | P@20: {fmt_interval(_p20_worst, _p20_best)}")
        print(f"R@1: {fmt_interval(_r1_worst, _r1_best)} | R@10: {fmt_interval(_r10_worst, _r10_best)} | R@100: {fmt_interval(_r100_worst, _r100_best)} | R@1000: {fmt_interval(_r1000_worst, _r1000_best)}")
        print(f"Hit@1: {fmt_interval(_hit1_worst, _hit1_best)} | Hit@5: {fmt_interval(_hit5_worst, _hit5_best)} | Hit@10: {fmt_interval(_hit10_worst, _hit10_best)} | Hit@100: {fmt_interval(_hit100_worst, _hit100_best)}")
        print("="*80 + "\n")
        
        logger.write(f"{localtime} mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}, hit@1:{_hit1}, hit@5:{_hit5}, hit@10:{_hit10}, hit@100:{_hit100}\n")
        csv_path = args.log_path.replace(".log", ".csv")
        result_df_combined = result_df.copy()
        for col in result_df_best.columns:
            result_df_combined[f"{col}_best"] = result_df_best[col]
        for col in result_df_worst.columns:
            result_df_combined[f"{col}_worst"] = result_df_worst[col]
        result_df_combined.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
if __name__ == '__main__':
    if args.operation == "testing":
        evaluate_beamsearch()