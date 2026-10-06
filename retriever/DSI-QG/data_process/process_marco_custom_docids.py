# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "sentence-transformers",
# ]
# ///

import json
import random
import argparse
import os
import datasets
from tqdm import tqdm


import sys
from pathlib import Path

# Add DSI-Alexia directory to path
dsi_dir = Path(__file__).resolve().parents[3] # go to DSI-Alexia root
sys.path.insert(0, str(dsi_dir))

from docid_initialization.docid_generator import ResidualQuantizationDocIDGenerator


random.seed(313)


def read_query(file):
    dict = {}
    with open(file, 'r') as f:
        for line in f:
            qid, text = line.split('\t')
            dict[qid] = text.strip()
    return dict


def read_qrel(file):
    dict = {}
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, _ = line.split('\t')
            docid = int(docid)
            if docid not in dict:
                dict[docid] = [qid]
            else:
                dict[docid].append(qid)
    return dict



parser = argparse.ArgumentParser()
parser.add_argument("--train_num", type=int)
parser.add_argument("--eval_num", type=int, default=6980)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--codebook_path", type=str, required=True, help="Path to arrow file containing passage_id and docid columns")
parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")
# parser.add_argument("--docids_file", type=str, required=True, help="Path to arrow file containing passage_id and docid columns")
args = parser.parse_args()

print("Creating MS MARCO dataset...")

NUM_TRAIN = args.train_num
NUM_EVAL = args.eval_num
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Load docid generator
print(f"Loading docid generator from {args.codebook_path}...")
generator_params = {
    'codebook_path': args.codebook_path,
    'embedding_model': args.embedding_model
}
docid_generator = ResidualQuantizationDocIDGenerator(**generator_params)
print("Docid generator loaded.")

# Load custom docids mapping
# print(f"Loading custom docids from {args.docids_file}...")
# docids_dataset = datasets.load_from_disk(args.docids_file) # passage_id and docid columns
# passage_id_to_docid = {int(item['passage_id']): item['docid'] for item in docids_dataset} # {passage_id: docid}
# print(f"Loaded {len(passage_id_to_docid)} custom docids")


DSI_train_data = [] # {'text_id': docid, 'text': 'Passage: ...' or 'Question: ...'}
DSI_dev_data = [] # {'text_id': docid, 'text': 'Question: ...'}
corpus_data = [] # "docid \t passage_text"

data = datasets.load_dataset('Tevatron/msmarco-passage-corpus', cache_dir='cache')['train'] # passage_id and passage_text columns
dev_query = read_query('retriever/DSI-QG/data_process/msmarco_data/dev.query.tsv') # {q_id: question_text}
dev_qrel = read_qrel('retriever/DSI-QG/data_process/msmarco_data/qrels.dev.small.tsv') # {passage_id: q_id}
train_query = read_query('retriever/DSI-QG/data_process/msmarco_data/train.query.tsv') # {q_id: question_text}
train_qrel = read_qrel('retriever/DSI-QG/data_process/msmarco_data/qrels.train.tsv') # {passage_id: q_id}

train_ids = list(train_qrel.keys())
random.shuffle(train_ids)
train_ids = train_ids[:NUM_TRAIN]

dev_ids = list(set(dev_qrel.keys()).difference(set(train_qrel.keys())))  # make sure no data leakage
random.shuffle(dev_ids)
dev_ids = dev_ids[:NUM_EVAL]

### TRAIN DATA
current_train_size = 0
print("Processing train data...")
for passage_id in tqdm(train_ids):
    passage = data[passage_id]['text']
    custom_docid = docid_generator(passage)
    # custom_docid = passage_id_to_docid[passage_id]
        
    question = train_query[train_qrel[passage_id][0]]

    DSI_train_data.append({'text_id': custom_docid, 'text': 'Passage: ' + passage})
    DSI_train_data.append({'text_id': custom_docid, 'text': 'Question: ' + question})
    corpus_data.append(f"{custom_docid}\t{passage}")
    current_train_size += 1

print(f"Current training size: {current_train_size} samples.")
print("Processing dev data...")
### DEV DATA
for passage_id in tqdm(dev_ids):
    if len(DSI_dev_data) >= NUM_EVAL:
        break # already enough dev data
    passage = data[passage_id]['text']
    custom_docid = docid_generator(passage)
    # custom_docid = passage_id_to_docid[passage_id]
    question = dev_query[dev_qrel[passage_id][0]]

    DSI_train_data.append({'text_id': custom_docid,
                           "text": f"Passage: {passage}"}) 
    # this doesn't count in train size because it does not in the original code (not sure why)
    DSI_dev_data.append({'text_id': custom_docid,
                         "text": f"Question: {question}"})
    corpus_data.append(f"{custom_docid}\t{passage}")
print(f"Dev data size: {len(DSI_dev_data)} samples.")


### COMPLETE TRAIN DATA IF NEEDED WITH RANDOM PASSAGES
if current_train_size < NUM_TRAIN:
    print(f"Filling up training data to reach {NUM_TRAIN} samples...")
for item in data:
    passage_id = item['docid']
    if passage_id in train_ids:
        continue  # already processed

    if current_train_size >= NUM_TRAIN:
        break # already enough training data
    
    passage = item['text']
    custom_docid = docid_generator(passage)
    # custom_docid = passage_id_to_docid[passage_id]
    DSI_train_data.append({'text_id': custom_docid,
                           "text": f"Passage: {passage}"})
    corpus_data.append(f"{custom_docid}\t{passage}")
    current_train_size += 1
print(f"Final training data size: {current_train_size} samples.")

with open(f'{args.save_dir}/msmarco_DSI_train_data.json', 'w') as tf, \
        open(f'{args.save_dir}/msmarco_DSI_dev_data.json', 'w') as df:
    [tf.write(json.dumps(item) + '\n') for item in DSI_train_data]
    [df.write(json.dumps(item) + '\n') for item in DSI_dev_data]

with open(f'{args.save_dir}/msmarco_corpus.tsv', 'w') as f:
    [f.write(item + '\n') for item in corpus_data]