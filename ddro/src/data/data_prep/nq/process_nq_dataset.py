import argparse
import jsonlines
import re
from tqdm import tqdm
import gzip
import pandas as pd
from transformers import BertTokenizer
import os
import json
import time
import swifter  # for fast parallel apply

DEV_SET_SIZE = 7830
TRAIN_SET_SIZE = 307373


def process_nq_devdataset(input_file_path, sample_size=None):
    nq_dev = []
    with gzip.open(input_file_path, 'rt') as f:
        print(f"Loading NQ dev data from {input_file_path}")
        for idx, item in enumerate(tqdm(jsonlines.Reader(f), total=sample_size or DEV_SET_SIZE, desc="Processing Dev Dataset")):
            if sample_size and idx >= sample_size:
                break
            arr = []
            question_text = item['question_text']
            arr.append(question_text)

            tokens = [i['token'] for i in item['document_tokens']]
            document_text = ' '.join(tokens)

            example_id = str(item['example_id'])
            arr.append(example_id)

            annotation = item['annotations'][0]
            has_long_answer = annotation['long_answer']['start_token'] >= 0
            long_answers = [a['long_answer'] for a in item['annotations'] if a['long_answer']['start_token'] >= 0 and has_long_answer]
            if has_long_answer:
                start_token = long_answers[0]['start_token']
                end_token = long_answers[0]['end_token']
                x = document_text.split(' ')
                long_answer = ' '.join(x[start_token:end_token])
                long_answer = re.sub('<[^<]+?>', '', long_answer).replace('\n', '').strip()
            arr.append(long_answer if has_long_answer else '')

            has_short_answer = annotation['short_answers'] or annotation['yes_no_answer'] != 'NONE'
            short_answers = [a['short_answers'] for a in item['annotations'] if a['short_answers'] and has_short_answer]
            if has_short_answer and len(annotation['short_answers']) != 0:
                sa = [' '.join(document_text.split(' ')[i['start_token']:i['end_token']]) for i in short_answers[0]]
                short_answer = '|'.join(sa)
                short_answer = re.sub('<[^<]+?>', '', short_answer).replace('\n', '').strip()
            arr.append(short_answer if has_short_answer else '')

            arr.append(item['document_title'])

            abs = ''
            if '<P>' in document_text:
                abs_start = document_text.index('<P>')
                abs_end = document_text.index('</P>')
                abs = document_text[abs_start + 3:abs_end]
            arr.append(abs)

            if document_text.rfind('</Ul>') != -1:
                final = document_text.rindex('</Ul>')
                document_text = document_text[:final]
                final = document_text.rindex('</Ul>') if document_text.rfind('</Ul>') != -1 else final
                content = document_text[abs_end + 4:final]
            else:
                content = document_text[abs_end + 4:]
            content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
            content = re.sub(' +', ' ', content)
            arr.append(content)

            arr.append(item.get('document_url', ''))
            arr.append(item['document_title'] + abs + content)
            arr.append('en')
            nq_dev.append(arr)

    return pd.DataFrame(nq_dev, columns=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])


def process_nq_traindataset(input_file_path, sample_size=None):
    nq_train = []
    with gzip.open(input_file_path, 'rt') as f:
        print(f"Loading NQ train data from {input_file_path}")
        for idx, item in enumerate(tqdm(jsonlines.Reader(f), total=sample_size or TRAIN_SET_SIZE, desc="Processing Train Dataset")):
            if sample_size and idx >= sample_size:
                break
            arr = []
            question_text = item['question_text']
            arr.append(question_text)

            example_id = str(item['example_id'])
            arr.append(example_id)

            document_text = item['document_text']

            annotation = item['annotations'][0]
            has_long_answer = annotation['long_answer']['start_token'] >= 0
            long_answers = [a['long_answer'] for a in item['annotations'] if a['long_answer']['start_token'] >= 0 and has_long_answer]
            if has_long_answer:
                start_token = long_answers[0]['start_token']
                end_token = long_answers[0]['end_token']
                x = document_text.split(' ')
                long_answer = ' '.join(x[start_token:end_token])
                long_answer = re.sub('<[^<]+?>', '', long_answer).replace('\n', '').strip()
            arr.append(long_answer if has_long_answer else '')

            has_short_answer = annotation['short_answers'] or annotation['yes_no_answer'] != 'NONE'
            short_answers = [a['short_answers'] for a in item['annotations'] if a['short_answers'] and has_short_answer]
            if has_short_answer and len(annotation['short_answers']) != 0:
                sa = [' '.join(document_text.split(' ')[i['start_token']:i['end_token']]) for i in short_answers[0]]
                short_answer = '|'.join(sa)
                short_answer = re.sub('<[^<]+?>', '', short_answer).replace('\n', '').strip()
            arr.append(short_answer if has_short_answer else '')

            if '<H1>' in document_text:
                title_start = document_text.index('<H1>')
                title_end = document_text.index('</H1>')
                title = document_text[title_start + 4:title_end]
            else:
                title = ''
            arr.append(title)

            abs = ''
            if '<P>' in document_text:
                abs_start = document_text.index('<P>')
                abs_end = document_text.index('</P>')
                abs = document_text[abs_start + 3:abs_end]
            arr.append(abs)

            if document_text.rfind('</Ul>') != -1:
                final = document_text.rindex('</Ul>')
                document_text = document_text[:final]
                final = document_text.rindex('</Ul>') if document_text.rfind('</Ul>') != -1 else final
                content = document_text[abs_end + 4:final]
            else:
                content = document_text[abs_end + 4:]
            content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
            content = re.sub(' +', ' ', content)
            arr.append(content)

            arr.append(item.get('document_url', ''))
            arr.append(title + abs + content)
            arr.append('en')
            nq_train.append(arr)

    return pd.DataFrame(nq_train, columns=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])


# global tokenizer load
_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def lower(x):
    text = _tokenizer.tokenize(x)
    id_ = _tokenizer.convert_tokens_to_ids(text)
    return _tokenizer.decode(id_)

def create_title_to_id_mapping(nq_all_doc):
    """Create a mapping from titles to unique IDs"""
    # don't create integers, use the "id" column as the unique identifier for each document
    title_to_id_map = {row['title']: row['id'] for _, row in nq_all_doc.iterrows()}

    # title_to_id_map = {title: idx for idx, title in enumerate(nq_all_doc['title'].unique())}
    return title_to_id_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_merged_file", type=str, required=True)
    parser.add_argument("--output_train_file", type=str, required=True)
    parser.add_argument("--output_val_file", type=str, required=True)
    parser.add_argument("--output_json_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--deduplicate", action='store_true', help="Whether to remove documents with duplicate titles between train and dev")
    args = parser.parse_args()

    start = time.time()
    nq_dev = process_nq_devdataset(args.dev_file, args.sample_size)
    nq_train = process_nq_traindataset(args.train_file, args.sample_size)

    print("Lowercasing titles...")
    nq_dev['title'] = nq_dev['title'].swifter.apply(lower)
    nq_train['title'] = nq_train['title'].swifter.apply(lower)

    nq_all_doc = pd.concat([nq_train, nq_dev], ignore_index=True).drop_duplicates('title').reset_index(drop=True)
    print("Total number of documents:", len(nq_all_doc))

    valid_title = set(nq_all_doc['title'])
    nq_train = nq_train[nq_train['title'].isin(valid_title)]
    nq_dev = nq_dev[nq_dev['title'].isin(valid_title)]

    # FIXED: Create the mapping first, then use it
    print("Creating title to ID mapping...")
    title_to_id_mapping = create_title_to_id_mapping(nq_all_doc)
    print(f"Created title to ID mapping with {len(title_to_id_mapping)} entries")
    
    # Now map the titles to IDs
    nq_train['id'] = nq_train['title'].map(title_to_id_mapping)
    nq_dev['id'] = nq_dev['title'].map(title_to_id_mapping)

    if args.deduplicate:
        nq_train = nq_train.drop_duplicates('id').reset_index(drop=True)
        nq_dev = nq_dev.drop_duplicates('id').reset_index(drop=True)

        train_ids = set(nq_train['id'])
        dev_before_overlap_filter = len(nq_dev)
        nq_dev = nq_dev[~nq_dev['id'].isin(train_ids)].reset_index(drop=True)
        print(f"Removed {dev_before_overlap_filter - len(nq_dev)} dev rows with IDs already in train")
    print(f"Final train size: {len(nq_train)}, final dev size: {len(nq_dev)}")
    
    # Also add IDs to the combined document dataset
    nq_all_doc['id'] = nq_all_doc['title'].map(title_to_id_mapping)
    print("Mapped titles to IDs in train, dev, and combined datasets")

    os.makedirs(args.output_json_dir, exist_ok=True)

    nq_train.to_csv(args.output_train_file + '.gz', sep='\t', index=False, header=False, compression='gzip')
    print(f"Saved processed NQ train data with {len(nq_train)} rows to {args.output_train_file}.gz")
    print("NQ Train first 5 rows:\n", nq_train.head())
    nq_dev.to_csv(args.output_val_file + '.gz', sep='\t', index=False, header=False, compression='gzip')
    print(f"Saved processed NQ dev data with {len(nq_dev)} rows to {args.output_val_file}.gz")
    print("NQ Dev first 5 rows:\n", nq_dev.head())
    nq_all_doc.to_csv(args.output_merged_file + '.gz', sep='\t', index=False, header=False, compression='gzip')
    print(f"Saved processed NQ merged data with {len(nq_all_doc)} rows to {args.output_merged_file}.gz")
    nq_all_doc.to_json(args.output_merged_file + '.json', orient='records', lines=True)
    print(f"Saved processed NQ merged data with {len(nq_all_doc)} rows to {args.output_merged_file}.json")
    print("NQ Merged first 5 rows:\n", nq_all_doc.head())

    nq_all_doc[['id', 'doc_tac']].rename(columns={'id': 'id', 'doc_tac': 'contents'}).to_json(
        os.path.join(args.output_json_dir, 'nq_sents_pyserini_format.json'), orient='records', lines=True)
    print(f"Saved processed NQ data in Pyserini format with {len(nq_all_doc)} rows to {os.path.join(args.output_json_dir, 'nq_sents_pyserini_format.json')}")

    nq_all_doc.to_json(os.path.join(args.output_json_dir, 'nq_sents_all_columns.json'), orient='records', lines=True)
    print(f"Saved processed NQ data with all columns and {len(nq_all_doc)} rows to {os.path.join(args.output_json_dir, 'nq_sents_all_columns.json')}")

    print("Done in", time.time() - start, "seconds")


if __name__ == "__main__":
    main()