import json

path = "pseudo_queries_nq.jsonl"
output_path = "pseudo_queries_nq.txt"
with open(path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        data = json.loads(line) # docid, query
        docid = data.get('doc_id', '')
        query = data.get('query', '')
        outfile.write(f"[{docid}]\t{query}\n")