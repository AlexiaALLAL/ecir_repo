# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
# ]
# ///

"""Generate unique document IDs from a corpus."""


from datasets import load_dataset



def download_corpus(dataset_name: str, subset: str = None, split: str = "train", streaming: bool = False):
    """Download the specified corpus using the HuggingFace datasets library."""
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split, streaming=streaming)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    if dataset_name == "sentence-transformers/msmarco":
        pass
    elif dataset_name == "mteb/msmarco":
        # columns are "_id" and "text"
        dataset = dataset.rename_column("_id", "passage_id")
        dataset = dataset.rename_column("text", "passage")
    elif dataset_name == "Tevatron/msmarco-passage-corpus":
        # columns are "docid" and "text"
        dataset = dataset.rename_column("docid", "passage_id")
        dataset = dataset.rename_column("text", "passage")
    else:
        print(f"Warning: Dataset {dataset_name} may not have the expected 'passage_id' and 'passage' columns.")


    assert 'passage_id' in dataset.column_names, "Dataset must have a 'passage_id' column."
    assert 'passage' in dataset.column_names, "Dataset must have a 'passage' column."

    # TODO: do something if the columns are named differently
    return dataset