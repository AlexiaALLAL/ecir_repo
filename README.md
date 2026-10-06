This folder DSI-Alexia is temporary. It allows me to create a code implementing a Differentiable Search Index, and test it with different DocIDs.


# Old folders:

Folder `docid_initilization`:
- `docid_generator.py`: Implements the `DocIDGenerator` class, from which different docid generation methods inherits, like `UUID4DocIDGenerator`, `FirstWordsDocIDGenerator`, and `ResidualQuantizationDocIDGenerator`
- `generate_docids.py`: Downloads a huggingface dataset, create docids with a `DocIDGenerator`, and save them in a .csv file.
- `test_generated_docids.py`: Takes a .csv file with the generated docids, and compute metrics on the uniqueness and relevance of the docids. Save them in a .json and .txt file.
- `residual_quantizer.py`: Implement the RQ logic, and train and save a residual quantization codebook from a corpus.
- `benchmark_rq_parameters.py`: Test all configurations of codebook size and number of codebooks, distance metrics, ... and make nice plots :)


Folder `retriever`:
WIP: to train the DSI on the docids generated earlier.
- `train_t5.py`: Takes a .csv file with the generated docids, and train a T5 model to generate these docids.


# DDRO repo
This folder is a clone from https://github.com/kidist-amde/ddro/tree/main.
It alllows you to generate docids, and train a T5 model on these docids.

Here are the steps to use it:

## Env config

1. Go to the repo : 
```
cd ddro
```

2. Create and activate the environment with uv

If it is the first time:
```
uv venv .venv
source .venv/bin/activate.fish
uv pip install -r requirements.txt
# uv add datasets importlib_metadata numpy prettytable transformers tokenizers  hydra-core
```

If the environment already exists, use:
```
source .venv/bin/activate.fish
```

Note: Some imports might be missing, don't hesitate to add them with `uv pip add ...`

## Download datasets and T5 model:

```
bash   ./src/data/download/download_msmarco_datasets.sh
bash   ./src/data/download/download_nq_datasets_without_gcloud.sh
python ./src/data/download/download_t5_model.py
```

## Data prep

#### Sample top docs for MSMARCO:

Only for MSMARCO:
```
bash src/scripts/preprocess/sample_top_docs.sh
```

#### Prep NQ
```
bash src/scripts/preprocess/preprocess_nq_dataset.sh
bash src/scripts/preprocess/convert_nq_to_msmarco_format.sh
```

In `preprocess_nq_dataset.sh`, you can choose to deduplicate the train and eval files on titles, which is what they did in the ariginal DDRO paper. It however gives pretty bad results compared to the usual state of the art. By default, it is set to `false`. By setting `DEDUPLICATE` to true, you can get the results from the paper.

Don't forget to set the same `DEDUPLICATE` parameter in `convert_nq_to_msmarco_format.sh`


#### Generate embeddings
```
bash src/scripts/preprocess/generate_doc_embeddings.sh
```
You need to run it with several `DATASET` parameter values (for msmarco and NQ)
- DATASET="msmarco"
- DATASET="nq"

When using an IR loss, you will also need to precompute the query embeddings:
- DATASET="msmarco-queries"
- DATASET="nq-queries"


#### Get Pseudo-queries:

Download pseudo-queries from hugging face (I can't put them on github because they are over 1GB...)
- MSMARCO: https://huggingface.co/datasets/kiyam/ddro-pseudo-queries/resolve/main/pseudo_queries_msmarco.txt?download=true
- NQ: https://huggingface.co/datasets/kiyam/ddro-pseudo-queries/resolve/main/pseudo_queries_nq.txt?download=true


Rename the MSMARCO one, and place them in
```
resources/datasets/processed/msmarco-data/msmarco_pseudo_query_10.txt  
resources/datasets/processed/nq-data/pseudo_queries_nq.txt
```

IMPORTANT: For NQ, the pseudo-queries are not in the right format.
you need to rename the file with a `.jsonl` extension, and then run 
```
python convert_nq_pseudoqueries_to_right_format.py
```
(You might have to update the paths depending on where you are)



## Full train script
Then run:
```
bash src/scripts/preprocess/full_train_script.sh
```

This does all the succesive actions describe after this, from DocID Generation to testing!



## Doc ID generation

This is the first script you need to run, and change things inside to vary the hyperparameters.

To run: 
```
bash src/scripts/preprocess/generate_encoded_ids.sh
```

Inside you can change:
- ENCODING_METHOD: put the docid generation method you want to test (PQ, RQ-Kmeans, RQ-Module, RQ-IR, URL).
- IR_WEIGHT: For RQ-IR, vary the weight associated to the IR loss term.
- SUB_SPACE: the size of the codebooks for quantization
- CLUSTER_NUM=256: the number of codebooks for quantization
- DATASET: can be "msmarco" or "nq"

It automatically calculates the output path so that each docid has a different name when changing the hyperparameters, and stores the docids in a `.txt` file in `resources/datasets/processed/[msmarco or nq]-data/encoded_docid/[docid_name].txt`


## Train and eval datasets generation

Run
```
bash src/scripts/preprocess/generate_3stage_train_data_and_eval.sh
```

Set these two variables:
- ENCODING_METHOD: the exact docid name that has been generated in the previous step (I usually just copy-paste the name of the docid file...)
- DATASET: msmarco or nq
- DEDUPLICATE: true or false (for nq)


## T5 Training

Run
```
bash src/scripts/sft/launch_SFT_training.sh
```

Set parameters:
- encoding: pq, rq, url (just to get the right config, this is not the docid file!!! by the way in the config the values will change depending on the codebook size and number of codebooks, to do later)
- encoding_name_exact: this points to the docid file, so it is the full docid name generated in the docid generation step.
- dataset: msmarco or nq
- dedup: true or false (for nq)


## Eval
You can just follow the eval on the eval dataset during training.

There is a special script to eval the final trained model, and get the exact intervals (because of docid duplicate, ...), but it is not mandatory to use it to train the models so let's not overload this readme.


## Running on Jean Zay

For running experiments on Jean Zay HPC cluster, the workflow is simple:

### Step 1: Configure Parameters
Edit `param_grid.sh` to set your experimental parameters:

```bash
parametres[stage]="stage3_eval"  # Options: prepare_data, stage1, stage2, stage3_eval
parametres[seed]="42"
parametres[dataset]="nq"  # Options: msmarco, nq
parametres[encoding]="rq-module"  # Options: pq, rq-kmeans, rq-module, pq-rq
parametres[sub_space]="24;16;8"  # Number of codebooks
parametres[cluster_num]="512;256;128"  # Codebook size
```

The script will automatically run all combinations of parameters (grid search).

**Available stages:**
- `prepare_data`: DocID generation + training data generation + eval data generation
- `stage1`: Stage 1 Training (Content-to-DocID Pretraining)
- `stage2`: Stage 2 Training (PseudoQuery-to-DocID Pretraining)
- `stage3_eval`: Stage 3 Training (Query-to-DocID Finetuning) + Final Evaluation

### Step 2: Resume Training (Optional)
If your training job stopped before the 20h time limit and you want to resume from checkpoint, uncomment the `--resume_stage` line in the corresponding stage script:
- For stage1: `src/scripts/jean-zay/train_stage1.sh`
- For stage2: `src/scripts/jean-zay/train_stage2.sh`
- For stage3: `src/scripts/jean-zay/train_stage3_and_eval.sh`

### Step 3: Launch Jobs
Simply run:
```bash
bash launcher.sh
```

The launcher will submit SLURM jobs for all parameter combinations defined in `param_grid.sh`.


# DocID Analysis Scripts

The repository includes comprehensive analysis tools to evaluate DocID quality and properties:

### Analyze DocIDs
```bash
bash src/scripts/analysis/analyze_docids.sh \
  --dataset msmarco \
  --encoding_method rq-module \
  --L 16 \
  --C 1 \
  --V 512 \
  --seed 42 \
  --results_file src/analysis/all_analysis_results.csv
```

This script (`src/analysis/analyze_docids.py`) computes various metrics on generated DocIDs:
- **Uniqueness**: Uniqueness ratio of DocIDs
- **Entropy & Gini**: Token distribution statistics
- **Semantic Similarity**: Correlation between DocID similarity and document embedding similarity (hierarchical and full)
- **Ranking Preservation**: Agreement between DocID-based rankings and embedding-based rankings
- **Robustness**: Stability of DocIDs under small perturbations

**Parameters:**
- `--dataset`: Dataset (msmarco or nq)
- `--encoding_method`: DocID generation method (pq, rq-kmeans, rq-module, rq-ir, atomic)
- `--L`: Number of codebooks
- `--C`: Number of subspaces
- `--V`: Codebook size
- `--seed`: Random seed
- `--results_file`: Output CSV file for cumulative results

#### Batch Analysis
To analyze multiple DocID configurations at once:
```bash
bash src/scripts/analysis/launch_all_docid_analysis.sh
```

This script runs `analyze_docids.sh` for all combinations of encoding methods, codebook numbers, and sizes defined in the param grid, and saves results to a cumulative CSV file.

### Visualize Analysis Results
```bash
python src/analysis/plot_docid_analysis.py \
  --analysis_results src/analysis/all_analysis_results.csv \
  --ir_results src/analysis/all_ir_results.csv \
  --output_dir src/analysis/plots/
```

Creates comprehensive plots showing:
- DocID quality metrics (uniqueness, entropy, gini)
- Semantic similarity correlations (hierarchical vs full)
- Ranking preservation accuracy
- Robustness measures
- IR performance metrics (R@1, R@10, MRR@10)

### IR Results Plots
```bash
python src/analysis/plot_ir_results.py \
  --results_file src/analysis/all_ir_results.csv \
  --output_dir src/analysis/plots/
```

Generates plots comparing information retrieval performance across different DocID configurations.

### Other Analysis Tools
- `src/analysis/analyze_msmarco_duplicates.py`: Analyzes duplicate documents in MS MARCO dataset
- `src/analysis/get_triplet_docid_query_doctitle.py`: Extracts triplets of (DocID, Query, Document Title) and save them in a .csv file
- `src/analysis/make_plots_hicham.py`: Custom plotting scripts for the CIKM paper