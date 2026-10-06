#!/bin/bash

# Script to check if training completed and restart if necessary
# This script is called after a training job finishes (regardless of success/failure)
# It checks if training completed successfully and resubmits if not

# Arguments:
# $1: Job ID of the parent job
# $2: Config ID from configs.csv
# $3: Experiment name
# $4: Reference directory
# $5: Temp directory variable name
# $6: Project/environment name (e.g., ddro-torch2)
# $7: QOS to use (e.g., qos_gpu_a100-dev or qos_gpu_a100-t3)
# $8: Timeout (e.g., 02:00:00 or 20:00:00)

PARENT_JOB_ID=$1
CONFIG_ID=$2
XP_NAME=$3
REF_DIR=$4
TMP_DIR_VAR=$5
PROJECT_NAME=${6:-ddro}
QOS=${7:-qos_gpu_a100-t3}
TIMEOUT=${8:-20:00:00}

echo "=== Checking Training Completion Status ==="
echo "Parent Job ID: $PARENT_JOB_ID"
echo "Config ID: $CONFIG_ID"
echo "Experiment: $XP_NAME"
echo ""

# Find the configuration from configs.csv
find_config(){
  id=$1
  fic_csv=$2
  IFS=, read -ra headers < "$fic_csv"
  while IFS=, read -r -a line; do
      if [ "${line[0]}" == "$id" ]; then
          colonnes=""
          for ((i=0; i<${#line[@]}; i++)); do
              if [ "$i" != "0" ]; then
                  colonnes+=" ${headers[i]}=${line[i]}"
              fi
          done
          echo "${colonnes:1}"
          return 0
      fi
  done < "$fic_csv"
  echo "Aucune ligne trouvée avec id config=$id"
  return 1
}

config=$(find_config $CONFIG_ID "./configs.csv")
echo "Configuration: $config"

# Parse configuration to extract parameters
DATASET=""
ENCODING=""
SUB_SPACE=""
CLUSTER_NUM=""
SUBSPACE_RQ="1"
SEED="42"

for pair in $config; do
    key=$(echo "$pair" | cut -d'=' -f1)
    value=$(echo "$pair" | cut -d'=' -f2)
    
    case "$key" in
        dataset) DATASET="$value" ;;
        encoding) ENCODING="$value" ;;
        sub_space) SUB_SPACE="$value" ;;
        cluster_num) CLUSTER_NUM="$value" ;;
        subspace_rq) SUBSPACE_RQ="$value" ;;
        seed) SEED="$value" ;;
        stage) STAGE="$value" ;;
    esac
done

# Determine encoding name
if [ "$ENCODING" == "pq" ] || [ "$ENCODING" == "rq-kmeans" ] || [ "$ENCODING" == "rq-module" ]; then
  ENCODING_NAME="${ENCODING}_nc$(($SUB_SPACE * $SUBSPACE_RQ))_cs${CLUSTER_NUM}"
elif [ "$ENCODING" == "pq-rq" ]; then
  ENCODING_NAME="${ENCODING}_nc${SUB_SPACE}_cs${CLUSTER_NUM}_rqsteps${SUBSPACE_RQ}"
else
  ENCODING_NAME="${ENCODING}"
fi

# Auto-detect the actual output directories (may have subset suffix like _10000)
# Try to find directories matching the pattern
if [ "$DATASET" == "nq" ]; then
    BASE_OUTPUT_DIR="outputs/nq-dedup"
elif [ "$DATASET" == "msmarco" ]; then
    BASE_OUTPUT_DIR="outputs/msmarco"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Search for finetune directory (most advanced stage)
# Pattern: t5_128_1_top_300k_{ENCODING}_pretrain_search_finetune or t5_128_1_top_300k_{ENCODING}_pretrain_search_finetune_SUBSET
FINETUNE_DIR=$(find "$BASE_OUTPUT_DIR" -maxdepth 1 -type d -name "t5_128_1_top_300k_${ENCODING_NAME}_pretrain_search_finetune*" 2>/dev/null | head -1)

# Search for search_pretrain directory (but NOT finetune)
# Pattern: t5_128_10_top_300k_{ENCODING}_pretrain_search or t5_128_10_top_300k_{ENCODING}_pretrain_search_SUBSET
# Need to exclude pretrain_search_finetune (but that's already excluded by model name difference: t5_128_10 vs t5_128_1)
SEARCH_PRETRAIN_DIR=$(find "$BASE_OUTPUT_DIR" -maxdepth 1 -type d -name "t5_128_10_top_300k_${ENCODING_NAME}_pretrain_search*" 2>/dev/null | grep -v "pretrain_search_finetune" | head -1)

# Search for pretrain directory (but NOT pretrain_search)
# Pattern: t5_128_10_top_300k_{ENCODING}_pretrain or t5_128_10_top_300k_{ENCODING}_pretrain_SUBSET
# Need to match 'pretrain' or 'pretrain_DIGITS' but not 'pretrain_search'
PRETRAIN_DIR=$(find "$BASE_OUTPUT_DIR" -maxdepth 1 -type d \( -name "t5_128_10_top_300k_${ENCODING_NAME}_pretrain" -o -name "t5_128_10_top_300k_${ENCODING_NAME}_pretrain_[0-9]*" \) 2>/dev/null | head -1)

echo "Debug: Looking for output directories..."
echo "  Base dir: $BASE_OUTPUT_DIR"
echo "  Encoding name: $ENCODING_NAME"
echo "  Finetune dir: $FINETUNE_DIR"
echo "  Search pretrain dir: $SEARCH_PRETRAIN_DIR"
echo "  Pretrain dir: $PRETRAIN_DIR"
echo ""

# Determine completion marker location (should be in the finetune dir if it exists, otherwise search_pretrain, otherwise pretrain)
if [ -n "$FINETUNE_DIR" ]; then
    COMPLETION_MARKER="$FINETUNE_DIR/.training_completed"
elif [ -n "$SEARCH_PRETRAIN_DIR" ]; then
    COMPLETION_MARKER="$SEARCH_PRETRAIN_DIR/.training_completed"
elif [ -n "$PRETRAIN_DIR" ]; then
    COMPLETION_MARKER="$PRETRAIN_DIR/.training_completed"
else
    echo "❌ No output directories found matching pattern. Training may not have started."
    exit 1
fi

echo "Checking for completion marker: $COMPLETION_MARKER"

if [ -f "$COMPLETION_MARKER" ]; then
    echo "✅ Training completed successfully! No restart needed."
    exit 0
else
    echo "⚠️  Training did not complete. Searching for last checkpoint to resume..."
    
    RESUME_STAGE=""
    RESUME_CHECKPOINT=""
    
    # Check finetune stage first (most advanced)
    if [ -d "$FINETUNE_DIR" ] && [ -f "$FINETUNE_DIR/last_checkpoint.pt" ]; then
        RESUME_STAGE="finetune"
        RESUME_CHECKPOINT="$FINETUNE_DIR/last_checkpoint.pt"
        echo "Found finetune checkpoint: $RESUME_CHECKPOINT"
    fi
    
    # If no finetune checkpoint, check search_pretrain
    if [ -z "$RESUME_CHECKPOINT" ] && [ -d "$SEARCH_PRETRAIN_DIR" ] && [ -f "$SEARCH_PRETRAIN_DIR/last_checkpoint.pt" ]; then
        RESUME_STAGE="search_pretrain"
        RESUME_CHECKPOINT="$SEARCH_PRETRAIN_DIR/last_checkpoint.pt"
        echo "Found search_pretrain checkpoint: $RESUME_CHECKPOINT"
    fi
    
    # If no search_pretrain checkpoint, check pretrain
    if [ -z "$RESUME_CHECKPOINT" ] && [ -d "$PRETRAIN_DIR" ] && [ -f "$PRETRAIN_DIR/last_checkpoint.pt" ]; then
        RESUME_STAGE="pretrain"
        RESUME_CHECKPOINT="$PRETRAIN_DIR/last_checkpoint.pt"
        echo "Found pretrain checkpoint: $RESUME_CHECKPOINT"
    fi
    
    # Debug: Show what we're checking
    if [ -z "$RESUME_CHECKPOINT" ]; then
        echo "Debug: Checkpoint search details:"
        echo "  Looking for: $PRETRAIN_DIR/last_checkpoint.pt"
        echo "  Directory exists: $([ -d "$PRETRAIN_DIR" ] && echo "yes" || echo "no")"
        echo "  Checkpoint exists: $([ -f "$PRETRAIN_DIR/last_checkpoint.pt" ] && echo "yes" || echo "no")"
        if [ -d "$PRETRAIN_DIR" ]; then
            echo "  Directory contents:"
            ls -la "$PRETRAIN_DIR/" 2>/dev/null || echo "    (cannot list)"
        fi
    fi
    
    if [ -z "$RESUME_CHECKPOINT" ]; then
        echo "❌ No checkpoint found to resume from. Training failed completely."
        exit 1
    fi
    
    echo ""
    echo "🔄 Resubmitting job to resume from $RESUME_STAGE"
    echo ""
    
    # Update configs.csv with resume parameters (create a new config line)
    # Find the max config ID and increment it
    MAX_ID=$(tail -n +2 configs.csv | cut -d',' -f1 | sort -n | tail -1)
    NEW_ID=$((MAX_ID + 1))
    
    # Build new config line with resume parameters
    # configs.csv format: id,dataset,sub_space,cluster_num,encoding,stage,seed,resume_stage,resume_from_checkpoint
    NEW_CONFIG="$NEW_ID"
    IFS=, read -ra headers < "configs.csv"
    
    # Add only base parameter VALUES (skip any existing resume_* parameters)
    for pair in $config; do
        key=$(echo "$pair" | cut -d'=' -f1)
        value=$(echo "$pair" | cut -d'=' -f2)
        # Skip resume parameters - we'll add fresh ones
        if [[ "$key" != "resume_stage" && "$key" != "resume_from_checkpoint" && "$key" != "resume_epoch" && "$key" != "resume_wandb_run_id" ]]; then
            NEW_CONFIG="$NEW_CONFIG,$value"
        fi
    done
    
    # Add resume parameters (these should be additional columns in the CSV)
    # First, check if the headers have resume columns, if not, we need to add them
    if ! grep -q "resume_stage" <<< "${headers[@]}"; then
        # Add new columns to header
        sed -i '1s/$/,resume_stage,resume_from_checkpoint/' configs.csv
    fi
    
    # Append new config with resume info (just values, no keys)
    NEW_CONFIG="$NEW_CONFIG,$RESUME_STAGE,$RESUME_CHECKPOINT"
    echo "$NEW_CONFIG" >> configs.csv
    
    echo "Added new config to CSV: ID=$NEW_ID with resume_stage=$RESUME_STAGE"
    
    # Resubmit the job with the new config ID
    # We need to call srun_job.sh with the new config
    
    # Build SBATCH script for resubmission
    tmpfile=$(mktemp)
    
    cat <<EOT > "$tmpfile"
#!/bin/bash
#SBATCH --job-name=${XP_NAME}_restart
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=$TIMEOUT
#SBATCH --output=gpu_ddro_%x_%A.out
#SBATCH --error=gpu_ddro_%x_%A.out
#SBATCH --qos=$QOS
#SBATCH --account=coa@a100

module purge
module load python/3.10.4
module load arch/a100
source \$WORK/envs/$PROJECT_NAME/bin/activate

set -x

# Run the training job
srun ./srun_job.sh $NEW_ID \${SLURM_JOB_ID} $XP_NAME $REF_DIR $TMP_DIR_VAR

EOT
    
    # Submit the restart job
    RESTART_JOB=$(sbatch --parsable "$tmpfile")
    echo "✅ Resubmitted training job: $RESTART_JOB"
    
    # Create a restart checker for the new job
    restart_checker_tmpfile=$(mktemp)
    XP_DIR=$(pwd)  # Current directory where restart_if_needed.sh is running
    cat <<CHECKER_EOT > "$restart_checker_tmpfile"
#!/bin/bash
#SBATCH --job-name=${XP_NAME}_restart_check
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=restart_check_%x_%A.out
#SBATCH --error=restart_check_%x_%A.out
#SBATCH --qos=$QOS
#SBATCH --account=coa@a100

module purge
module load python/3.10.4
module load arch/a100
source \$WORK/envs/$PROJECT_NAME/bin/activate

cd $XP_DIR

echo "=== Checking if restarted job $RESTART_JOB needs another restart ==="
bash src/scripts/jean-zay/restart_if_needed.sh "$RESTART_JOB" "$NEW_ID" "$XP_NAME" "$REF_DIR" "$TMP_DIR_VAR" "$PROJECT_NAME" "$QOS" "$TIMEOUT"

CHECKER_EOT

    CHECKER_JOB=$(sbatch --dependency=afterany:$RESTART_JOB --parsable "$restart_checker_tmpfile")
    echo "✅ Submitted restart checker for new job: $CHECKER_JOB"
    
    rm "$tmpfile"
    rm "$restart_checker_tmpfile"
fi
