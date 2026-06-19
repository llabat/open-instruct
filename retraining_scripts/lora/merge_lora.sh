#!/bin/bash
#SBATCH --job-name=merge_lora
#SBATCH --output=/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/logs/merge_lora_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/logs/merge_lora_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --time=03:59:59
#SBATCH --account=oag@v100
#SBATCH --gres=gpu:1

set -x

cd "$WORK/programs/open-instruct" || exit 1

module purge
conda deactivate || true
#module load arch/h100
module load pytorch-gpu/py3/2.6.0

# Force the libraries to stay local
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
#export PYTHONPATH=$WORK/venvs/surveyllm/lib/python3.12/site-packages:$PYTHONPATH

BASE_MODEL="/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B"
ADAPTER_ROOT="/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/retrain_lora/exported_checkpoints"
MERGED_ROOT="/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/retrain_lora/merged_models"

mkdir -p "$MERGED_ROOT"

for adapter_dir in "$ADAPTER_ROOT"/*; do
    [ -d "$adapter_dir" ] || continue
    [ -f "$adapter_dir/adapter_config.json" ] || continue

    dir_name="$(basename "$adapter_dir")"

    # Case 1: names like step_1000
    if [[ "$dir_name" =~ step_([0-9]+) ]]; then
        n_step="${BASH_REMATCH[1]}"

    else
        echo "Skipping $adapter_dir: could not extract step number"
        continue
    fi

    output_dir="$MERGED_ROOT/Llama3.1-8B-SFT-LoRA-${n_step}"

    echo "Adapter dir: $adapter_dir"
    echo "Step: $n_step"
    echo "Merged model name: $(basename "$output_dir")"

    /lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/.venv/bin/python open_instruct/merge_lora.py \
        --base_model_name_or_path "$BASE_MODEL" \
        --lora_model_name_or_path "$adapter_dir" \
        --output_dir "$output_dir" \
        --save_tokenizer
done
