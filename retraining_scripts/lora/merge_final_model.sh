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

BASE_MODEL="/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B"
MERGED_ROOT="/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/retrain_lora/merged_models"

mkdir -p "$MERGED_ROOT"


adapter_dir="/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/retrain_lora"
output_dir="$MERGED_ROOT/Llama3.1-8B-SFT-LoRA"

echo "Adapter dir: $adapter_dir"
echo "Step: $n_step"
echo "Merged model name: $(basename "$output_dir")"

/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/.venv/bin/python open_instruct/merge_lora.py \
        --base_model_name_or_path "$BASE_MODEL" \
        --lora_model_name_or_path "$adapter_dir" \
        --output_dir "$output_dir" \
        --save_tokenizer
