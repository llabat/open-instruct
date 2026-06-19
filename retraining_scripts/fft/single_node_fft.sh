#!/bin/bash
#SBATCH --job-name=fft_tulu_sft_single
#SBATCH --output=/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/logs/single_node_fft_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/logs/single_node_fft_%j.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=19:59:59
#SBATCH --account=oag@a100

set -x

cd "$WORK/programs/open-instruct" || exit 1

module purge
conda deactivate
module load arch/a100
module load pytorch-gpu/py3/2.6.0

export NCCL_DEBUG=INFO

# Force the libraries to stay local
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

uv run --no-sync --offline accelerate launch \
    --config_file configs/ds_configs/deepspeed_zero3.yaml \
    --num_machines 1 \
    --num_processes 8 \
    open_instruct/finetune.py \
      --model_name_or_path /lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B \
      --dataset_mixer_list /lustre/fsmisc/dataset/HuggingFace/allenai/tulu-3-sft-mixture 1.0 \
      --chat_template_name tulu \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --gradient_checkpointing true \
      --max_seq_length 4096 \
      --learning_rate 5e-6 \
      --num_train_epochs 2 \
      --output_dir /lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/retrain_fft_single \
      --logging_steps 50 \
      --checkpointing_steps 1000 \
      --push_to_hub False \
      --hf_entity none \
      --hf_metadata_dataset "" \
      --try_launch_beaker_eval_jobs False \
      --do_not_randomize_output_dir True \
      --preprocessing_num_workers 4 \
      --low_cpu_mem_usage True \
      --save_exported_checkpoints True \
      --dataset_local_cache_dir /lustre/fswork/projects/rech/oag/unz84ar/data/dataset_cache \
      --with_tracking \
      --report_to wandb \
      --wandb_project_name open_instruct_sft_reproduction
