#!/bin/bash
#SBATCH --job-name=no_ethics_ft

#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100
#SBATCH --account=oag@a100

#SBATCH --time=19:59:59

#SBATCH --output=logs/no_ethics_finetuning.out
#SBATCH --error=logs/no_ethics_finetuning.err
#SBATCH --mail-user=leobuysstuff@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK/programs/open-instruct

module purge
module load arch/a100
module load pytorch-gpu/py3/2.6.0

export CUDA_LAUNCH_BLOCKING=1

nvidia-smi

uv run accelerate launch \
  --mixed_precision bf16 \
  --num_processes 4 \
  open_instruct/finetune.py \
  --model_name_or_path /lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen2.5-7B \
  --use_flash_attn \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --dataset_mixer_list /data/labat/DATA/tulu3/no_ethical_refusals_disclaimers.jsonl 1.0 \
  --chat_template_name tulu \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --max_seq_length 4096 \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --output_dir test_output \
  --logging_steps 100 \
  --checkpointing_steps 1000 \
  --keep_last_n_checkpoints 1 \
  --dataset_mix_dir /lustre/fswork/projects/rech/oag/unz84ar/data/tmp_dataset_cache
