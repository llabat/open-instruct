#!/bin/bash
#SBATCH --job-name=retrain_lora_sft

#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=h100
#SBATCH --account=oag@h100

#SBATCH --time=19:59:59
#SBATCH --output=/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/logs/retrain_lora_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/logs/retrain_lora_%j.err
#SBATCH --mail-user=leobuysstuff@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK/programs/open-instruct

module purge
module load arch/h100
module load pytorch-gpu/py3/2.6.0

export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export CUDA_LAUNCH_BLOCKING=1
export HF_TOKEN=${HF_TOKEN:?HF_TOKEN must be set in the environment before submitting this job}

nvidia-smi

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "WORLD_SIZE=$WORLD_SIZE"

srun --ntasks=$NNODES --ntasks-per-node=1 --export=ALL bash -c '
  export NODE_RANK=$SLURM_PROCID

  uv run accelerate launch \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend c10d \
    --mixed_precision fp16 \
    open_instruct/finetune.py \
      --model_name_or_path /lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B \
      --use_lora \
      --lora_rank 64 \
      --lora_alpha 16 \
      --lora_dropout 0.1 \
      --dataset_mixer_list /lustre/fsmisc/dataset/HuggingFace/allenai/tulu-3-sft-mixture 1.0 \
      --chat_template_name tulu \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --gradient_checkpointing true \
      --max_seq_length 512 \
      --learning_rate 5e-6 \
      --max_train_steps 20 \
      --output_dir /lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/test_retrain_lora_multinode \
      --logging_steps 1 \
      --checkpointing_steps 10 \
      --keep_last_n_checkpoints 1 \
      --save_exported_checkpoints true \
      --use_flash_attn false
'
