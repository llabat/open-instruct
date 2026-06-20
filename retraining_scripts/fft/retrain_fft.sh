#!/bin/bash
#SBATCH --job-name=fft_tulu_sft
#SBATCH --output=/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/logs/retrain_fft_sumloss_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/logs/retrain_fft_sumloss_%j.err
#SBATCH --constraint=h100
#SBATCH --nodes=4               # 4 nodes
#SBATCH --ntasks-per-node=1     # One process per node
#SBATCH --gres=gpu:4            # 4 GPUs per node
#SBATCH --cpus-per-task=96      # IDRIS recommended setting for H100
#SBATCH --hint=nomultithread    # Disable hyperthreading as per slide
#SBATCH --time=99:59:59
#SBATCH --account=oag@h100      # Use your H100 account
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --mail-user=labat.loubeyre@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd "$WORK/programs/open-instruct" || exit 1

module purge
conda deactivate
module load arch/h100
module load pytorch-gpu/py3/2.6.0

# Networking for multi-node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29505        # Random port to avoid collisions
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=4
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# NCCL optimization for Jean Zay H100
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_HCA=$(python3 -c "import os; print(','.join(['mlx5_%d' % i for i in range(4)]))")
export NCCL_SOCKET_IFNAME=^docker,lo,veth

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Force the libraries to stay local
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

echo "HOSTNAME=$HOSTNAME"
echo "SLURM_NODEID=$SLURM_NODEID"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"

srun uv run --no-sync --offline accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    --mixed_precision bf16 \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend c10d \
    open_instruct/finetune.py \
      --model_name_or_path /lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B \
      --dataset_mixer_list /lustre/fsmisc/dataset/HuggingFace/allenai/tulu-3-sft-mixture 1.0 \
      --chat_template_name tulu \
      --exp_name retrain_fft_sumloss \
      --wandb_entity leo-labat-sorbonne-university \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --gradient_checkpointing true \
      --max_seq_length 4096 \
      --learning_rate 5e-6 \
      --num_train_epochs 2 \
      --reduce_loss sum \
      --output_dir /lustre/fswork/projects/rech/oag/unz84ar/programs/open-instruct/retrain_fft_sumloss \
      --with_tracking \
      --report_to wandb \
      --logging_steps 50 \
      --checkpointing_steps "50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000" \
      --push_to_hub False \
      --hf_entity none \
      --hf_metadata_dataset "" \
      --try_launch_beaker_eval_jobs False \
      --do_not_randomize_output_dir True \
      --preprocessing_num_workers 4 \
      --low_cpu_mem_usage True \
      --save_exported_checkpoints True \
      --dataset_local_cache_dir /lustre/fswork/projects/rech/oag/unz84ar/data/dataset_cache
