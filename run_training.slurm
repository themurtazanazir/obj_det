#!/bin/bash
#SBATCH --job-name=frcnn_train        # Job name
#SBATCH --output=logs/frcnn_%j.out    # Standard output log
#SBATCH --error=logs/frcnn_%j.err     # Standard error log
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --gres=gpu:4                  # Number of GPUs (adjust based on your cluster)
#SBATCH --cpus-per-task=16            # Number of CPU cores per task
#SBATCH --mem=64G                     # Memory per node
#SBATCH --time=72:00:00              # Time limit hrs:min:sec
#SBATCH --partition=gpu               # Partition/queue name
#SBATCH --mail-type=BEGIN,END,FAIL    # Email notification
#SBATCH --mail-user=your.email@domain.com  # Email address

# Load necessary modules (adjust based on your cluster setup)
module purge
module load cuda/11.8
module load anaconda/3

# Activate conda environment
source activate frcnn_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set cache directories for huggingface and torch hub
export HF_HOME=$HOME/.cache/huggingface
export TORCH_HOME=$HOME/.cache/torch

# Set environment variables for distributed training
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $(nvidia-smi -L | wc -l)))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Directory setup
PROJ_DIR=$HOME/faster-rcnn-project  # Adjust to your project directory
DATA_DIR=/path/to/coco/dataset      # Adjust to your data directory
OUTPUT_DIR=$PROJ_DIR/outputs

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the training script
cd $PROJ_DIR

python train.py \
    --train_ann_file $DATA_DIR/annotations/instances_train2017.json \
    --train_img_dir $DATA_DIR/train2017 \
    --val_ann_file $DATA_DIR/annotations/instances_val2017.json \
    --val_img_dir $DATA_DIR/val2017 \
    --experiment_name "faster-rcnn-efficientnet-panet" \
    --output_dir $OUTPUT_DIR \
    --batch_size 16 \
    --num_workers 8 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --max_epochs 100 \
    --num_gpus 4 \
    --use_amp true \
    --gradient_clip_val 0.1 \
    --accumulate_grad_batches 1 \
    --val_check_interval 1.0

# Optional: Copy output files to a backup location
# cp -r $OUTPUT_DIR /path/to/backup/location/

# Deactivate conda environment
conda deactivate