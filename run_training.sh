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
# Create logs directory if it doesn't exist
mkdir -p logs

# Set cache directories for huggingface and torch hub
export HF_HOME=/data/inversion/huggingface/
export TORCH_HOME=$HOME/.cache/torch

# Set environment variables for distributed training

# Directory setup
PROJ_DIR=./  # Adjust to your project directory
DATA_DIR=/data/coco      # Adjust to your data directory
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
    --num_gpus 1 \
    --use_amp true \
    --gradient_clip_val 0.1 \
    --accumulate_grad_batches 1 \
    --val_check_interval 1.0

# Optional: Copy output files to a backup location
# cp -r $OUTPUT_DIR /path/to/backup/location/

# Deactivate conda environment
