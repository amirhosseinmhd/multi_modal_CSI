#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=4:30:0
#SBATCH --mail-user=mdi.amirhossein@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=1
#SBATCH --job-name=csi_job

# Define important directories
PROJECT_DIR=/home/amirmhd/projects/def-hinat/amirmhd/multi_modal_CSI
CODE_DIR=benchmark/wifi_csi

# Create directory structure in SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/$CODE_DIR

# Copy code to temporary directory for faster execution
echo "Copying code to temporary directory..."
cp -r $PROJECT_DIR/$CODE_DIR/* $SLURM_TMPDIR/$CODE_DIR/
cd $SLURM_TMPDIR/$CODE_DIR

module purge
module load python/3.11.5 scipy-stack
source ~/py311/bin/activate

echo "Starting job at $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

python run_main.py

echo "Job finished at $(date)"