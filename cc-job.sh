#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=4:30:0
#SBATCH --mail-user=mdi.amirhossein@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=1
#SBATCH --job-name=csi_job
#export $OUTFILE_NAME=timeStream-%j.out

#SBATCH --output=$OUTFILE_NAME

# Define important directories
PROJECT_DIR=/home/amirmhd/projects/def-hinat/amirmhd/multi_modal_CSI
CODE_DIR=benchmark/wifi_csi
DATA_DIR=/home/amirmhd/projects/def-hinat/amirmhd/multi_modal_CSI/dataset

# Create directory structure in SLURM_TMPDIR
echo "Copying code and data to temporary directory..."
mkdir -p $SLURM_TMPDIR/$CODE_DIR
cp -r $PROJECT_DIR/$DATA_DIR/* $SLURM_TMPDIR/$DATA_DIR/

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


# Update your data paths in the Python script to use SLURM_TMPDIR
export DATA_PATH=DATA_DIR
export NUM_DECODER_LAYERS=1
export NUM_QUERIES=8

# Now we change the preset accordingly
python config_modifier.py preset.py modified_preset.py


python run_main.py


if [ -d "results" ]; then
    echo "Copying results back to project directory..."
    mkdir $PROJECT_DIR/$OUTFILE_NAME
    cp -r results $PROJECT_DIR/$OUTFILE_NAME
fi

echo "Job finished at $(date)"