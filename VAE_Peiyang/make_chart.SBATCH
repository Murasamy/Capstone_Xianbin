#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/xg7/VAE_trader/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python making_chart.py"
