#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=10000
#SBATCH --mail-type=end
#SBATCH --mail-user=a.ranganarsimhaiah@tu-braunschweig.de
#SBATCH --exclude=gpu[06,05]

module load cuda/10.0
module load lib/cudnn/7.6.1.34_cuda_10.0
module load anaconda/3-5.0.1

source activate tf-200-GPU-py37

srun python -u Load_hdfgen.py