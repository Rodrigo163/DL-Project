#!/bin/bash -l

# queue
#SBATCH --partition=gpu2

# jobname
#SBATCH --job-name=IM_PL_gpu2


#SBATCH --nodes=16
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2

# job time for backfill scheduling
#SBATCH --time=1-0

# User notify events
#SBATCH --mail-type=all
#SBATCH --mail-user=Rodrigo.Lopez@mpikg.mpg.de

#SBATCH --signal=SIGUSR1@180

module purge
module add python/3.8.2

# might need  the latest cuda
module load nvidia/cuda/9.1

set
# run script from above
srun ipython lightning_train.py
