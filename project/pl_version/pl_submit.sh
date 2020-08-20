#!/bin/bash -l

# queue
#SBATCH --partition=gpu2

# jobname
#SBATCH --job-name=IM_PL


#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4

# job time for backfill scheduling
#SBATCH --time=3-0

# User notify events
#SBATCH --mail-type=all
#SBATCH --mail-user=Rodrigo.Lopez@mpikg.mpg.de


module purge
module add python/3.8.2

# might need  the latest cuda
module load nvidia/cuda/9.1

set
# run script from above
srun ipython train.py
