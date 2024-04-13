#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** MPI Job in Normal Queue ***
# 
# Last revised: October 22, 2021
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch milan.mpi.slurm" on a Lonestar6 login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do NOT use mpirun or mpiexec.
#
#   -- Max recommended MPI ranks per Milan node: 128
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J dropout_ILT     # Job name
#SBATCH -o dropout_ILT.o%j       # Name of stdout output file
#SBATCH -e dropout_ILT.e%j       # Name of stderr error file
#SBATCH -p gpu-a100           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 3               # Total # of mpi tasks
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A MLL             # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=jiahui.k.chen@gmail.com

source ~/.bashrc
conda activate diffusion

cd /home1/09842/jc98685/LT-classification/ 

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo "MASTER_ADDR="$MASTER_ADDR

# one node DDP OOMs, need to add gradient checkpointing
#torchrun --nnodes=1 --nproc_per_node=3 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 main.py --cfg ./config/ImageNet_LT/full_synth_data.yaml

# DataParallel training
python main.py --cfg ./config/ImageNet_LT/full_synth_data.yaml
