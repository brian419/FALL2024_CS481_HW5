#!/bin/bash
#PBS -N game_of_life_gpu_job
#PBS -l select=1:ngpus=1:ncpus=1:mpiprocs=1:mem=4gb
#PBS -l walltime=08:00:00
#PBS -q classgpu
#PBS -o /scratch/ualclsd0197/gameoflife_output.txt  
#PBS -e /scratch/ualclsd0197/gameoflife_error.txt  

cd $PBS_O_WORKDIR

module load cuda
# nvcc -o gameoflife gameoflife.cu
# ./gameoflife 10000 5000 /scratch/ualclsd0197/output_dir

nvcc -o gameoflifeoptimized gameoflifeoptimized.cu
./gameoflifeoptimized 5000 5000 /scratch/ualclsd0197/output_dir