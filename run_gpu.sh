#!/bin/bash
#PBS -N game_of_life_mpi_job
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -l mem=4gb
#PBS -q class
#PBS -o /scratch/ualclsd0197/output_dir/game_of_life_output.txt
#PBS -e /scratch/ualclsd0197/output_dir/game_of_life_error.txt

cd $PBS_O_WORKDIR

module load cuda
nvcc -o gameoflife gameoflife.cu

./gameoflife 100 100

