#!/bin/bash
#PBS -N gpu_test
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=00:05:00
#PBS -q gpu
#PBS -o gpu_test_output.txt
#PBS -e gpu_test_error.txt

cd $PBS_O_WORKDIR

module load cuda
nvidia-smi
