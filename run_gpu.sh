#!/bin/bash
#PBS -N gpu_query
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=00:05:00
#PBS -q classgpu
#PBS -o gpu_info.txt
#PBS -e gpu_info_error.txt

module load cuda
nvidia-smi
