#!/bin/bash
#SBATCH -p batch_sw_grad 
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time=7-00:00:0
#SBATCH --gres=gpu:1