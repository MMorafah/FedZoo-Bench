#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=test
#SBATCH --err=results/test.err
#SBATCH --out=results/test.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for alg in pacfl
do
    dir='../save_results/'$alg'/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi
    
    python ../main.py \
    --ntrials=1 \
    --rounds=20 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=10 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.9 \
    --model=lenet5 \
    --dataset=cifar10 \
    --partition='niid-labelskew' \
    --datadir='../../data/' \
    --logdir='../save_results/' \
    --log_filename='test' \
    --alg=$alg \
    --iid_beta=0.5 \
    --niid_beta=2 \
    --pacfl_beta=14 \
    --noise=0 \
    --gpu=1 \
    --print_freq=10
done 
