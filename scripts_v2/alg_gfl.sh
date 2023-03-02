#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=alg_gfl_cifar10
#SBATCH --err=results/alg_gfl_cifar10.err
#SBATCH --out=results/alg_gfl_cifar10.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for alg in fedavg
do
    python ../main.py \
    --ntrials=3 \
    --rounds=100 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=10 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.9 \
    --model=lenet5 \
    --dataset=cifar10 \
    --partition='niid-labeldir' \
    --datadir='../../data/' \
    --logdir='../results/' \
    --log_filename='lenet5_E10_C0.1_sgd_0.01' \
    --alg=$alg \
    --iid_beta=0.5 \
    --niid_beta=0.5 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10
done 
