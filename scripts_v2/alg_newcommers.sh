#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=alg_newcomers_cifar10
#SBATCH --err=results/alg_newcomers_cifar10.err
#SBATCH --out=results/alg_newcomers_cifar10.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for alg in apfl
do
    python ../main.py \
    --ntrials=3 \
    --rounds=100 \
    --num_users=20 \
    --frac=0.2 \
    --local_ep=10 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.9 \
    --model=resnet9 \
    --dataset=cifar100 \
    --partition='niid-labeldir' \
    --datadir='../../data/' \
    --logdir='../results_newcomers_comp/' \
    --log_filename='resnet9_E10_C0.2_sgd_0.01' \
    --alg=$alg \
    --iid_beta=0.5 \
    --niid_beta=0.1 \
    --new_comer \
    --ft_epoch=15 \
    --pruning_percent_subfedavg=5 \
    --pruning_target_subfedavg=50 \
    --dist_thresh_subfedavg=0.0001 \
    --acc_thresh_subfedavg=55 \
    --gpu=0 \
    --print_freq=10
done 
