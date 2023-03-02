#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=fedavg_cifar10_comm_niid_labeldir
#SBATCH --err=results/fedavg_cifar10_comm_niid_labeldir.err
#SBATCH --out=results/fedavg_cifar10_comm_niid_labeldir.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for epoch in 20
do
    for round in 500 250
    do
        dir='../save_results/fedavg/cifar10'
        if [ ! -e $dir ]; then
        mkdir -p $dir
        fi

        python ../main.py \
        --ntrials=3 \
        --rounds=$round \
        --num_users=100 \
        --frac=0.1 \
        --local_ep=$epoch \
        --local_bs=10 \
        --lr=0.01 \
        --momentum=0.9 \
        --model=lenet5 \
        --dataset=cifar10 \
        --partition='niid-labelskew' \
        --datadir='../../data/' \
        --logdir='../results_comm/' \
        --log_filename='epoch_'$epoch'_comm_'$round \
        --alg='fedavg' \
        --iid_beta=0.5 \
        --niid_beta=3 \
        --local_view \
        --noise=0 \
        --gpu=0 \
        --print_freq=10
    done
done
