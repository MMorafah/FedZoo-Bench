import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.client import *
from src.clustering import *
from src.utils import *

def main_solo(args):

    path = args.path

    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ##################################### Data partitioning section
    print('-'*40)
    print('Getting Clients Data')

    train_ds_global, test_ds_global, train_dl_global, \
    test_dl_global = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0)

    train_ds_global1, test_ds_global1, train_dl_global1, \
    test_dl_global1 = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                         p_train=args.p_train, p_test=args.p_test)

    partitions_train, partitions_test, partitions_train_stat, \
    partitions_test_stat = partition_data(args.dataset, args.datadir, args.partition,
                                          args.num_users, niid_beta=args.niid_beta, iid_beta=args.iid_beta,
                                          p_train=args.p_train, p_test=args.p_test)

    print('-'*40)
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    print(f'MODEL: {args.model}, Dataset: {args.dataset}')
    users_model, net_glob, initial_state_dict = get_models(args, dropout_p=0.5)
    print('-'*40)
    print(net_glob)
    print('')

    total = 0
    for name, param in net_glob.named_parameters():
        print(name, param.size())
        total += np.prod(param.size())
        #print(np.array(param.data.cpu().numpy().reshape([-1])))
        #print(isinstance(param.data.cpu().numpy(), np.array))
    print(f'total params {total}')
    print('-'*40)
    ################################# Fixing all to the same Init and data partitioning and random users
    #print(os.getcwd())

    # tt = '../initialization/' + 'partitions_train_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     partitions_train = pickle.load(f)

    # tt = '../initialization/' + 'partitions_train_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     partitions_train = pickle.load(f)

    # tt = '../initialization/' + 'partitions_train_stat_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     partitions_train_stat = pickle.load(f)

    # tt = '../initialization/' + 'partitions_test_stat_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     partitions_test_stat = pickle.load(f)

    #tt = '../initialization/' + 'init_'+args.model+'_'+args.dataset+'.pth'
    #initial_state_dict = torch.load(tt, map_location=args.device)
    #net_glob.load_state_dict(initial_state_dict)

    #server_state_dict = copy.deepcopy(initial_state_dict)
    #for idx in range(args.num_users):
    #    users_model[idx].load_state_dict(initial_state_dict)

    # tt = '../initialization/' + 'comm_users.pkl'
    # with open(tt, 'rb') as f:
    #     comm_users = pickle.load(f)
    ################################# Initializing Clients
    print('-'*40)
    print('Initializing Clients')
    clients = []
    for idx in range(args.num_users):
        sys.stdout.flush()
        print(f'-- Client {idx}, Train Stat {partitions_train_stat[idx]} Test Stat {partitions_test_stat[idx]}')

        noise_level=0
        dataidxs = partitions_train[idx]
        dataidxs_test = partitions_test[idx]

        train_ds_local = get_subset(train_ds_global, dataidxs)
        test_ds_local  = get_subset(test_ds_global, dataidxs_test)

        transform_train, transform_test = get_transforms(args.dataset, noise_level=0, net_id=None, total=0)

        train_dl_local = DataLoader(dataset=train_ds_local, batch_size=args.local_bs, shuffle=True, drop_last=False)
        test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False)

        clients.append(Client_FedAvg(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                   args.lr, args.momentum, args.device, train_dl_local, test_dl_local))

    print('-'*40)
    ###################################### Federation
    print('Starting SOLO')
    print('-'*40)
    start = time.time()

    clients_local_acc = {i:[] for i in range(args.num_users)}

    for idx in range(args.num_users):
        print(f'Client {idx} is training...')
        sys.stdout.flush()
        for epoch in range(args.rounds):
            loss = clients[idx].train(is_print=False)

            if epoch in [int(0.5*args.rounds), int(0.8*args.rounds)]:
                _, acc = clients[idx].eval_test()
                clients_local_acc[idx].append(acc)

        _, acc = clients[idx].eval_test()
        clients_local_acc[idx].append(acc)

        template = ("Client {:3d}, labels {}, final_acc {:3.3f}, best_acc {:3.3f} \n")
        print(template.format(idx, partitions_train_stat[idx], clients_local_acc[idx][-1], np.max(clients_local_acc[idx])))

    end = time.time()
    duration = end-start
    print('-'*40)
    ############################### Printing Final Test and Train ACC / LOSS
    print('-'*40)
    final_acc = []
    best_acc = []
    for idx in range(args.num_users):
        final_acc.append(clients_local_acc[idx][-1])
        best_acc.append(np.max(clients_local_acc[idx]))

    avg_final_acc = np.mean(final_acc)
    avg_best_acc = np.mean(best_acc)
    print(f'Avg Final Acc: {avg_final_acc:.2f}, Avg Best Acc: {avg_best_acc:.2f}')
    print(f'SOLO Time: {duration/60:.2f} minutes')
    print('-'*40)

    ############################# Fairness
    template = ("-- STD of Local Acc: {:3.2f}")
    f1 = np.std(final_acc)
    print(template.format(f1))

    template = ("-- Top 10% Percentile of Local Acc: {:3.2f}")
    f2 = np.percentile(final_acc, 90)
    print(template.format(f2))

    template = ("-- Bottom 10% Percentile of Local Acc: {:3.2f}")
    f3 = np.percentile(final_acc, 10)
    print(template.format(f3))

    template = ("-- Avg Top 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(final_acc)
    d = int(0.9*args.num_users)
    f4 = np.mean(np.array(final_acc)[argsort[d:]])
    print(template.format(f4))

    template = ("-- Avg Bottom 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(final_acc)
    d = int(0.1*args.num_users)
    f5 = np.mean(np.array(final_acc)[argsort[0:d]])
    print(template.format(f5))

    template = ("-- Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
    f6 = f4 - f5
    print(template.format(f6))
    ###########################

    return avg_final_acc, avg_best_acc, duration, f1, f2, f3, f4, f5, f6

def run_solo(args, fname):
    alg_name = 'SOLO'

    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_fl_time=[]
    exp_f1=[]
    exp_f2=[]
    exp_f3=[]
    exp_f4=[]
    exp_f5=[]
    exp_f6=[]

    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))

        avg_final_local, avg_best_local, duration, f1, f2, f3, f4, f5, f6= main_solo(args)

        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
        exp_fl_time.append(duration/60)
        exp_f1.append(f1)
        exp_f2.append(f2)
        exp_f3.append(f3)
        exp_f4.append(f5)
        exp_f5.append(f4)
        exp_f6.append(f6)

        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

        print(f'-- SOLO Time: {exp_fl_time[-1]:.2f} minutes')

        template = ("-- STD of Local Acc: {:3.2f}")
        print(template.format(exp_f1[-1]))

        template = ("-- Top 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_f2[-1]))

        template = ("-- Bottom 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_f3[-1]))

        template = ("-- Avg Top 10% of Local Acc: {:3.2f}")
        print(template.format(exp_f4[-1]))

        template = ("-- Avg Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_f5[-1]))

        template = ("-- Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_f6[-1]))


    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)

    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

    print(f'-- SOLO Time: {np.mean(exp_fl_time):.2f} minutes')

    template = ("-- STD of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f1), np.std(exp_f1)))

    template = ("-- Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f2), np.std(exp_f2)))

    template = ("-- Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f3), np.std(exp_f3)))

    template = ("-- Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f4), np.std(exp_f4)))

    template = ("-- Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f5), np.std(exp_f5)))

    template = ("-- Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f6), np.std(exp_f6)))

    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)

        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        print(f'-- SOLO Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)

        template = ("-- STD of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f1), np.std(exp_f1)), file=text_file)

        template = ("-- Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f2), np.std(exp_f2)), file=text_file)

        template = ("-- Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f3), np.std(exp_f3)), file=text_file)

        template = ("-- Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f4), np.std(exp_f4)), file=text_file)

        template = ("-- Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f5), np.std(exp_f5)), file=text_file)

        template = ("-- Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f6), np.std(exp_f6)), file=text_file)
        print('*'*40)

    return
