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

def main_fedrep(args):

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
    global_keys = list(net_glob.state_dict().keys())[:-4]
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

        clients.append(Client_FedRep(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                   args.lr, args.momentum, args.device, train_dl_local, test_dl_local))

    print('-'*40)
    ###################################### Federation
    print('Starting FL')
    print('-'*40)
    start = time.time()

    if args.new_comer:
        num_users_FL = args.num_users * 4 // 5
        num_users_NC = args.num_users - num_users_FL
    else:
        num_users_FL = args.num_users

    loss_train = []
    clients_local_acc = {i:[] for i in range(num_users_FL)}
    w_locals, loss_locals = [], []
    glob_acc = []

    w_glob = copy.deepcopy(initial_state_dict)
    global_keys = list(w_glob.keys())[:-args.glob_layers]

    m = max(int(args.frac * num_users_FL), 1)

    for iteration in range(args.rounds):

        idxs_users = np.random.choice(range(num_users_FL), m, replace=False)
        #idxs_users = comm_users[iteration]

        print(f'----- ROUND {iteration+1} -----')
        sys.stdout.flush()
        for idx in idxs_users:
            w_client = copy.deepcopy(clients[idx].get_state_dict())
            for k in global_keys:
                w_client[k] = w_glob[k]
            clients[idx].set_state_dict(copy.deepcopy(w_client))

            loss = clients[idx].train(global_keys, args.local_rep_ep, is_print=False)
            loss_locals.append(copy.deepcopy(loss))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.3f}'
        print(template.format(loss_avg))

        ####### FedRep ####### START
        total_data_points = sum([len(partitions_train[r]) for r in idxs_users])
        fed_avg_freqs = [len(partitions_train[r]) / total_data_points for r in idxs_users]
        w_locals = []
        for idx in idxs_users:
            w_locals.append(copy.deepcopy(clients[idx].get_state_dict()))

        ww = AvgWeights(w_locals, weight_avg=fed_avg_freqs)
        for key, val in ww.items():
            if key not in global_keys:
                val = copy.deepcopy(w_glob[key])

        w_glob = copy.deepcopy(ww)
        net_glob.load_state_dict(copy.deepcopy(ww))

        ####### FedRep ####### END
        _, acc = eval_test(net_glob, args, test_dl_global1)

        glob_acc.append(acc)
        template = "-- Global Acc: {:.3f}, Global Best Acc: {:.3f}"
        print(template.format(glob_acc[-1], np.max(glob_acc)))

        print_flag = False
        if iteration+1 in [int(0.10*args.rounds), int(0.25*args.rounds), int(0.5*args.rounds), int(0.8*args.rounds)]:
            print_flag = True

        if print_flag:
            print('*'*25)
            print(f'Check Point @ Round {iteration+1} --------- {int((iteration+1)/args.rounds*100)}% Completed')
            temp_acc = []
            temp_best_acc = []
            for k in range(num_users_FL):
                sys.stdout.flush()
                loss, acc = clients[k].eval_test()
                clients_local_acc[k].append(acc)
                temp_acc.append(clients_local_acc[k][-1])
                temp_best_acc.append(np.max(clients_local_acc[k]))

                template = ("Client {:3d}, current_acc {:3.2f}, best_acc {:3.2f}")
                print(template.format(k, clients_local_acc[k][-1], np.max(clients_local_acc[k])))

            #print('*'*25)
            template = ("-- Avg Local Acc: {:3.2f}")
            print(template.format(np.mean(temp_acc)))
            template = ("-- Avg Best Local Acc: {:3.2f}")
            print(template.format(np.mean(temp_best_acc)))
            print('*'*25)

        loss_train.append(loss_avg)

        ## clear the placeholders for the next round
        loss_locals.clear()

        ## calling garbage collector
        gc.collect()

    end = time.time()
    duration = end-start
    print('-'*40)
    ############################### Testing Local Results
    print('*'*25)
    print('---- Testing Final Local Results ----')
    temp_acc = []
    temp_best_acc = []
    for k in range(num_users_FL):
        sys.stdout.flush()
        loss, acc = clients[k].eval_test()
        clients_local_acc[k].append(acc)
        temp_acc.append(clients_local_acc[k][-1])
        temp_best_acc.append(np.max(clients_local_acc[k]))

        template = ("Client {:3d}, Final_acc {:3.2f}, best_acc {:3.2f} \n")
        print(template.format(k, clients_local_acc[k][-1], np.max(clients_local_acc[k])))

    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))
    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))
    print('*'*25)
    ############################### FedRep Final Results
    print('-'*40)
    print('FINAL RESULTS')
    template = "-- Global Acc Final: {:.2f}"
    print(template.format(glob_acc[-1]))

    template = "-- Global Acc Avg Final [N*C] Rounds: {:.2f}"
    print(template.format(np.mean(glob_acc[-m:])))

    template = "-- Global Best Acc: {:.2f}"
    print(template.format(np.max(glob_acc)))

    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))

    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))

    print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
    ############################### FedRep+ (FedRep + FineTuning)
    print('-'*40)
    print('FedRep+ ::: FedRep + Local FineTuning')
    sys.stdout.flush()

    local_acc = []
    for idx in range(num_users_FL):
        w_client = copy.deepcopy(clients[idx].get_state_dict())
        for k in global_keys:
            w_client[k] = w_glob[k]
        clients[idx].set_state_dict(copy.deepcopy(w_client))
        loss = clients[idx].train(global_keys, args.local_rep_ep, is_print=False)
        _, acc = clients[idx].eval_test()
        local_acc.append(acc)

    fedrep_ft_local = np.mean(local_acc)
    print(f'-- FedRep+ :: AVG Local Acc: {np.mean(local_acc):.2f}')
    ############################# Saving Print Results

    ############################# Fairness
    template = ("-- STD of Local Acc: {:3.2f}")
    f1 = np.std(temp_acc)
    print(template.format(f1))

    template = ("-- Top 10% Percentile of Local Acc: {:3.2f}")
    f2 = np.percentile(temp_acc, 90)
    print(template.format(f2))

    template = ("-- Bottom 10% Percentile of Local Acc: {:3.2f}")
    f3 = np.percentile(temp_acc, 10)
    print(template.format(f3))

    template = ("-- Avg Top 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(temp_acc)
    d = int(0.9*num_users_FL)
    f4 = np.mean(np.array(temp_acc)[argsort[d:]])
    print(template.format(f4))

    template = ("-- Avg Bottom 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(temp_acc)
    d = int(0.1*num_users_FL)
    f5 = np.mean(np.array(temp_acc)[argsort[0:d]])
    print(template.format(f5))

    template = ("-- Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
    f6 = f4 - f5
    print(template.format(f6))
    ###########################

    ############################# Fairness
    template = ("-- FedRep+: STD of Local Acc: {:3.2f}")
    ff1 = np.std(local_acc)
    print(template.format(ff1))

    template = ("-- FedRep+: Top 10% Percentile of Local Acc: {:3.2f}")
    ff2 = np.percentile(local_acc, 90)
    print(template.format(ff2))

    template = ("-- FedRep+: Bottom 10% Percentile of Local Acc: {:3.2f}")
    ff3 = np.percentile(local_acc, 10)
    print(template.format(ff3))

    template = ("-- FedRep+: Avg Top 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(local_acc)
    d = int(0.9*num_users_FL)
    ff4 = np.mean(np.array(local_acc)[argsort[d:]])
    print(template.format(ff4))

    template = ("-- FedRep+: Avg Bottom 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(local_acc)
    d = int(0.1*num_users_FL)
    ff5 = np.mean(np.array(local_acc)[argsort[0:d]])
    print(template.format(ff5))

    template = ("-- FedRep+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
    ff6 = ff4 - ff5
    print(template.format(ff6))
    ###########################

    ############################### New Comers Start
    if args.new_comer:
        print('-'*40)
        print('Evaluating new comers')
        sys.stdout.flush()

        new_comer_avg_acc = []
        new_comer_acc = {i:[] for i in range(num_users_FL, args.num_users)}
        for idx in range(num_users_FL, args.num_users):
            clients[idx].set_state_dict(copy.deepcopy(w_glob))
            clients[idx].local_ep=1
            _, acc = clients[idx].eval_test()
            new_comer_acc[idx].append(acc)
            print(f'Client {idx:3d}, current_acc {acc:3.2f}, best_acc {np.max(new_comer_acc[idx]):3.2f}')
        new_comer_avg_acc.append(np.mean([acc[-1] for acc in new_comer_acc.values()]))
        print(f'-- New Comers Initial AVG Acc: {new_comer_avg_acc[-1]:3.2f}')

        for iteration in range(args.ft_epoch):
            for idx in range(num_users_FL, args.num_users):
                loss = clients[idx].train(global_keys, args.local_rep_ep, is_print=False)
                _, acc = clients[idx].eval_test()
                new_comer_acc[idx].append(acc)
            new_comer_avg_acc.append(np.mean([acc[-1] for acc in new_comer_acc.values()]))

            if iteration%5==0:
                print(f'-- Finetune Round: {iteration + 1}')
                for idx in range(num_users_FL, args.num_users):
                    print(f'Client {idx:3d}, current_acc {new_comer_acc[idx][-1]:3.2f}, best_acc {np.max(new_comer_acc[idx]):3.2f}')
                print(f'-- New Comers AVG Acc: {new_comer_avg_acc[-1]:3.2f}')

        print(f'-- Finetune Finished')
        print(f'-- New Comers Final AVG Acc: {new_comer_avg_acc[-1]:3.2f}')
        print(f'-- New Comers Final Best Acc: {np.max(new_comer_avg_acc):3.2f}')
        ff7 = new_comer_avg_acc[-1]
    else:
        ff7 = None

    ############################# New Comers End

    final_glob = glob_acc[-1]
    avg_final_glob = np.mean(glob_acc[-m:])
    best_glob = np.max(glob_acc)
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)

    return (final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, fedrep_ft_local, duration,
           f1, f2, f3, f4, f5, f6, ff1, ff2, ff3, ff4, ff5, ff6, ff7)

def run_fedrep(args, fname):
    alg_name = 'FedRep'

    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_fedrep_ft_local=[]
    exp_fl_time=[]
    exp_f1=[]
    exp_f2=[]
    exp_f3=[]
    exp_f4=[]
    exp_f5=[]
    exp_f6=[]
    exp_ff1=[]
    exp_ff2=[]
    exp_ff3=[]
    exp_ff4=[]
    exp_ff5=[]
    exp_ff6=[]
    exp_ff7=[]

    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))

        final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, \
        fedrep_ft_local, duration, f1, f2, f3, f4, f5, f6, ff1, ff2, ff3, ff4, ff5, ff6, ff7 = main_fedrep(args)

        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
        exp_fedrep_ft_local.append(fedrep_ft_local)
        exp_fl_time.append(duration/60)
        exp_f1.append(f1)
        exp_f2.append(f2)
        exp_f3.append(f3)
        exp_f4.append(f5)
        exp_f5.append(f4)
        exp_f6.append(f6)
        exp_ff1.append(ff1)
        exp_ff2.append(ff2)
        exp_ff3.append(ff3)
        exp_ff4.append(ff4)
        exp_ff5.append(ff5)
        exp_ff6.append(ff6)
        exp_ff7.append(ff7)

        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')

        template = "-- Global Final Acc: {:.2f}"
        print(template.format(exp_final_glob[-1]))

        template = "-- Global Avg Final [N*C] Rounds Acc: {:.2f}"
        print(template.format(exp_avg_final_glob[-1]))

        template = "-- Global Best Acc: {:.2f}"
        print(template.format(exp_best_glob[-1]))

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

        print(f'-- FedRep Fine Tuning Clients AVG Local Acc: {exp_fedrep_ft_local[-1]:.2f}')
        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')

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

        template = ("-- FedRep+: STD of Local Acc: {:3.2f}")
        print(template.format(exp_ff1[-1]))

        template = ("-- FedRep+: Top 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_ff2[-1]))

        template = ("-- FedRep+: Bottom 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_ff3[-1]))

        template = ("-- FedRep+: Avg Top 10% of Local Acc: {:3.2f}")
        print(template.format(exp_ff4[-1]))

        template = ("-- FedRep+: Avg Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_ff5[-1]))

        template = ("-- FedRep+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_ff6[-1]))

    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)

    template = "-- Global Final Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_final_glob), np.std(exp_final_glob)))

    template = "-- Global Avg Final [N*C] Rounds Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_avg_final_glob), np.std(exp_avg_final_glob)))

    template = "-- Global Best Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_best_glob), np.std(exp_best_glob)))

    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

    template = '-- FedRep Fine Tuning Clients AVG Local Acc: {:3.2f} +- {:.2f}'
    print(template.format(np.mean(exp_fedrep_ft_local), np.std(exp_fedrep_ft_local)))

    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')

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

    template = ("-- FedRep+: STD of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff1), np.std(exp_ff1)))

    template = ("-- FedRep+: Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff2), np.std(exp_ff2)))

    template = ("-- FedRep+: Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff3), np.std(exp_ff3)))

    template = ("-- FedRep+: Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff4), np.std(exp_ff4)))

    template = ("-- FedRep+: Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff5), np.std(exp_ff5)))

    template = ("-- FedRep+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff6), np.std(exp_ff6)))

    if args.new_comer:
        template = ("-- New Comers AVG Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff7), np.std(exp_ff7)))

    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)

        template = "-- Global Final Acc: {:.2f} +- {:.2f}"
        print(template.format(np.mean(exp_final_glob), np.std(exp_final_glob)), file=text_file)

        template = "-- Global Avg Final [N*C] Rounds Acc: {:.2f} +- {:.2f}"
        print(template.format(np.mean(exp_avg_final_glob), np.std(exp_avg_final_glob)), file=text_file)

        template = "-- Global Best Acc: {:.2f} +- {:.2f}"
        print(template.format(np.mean(exp_best_glob), np.std(exp_best_glob)), file=text_file)

        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        template = '-- FedRep Fine Tuning Clients AVG Local Acc: {:3.2f} +- {:.2f}'
        print(template.format(np.mean(exp_fedrep_ft_local), np.std(exp_fedrep_ft_local)), file=text_file)

        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)

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

        template = ("-- FedRep+: STD of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff1), np.std(exp_ff1)), file=text_file)

        template = ("-- FedRep+: Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff2), np.std(exp_ff2)), file=text_file)

        template = ("-- FedRep+: Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff3), np.std(exp_ff3)), file=text_file)

        template = ("-- FedRep+: Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff4), np.std(exp_ff4)), file=text_file)

        template = ("-- FedRep+: Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff5), np.std(exp_ff5)), file=text_file)

        template = ("-- FedRep+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff6), np.std(exp_ff6)), file=text_file)

        if args.new_comer:
            template = ("-- New Comers AVG Acc: {:3.2f} +- {:.2f}")
            print(template.format(np.mean(exp_ff7), np.std(exp_ff7)), file=text_file)

        print('*'*40)

    return
