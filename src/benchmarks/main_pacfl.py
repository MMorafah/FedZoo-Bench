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

torch.backends.cudnn.benchmark = True

def main_pacfl(args):

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
    #initial_state_dict = nn.DataParallel(initial_state_dict)
    #net_glob = nn.DataParallel(net_glob)
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

    traindata_cls_ratio = {}

    budget = 20
    for i in range(args.num_users):
        total_sum = sum(list(partitions_train_stat[i].values()))
        base = 1/len(list(partitions_train_stat[i].values()))
        temp_ratio = {}
        for k in partitions_train_stat[i].keys():
            ss = partitions_train_stat[i][k]/total_sum
            temp_ratio[k] = (partitions_train_stat[i][k]/total_sum)
            if ss >= (base + 0.05):
                temp_ratio[k] = partitions_train_stat[i][k]

        sub_sum = sum(list(temp_ratio.values()))
        for k in temp_ratio.keys():
            temp_ratio[k] = (temp_ratio[k]/sub_sum)*budget

        round_ratio = round_to(list(temp_ratio.values()), budget)
        cnt = 0
        for k in temp_ratio.keys():
            temp_ratio[k] = round_ratio[cnt]
            cnt+=1

    traindata_cls_ratio[i] = temp_ratio

    clients = []
    U_clients = []
    K = args.pacfl_n_basis

    for idx in range(args.num_users):
        sys.stdout.flush()
        print(f'-- Client {idx}, Train Stat {partitions_train_stat[idx]} Test Stat {partitions_test_stat[idx]}')

        noise_level=0
        dataidxs = partitions_train[idx]
        dataidxs_test = partitions_test[idx]

        train_ds_local = get_subset(train_ds_global, dataidxs)
        test_ds_local  = get_subset(test_ds_global, dataidxs_test)

        transform_train, transform_test = get_transforms(args.dataset, noise_level=0, net_id=None, total=0)

        train_dl_local = DataLoader(dataset=train_ds_local, batch_size=args.local_bs, shuffle=True, drop_last=False,
                                   num_workers=4, pin_memory=False)
        test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False, num_workers=4,
                                  pin_memory=False)

        idxs_local = np.arange(len(train_ds_local.data))
        labels_local = np.array(train_ds_local.target)
        # Sort Labels Train
        idxs_labels_local = np.vstack((idxs_local, labels_local))
        idxs_labels_local = idxs_labels_local[:, idxs_labels_local[1, :].argsort()]
        idxs_local = idxs_labels_local[0, :]
        labels_local = idxs_labels_local[1, :]

        uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)

        #print(f'Labels: {uni_labels}, Counts: {cnt_labels}')

        nlabels = len(uni_labels)
        cnt = 0
        U_temp = []
        for j in range(nlabels):
            local_ds1 = train_ds_local.data[idxs_local[cnt:cnt+cnt_labels[j]]]
#             if local_ds1.shape[2]==local_ds1.shape[3]:
#                 local_ds1 = np.transpose(local_ds1, (0,2,3,1))

            local_ds1 = local_ds1.reshape(cnt_labels[j], -1)
            local_ds1 = local_ds1.T
            if type(train_ds_local.target[idxs_local[cnt:cnt+cnt_labels[j]]]) == torch.Tensor:
                label1 = list(set(train_ds_local.target[idxs_local[cnt:cnt+cnt_labels[j]]].numpy()))
            else:
                label1 = list(set(train_ds_local.target[idxs_local[cnt:cnt+cnt_labels[j]]]))
            assert len(label1) == 1

            print(f'Label {j} : {label1}')

            if args.partition == 'noniid-labeldir':
                #print('Dir partition')
                if label1 in list(traindata_cls_ratio[idx].keys()):
                    K = traindata_cls_ratio[idx][label1[0]]
                else:
                    K = args.n_basis
            if K > 0:
                u1_temp, sh1_temp, vh1_temp = np.linalg.svd(local_ds1, full_matrices=False)
                u1_temp=u1_temp/np.linalg.norm(u1_temp, ord=2, axis=0)
                U_temp.append(u1_temp[:, 0:K])

            cnt+=cnt_labels[j]

        #U_temp = [u1_temp[:, 0:K], u2_temp[:, 0:K]]
        U_clients.append(copy.deepcopy(np.hstack(U_temp)))

        clients.append(Client_FedAvg(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                   args.lr, args.momentum, args.device, train_dl_local, test_dl_local))

    print('-'*40)
    ###################################### Clustering
    np.set_printoptions(precision=2)
    #m = max(int(args.frac * args.num_users), 1)
    #clients_idxs = np.random.choice(range(args.num_users), m, replace=False)

    cnt = args.num_users
    for r in range(1):
        print(f'Round {r}')
        clients_idxs = np.arange(cnt)
        #clients_idxs = np.arange(10)
        for idx in clients_idxs:
            print(f'Client {idx}, Labels: {partitions_train_stat[idx]}')

        adj_mat = calculating_adjacency(clients_idxs, U_clients)
        clusters = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=args.pacfl_beta, linkage=args.pacfl_linkage)

#         flag = True
#         if len(clusters) > 1 and len(clusters) < 4:
#             flag=False
        
        th = 100
        for kk in range(2000):
            th/=1.12
            clusters = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=th, linkage=args.pacfl_linkage)
            if len(clusters) <= 3:
                break

        #cnt+= 10
        print(f'Threshold {th}')
        print('')
        print('Adjacency Matrix')
        print(adj_mat)
        print('')
        print('Clusters: ')
        print(clusters)
        print('')
        print(f'Number of Clusters {len(clusters)}')
        print('')
        for jj in range(len(clusters)):
            print(f'Cluster {jj}: {len(clusters[jj])} Users')


    labels_clust = []
    for j in range(len(clusters)):
        tmp=[]
        for c in clusters[j]:
            tmp.extend(list(partitions_train_stat[c].keys()))
        uni_labels, cnt_labels = np.unique(tmp, return_counts=True)
        labels_clust.append(tmp)
    print(labels_clust)

    clients_clust_id = {i:None for i in range(args.num_users)}
    for i in range(args.num_users):
        for j in range(len(clusters)):
            if i in clusters[j]:
                clients_clust_id[i] = j
                break
    print(f'Clients: Cluster_ID \n{clients_clust_id}')

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
    w_glob_per_cluster = [copy.deepcopy(initial_state_dict) for _ in range(len(clusters))]
    best_glob_acc = [0 for _ in range(len(clusters))]

    m = max(int(args.frac * num_users_FL), 1)

    for iteration in range(args.rounds):

        idxs_users = np.random.choice(range(num_users_FL), m, replace=False)
        #idxs_users = comm_users[iteration]

        print(f'----- ROUND {iteration+1} -----')

        sys.stdout.flush()

        idx_clusters_round = {}
        for idx in idxs_users:
            idx_cluster = clients_clust_id[idx]
            idx_clusters_round[idx_cluster] = []

        for idx in idxs_users:
            idx_cluster = clients_clust_id[idx]
            idx_clusters_round[idx_cluster].append(idx)

            clients[idx].set_state_dict(copy.deepcopy(w_glob_per_cluster[idx_cluster]))

            loss = clients[idx].train(is_print=False)
            loss_locals.append(copy.deepcopy(loss))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.3f}'
        print(template.format(loss_avg))

        ####### FedAvg ####### START
        total_data_points = {}
        for k in idx_clusters_round.keys():
            temp_sum = []
            for r in idx_clusters_round[k]:
                temp_sum.append(len(partitions_train[r]))

            total_data_points[k] = sum(temp_sum)

        fed_avg_freqs = {}
        for k in idx_clusters_round.keys():
            fed_avg_freqs[k] = []
            for r in idx_clusters_round[k]:
                ratio = len(partitions_train[r]) / total_data_points[k]
                fed_avg_freqs[k].append(copy.deepcopy(ratio))

        for k in idx_clusters_round.keys():
            w_locals = []
            for el in idx_clusters_round[k]:
                w_locals.append(copy.deepcopy(clients[el].get_state_dict()))

            ww = AvgWeights(w_locals, weight_avg=fed_avg_freqs[k])
            w_glob_per_cluster[k] = copy.deepcopy(ww)
            net_glob.load_state_dict(copy.deepcopy(ww))
            _, acc = eval_test(net_glob, args, test_dl_global1)
            if acc > best_glob_acc[k]:
                best_glob_acc[k] = acc

        ####### FedAvg ####### END

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
    ############################### FedAvg Final Results
    print('-'*40)
    print('FINAL RESULTS')
    for jj in range(len(clusters)):
        print(f'Cluster {jj}, Best Glob Acc {best_glob_acc[jj]:.3f}')

    print(f'Average Best Glob Acc {np.mean(best_glob_acc[0:len(clusters)]):.3f}')

    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))

    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))

    print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
    ############################### FedAvg+ (FedAvg + FineTuning)
    print('-'*40)
    print('PACFL+ ::: PACFL + Local FineTuning')
    sys.stdout.flush()

    local_acc = []
    for idx in range(num_users_FL):
        idx_cluster = clients_clust_id[idx]
        clients[idx].set_state_dict(copy.deepcopy(w_glob_per_cluster[idx_cluster]))

        loss = clients[idx].train(is_print=False)
        _, acc = clients[idx].eval_test()
        local_acc.append(acc)

    pacfl_ft_local = np.mean(local_acc)
    print(f'-- PACFL+ :: AVG Local Acc: {np.mean(local_acc):.2f}')
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
    template = ("-- PACFL+: STD of Local Acc: {:3.2f}")
    ff1 = np.std(local_acc)
    print(template.format(ff1))

    template = ("-- PACFL+: Top 10% Percentile of Local Acc: {:3.2f}")
    ff2 = np.percentile(local_acc, 90)
    print(template.format(ff2))

    template = ("-- PACFL+: Bottom 10% Percentile of Local Acc: {:3.2f}")
    ff3 = np.percentile(local_acc, 10)
    print(template.format(ff3))

    template = ("-- PACFL+: Avg Top 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(local_acc)
    d = int(0.9*num_users_FL)
    ff4 = np.mean(np.array(local_acc)[argsort[d:]])
    print(template.format(ff4))

    template = ("-- PACFL+: Avg Bottom 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(local_acc)
    d = int(0.1*num_users_FL)
    ff5 = np.mean(np.array(local_acc)[argsort[0:d]])
    print(template.format(ff5))

    template = ("-- PACFL+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
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
            idx_cluster = clients_clust_id[idx]
            clients[idx].set_state_dict(copy.deepcopy(w_glob_per_cluster[idx_cluster]))
            clients[idx].local_ep=1
            _, acc = clients[idx].eval_test()
            new_comer_acc[idx].append(acc)
            print(f'Client {idx:3d}, current_acc {acc:3.2f}, best_acc {np.max(new_comer_acc[idx]):3.2f}')
        new_comer_avg_acc.append(np.mean([acc[-1] for acc in new_comer_acc.values()]))
        print(f'-- New Comers Initial AVG Acc: {new_comer_avg_acc[-1]:3.2f}')

        for iteration in range(args.ft_epoch):
            for idx in range(num_users_FL, args.num_users):
                loss = clients[idx].train(is_print=False)
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

    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)

    return (avg_final_local, avg_best_local, pacfl_ft_local, duration,
           f1, f2, f3, f4, f5, f6, ff1, ff2, ff3, ff4, ff5, ff6, ff7)

def run_pacfl(args, fname):
    alg_name = 'PACFL'

    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_pacfl_ft_local=[]
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

        avg_final_local, avg_best_local, \
        pacfl_ft_local, duration, f1, f2, f3, f4, f5, f6, ff1, ff2, ff3, ff4, ff5, ff6, ff7 = main_pacfl(args)

        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
        exp_pacfl_ft_local.append(pacfl_ft_local)
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

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

        print(f'-- PACFL+ Fine Tuning Clients AVG Local Acc: {exp_pacfl_ft_local[-1]:.2f}')
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

        template = ("-- PACFL+: STD of Local Acc: {:3.2f}")
        print(template.format(exp_ff1[-1]))

        template = ("-- PACFL+: Top 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_ff2[-1]))

        template = ("-- PACFL+: Bottom 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_ff3[-1]))

        template = ("-- PACFL+: Avg Top 10% of Local Acc: {:3.2f}")
        print(template.format(exp_ff4[-1]))

        template = ("-- PACFL+: Avg Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_ff5[-1]))

        template = ("-- PACFL+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_ff6[-1]))

    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)

    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

    template = '-- PACFL+ Fine Tuning Clients AVG Local Acc: {:.2f} +- {:.2f}'
    print(template.format(np.mean(exp_pacfl_ft_local), np.std(exp_pacfl_ft_local)))

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

    template = ("-- PACFL+: STD of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff1), np.std(exp_ff1)))

    template = ("-- PACFL+: Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff2), np.std(exp_ff2)))

    template = ("-- PACFL+: Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff3), np.std(exp_ff3)))

    template = ("-- PACFL+: Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff4), np.std(exp_ff4)))

    template = ("-- PACFL+: Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff5), np.std(exp_ff5)))

    template = ("-- PACFL+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff6), np.std(exp_ff6)))

    if args.new_comer:
        template = ("-- New Comers AVG Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff7), np.std(exp_ff7)))

    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)

        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        template = '-- PACFL+ Fine Tuning Clients AVG Local Acc: {:.2f} +- {:.2f}'
        print(template.format(np.mean(exp_pacfl_ft_local), np.std(exp_pacfl_ft_local)), file=text_file)

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

        template = ("-- PACFL+: STD of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff1), np.std(exp_ff1)), file=text_file)

        template = ("-- FPACFL+: Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff2), np.std(exp_ff2)), file=text_file)

        template = ("-- PACFL+: Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff3), np.std(exp_ff3)), file=text_file)

        template = ("-- PACFL+: Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff4), np.std(exp_ff4)), file=text_file)

        template = ("-- PACFL+: Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff5), np.std(exp_ff5)), file=text_file)

        template = ("-- PACFL+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff6), np.std(exp_ff6)), file=text_file)

        if args.new_comer:
            template = ("-- New Comers AVG Acc: {:3.2f} +- {:.2f}")
            print(template.format(np.mean(exp_ff7), np.std(exp_ff7)), file=text_file)

        print('*'*40)

    return
