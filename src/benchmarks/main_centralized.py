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

def main_centralized(args):

    path = args.path

    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ##################################### Data partitioning section
    print('-'*40)
    print('Getting Clients Data')

    train_ds_global, test_ds_global, train_dl_global, \
    test_dl_global = get_dataset_global(args.dataset, args.datadir, batch_size=128,
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
    ###################################### Federation
    print('Starting FL')
    print('-'*40)
    start = time.time()

    if args.new_comer:
        num_users_FL = args.num_users * 4 // 5
        num_users_NC = args.num_users - num_users_FL
    else:
        num_users_FL = args.num_users

    m = max(int(args.frac * num_users_FL), 1)

    loss_train = []
    glob_acc = []
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)
    loss_func = nn.CrossEntropyLoss().to(args.device)

    epoch_loss = []
    for iteration in range(args.rounds):

        print(f'----- ROUND {iteration+1} -----')

        sys.stdout.flush()

        net_glob.to(args.device)
        net_glob.train()

        batch_loss = []
        for batch_idx, (images, labels) in enumerate(train_dl_global):
            images, labels = images.to(args.device), labels.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)

            optimizer.zero_grad()
            log_probs = net_glob(images)
            loss = loss_func(log_probs, labels)
            loss.backward()

            optimizer.step()
            batch_loss.append(loss.item())

        # print loss
        loss_avg = sum(batch_loss)/len(batch_loss)
        template = '-- Average Train loss {:.3f}'
        print(template.format(loss_avg))

        _, acc = eval_test(net_glob, args, test_dl_global)

        glob_acc.append(acc)
        template = "-- Global Acc: {:.3f}, Global Best Acc: {:.3f}\n"
        print(template.format(glob_acc[-1], np.max(glob_acc)))


        loss_train.append(loss_avg)

        ## clear the placeholders for the next round

        ## calling garbage collector
        gc.collect()

    end = time.time()
    duration = end-start
    print('-'*40)
    ############################### Centralized Final Results
    print('-'*40)
    print('FINAL RESULTS')
    template = "-- Global Acc Final: {:.2f}"
    print(template.format(glob_acc[-1]))

    template = "-- Global Acc Avg Final [N*C] Rounds: {:.2f}"
    print(template.format(np.mean(glob_acc[-m:])))

    template = "-- Global Best Acc: {:.2f}"
    print(template.format(np.max(glob_acc)))

    print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
    ############################# Saving Print Results

    final_glob = glob_acc[-1]
    avg_final_glob = np.mean(glob_acc[-m:])
    best_glob = np.max(glob_acc)

    return (final_glob, avg_final_glob, best_glob, duration)

def run_centralized(args, fname):
    alg_name = 'Centralized'

    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    exp_fl_time=[]

    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))

        final_glob, avg_final_glob, best_glob, duration = main_centralized(args)

        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        exp_fl_time.append(duration/60)

        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')

        template = "-- Global Final Acc: {:.2f}"
        print(template.format(exp_final_glob[-1]))

        template = "-- Global Avg Final [N*C] Rounds Acc : {:.2f}"
        print(template.format(exp_avg_final_glob[-1]))

        template = "-- Global Best Acc: {:.2f}"
        print(template.format(exp_best_glob[-1]))

        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')

    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)

    template = "-- Global Final Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_final_glob), np.std(exp_final_glob)))

    template = "-- Global Avg Final [N*C] Rounds Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_avg_final_glob), np.std(exp_avg_final_glob)))

    template = "-- Global Best Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_best_glob), np.std(exp_best_glob)))

    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')

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

        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)

    return
