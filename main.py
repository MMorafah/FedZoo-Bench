import numpy as np

import copy
import os
import gc
import pickle
import time
import sys
import datetime

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
from src.benchmarks import *

if __name__ == '__main__':
    print('-'*40)

    args = args_parser()
    if args.gpu == -1:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(args.gpu) ## Setting cuda on GPU

    args.path = args.logdir + args.alg +'/' + args.dataset + '/' + args.partition + '/'
    if args.partition != 'iid':
        if args.partition == 'iid_qskew':
            args.path = args.path + str(args.iid_beta) + '/'
        else:
            if args.niid_beta.is_integer():
                args.path = args.path + str(int(args.niid_beta)) + '/'
            else:
                args.path = args.path + str(args.niid_beta) + '/'

    mkdirs(args.path)

    if args.log_filename is None:
        filename='logs_%s.txt' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    else:
        filename='logs_'+args.log_filename+'.txt'

    sys.stdout = Logger(fname=args.path+filename)

    fname=args.path+filename
    fname=fname[0:-4]
    if args.alg == 'solo':
        alg_name = 'SOLO'
        run_solo(args, fname=fname)
    elif args.alg == 'fedavg':
        alg_name = 'FedAvg'
        run_fedavg(args, fname=fname)
    elif args.alg == 'fedprox':
        alg_name = 'FedProx'
        run_fedprox(args, fname=fname)
    elif args.alg == 'fednova':
        alg_name = 'FedNova'
        run_fednova(args, fname=fname)
    elif args.alg == 'scaffold':
        alg_name = 'Scaffold'
        run_scaffold(args, fname=fname)
    elif args.alg == 'lg':
        alg_name = 'LG'
        run_lg(args, fname=fname)
    elif args.alg == 'perfedavg':
        alg_name = 'Per-FedAvg'
        run_per_fedavg(args, fname=fname)
    elif args.alg == 'pfedme':
        alg_name = 'pFedMe'
        run_pfedme(args, fname=fname)
    elif args.alg == 'ifca':
        alg_name = 'IFCA'
        run_ifca(args, fname=fname)
    elif args.alg == 'cfl':
        alg_name = 'CFL'
        run_cfl(args, fname=fname)
    elif args.alg == 'fedrep':
        alg_name = 'FedRep'
        run_fedrep(args, fname=fname)
    elif args.alg == 'fedper':
        alg_name = 'FedPer'
        run_fedper(args, fname=fname)
    elif args.alg == 'feddf':
        alg_name = 'FedDF'
        run_feddf(args, fname=fname)
    elif args.alg == 'ditto':
        alg_name = 'Ditto'
        run_ditto(args, fname=fname)
    elif args.alg == 'fedem':
        alg_name = 'FedEM'
        run_fedem(args, fname=fname)
    elif args.alg == 'fedbn':
        alg_name = 'FedBN'
        run_fedbn(args, fname=fname)
    elif args.alg == 'mtl':
        alg_name = 'MTL'
        run_mtl(args, fname=fname)
    elif args.alg == 'apfl':
        alg_name = 'APFL'
        run_apfl(args, fname=fname)
    elif args.alg == 'moon':
        alg_name = 'MOON'
        run_moon(args, fname=fname)
    elif args.alg == 'subfedavg_u':
        alg_name = 'SubFedAvg_U'
        run_subfedavg_u(args, fname=fname)
    elif args.alg == 'pacfl':
        alg_name = 'PACFL'
        run_pacfl(args, fname=fname)
    elif args.alg == 'flis':
        alg_name = 'FLIS'
        run_flis(args, fname=fname)
    elif args.alg == 'centralized':
        alg_name = 'Centralized'
        run_centralized(args, fname=fname)
    else:
        print('Algorithm Does Not Exist')
        sys.exit()
