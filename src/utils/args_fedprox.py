import argparse

def args_parser_fedprox():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--rounds', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--ntrials', type=int, default=1, help="the number of trials")
    parser.add_argument('--mu', type=float, default=0.001, help="FedProx Regularizer")
    parser.add_argument('--log_filename', type=str, default=None, help='The log file name')
    parser.add_argument('--model', type=str, default='lenet5', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help="name of dataset: mnist, cifar10, cifar100")
    parser.add_argument('--savedir', type=str, default='../save/', help='save directory')
    parser.add_argument('--datadir', type=str, default='../data/', help='data directory')
    parser.add_argument('--partition', type=str, default='noniid-#label2', help='method of partitioning')
    parser.add_argument('--alg', type=str, default='fedavg', help='Algorithm')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--niid_beta', type=float, default=0.5, help='The parameter for non-iid data partitioning')
    parser.add_argument('--iid_beta', type=float, default=0.5, help='The parameter for iid data partitioning')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--is_print', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="printing frequency during training rounds")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--load_initial', type=str, default='', help='define initial model path')

    args = parser.parse_args()
    return args
