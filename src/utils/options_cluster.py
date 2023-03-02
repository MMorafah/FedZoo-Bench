import argparse

## CIFAR-10 has 50000 training images (5000 per class), 10 classes, 10000 test images (1000 per class)
## CIFAR-100 has 50000 training images (500 per class), 100 classes, 10000 test images (100 per class)
## MNIST has 60000 training images (min: 5421, max: 6742 per class), 10000 test images (min: 892, max: 1135
## per class) --> in the code we fixed 5000 training image per class, and 900 test image per class to be
## consistent with CIFAR-10

## CIFAR-10 Non-IID 250 samples per label for 2 class non-iid is the benchmark (500 samples for each client)

def args_parser():
    parser = argparse.ArgumentParser()
    # general federated arguments
    parser.add_argument('--rounds', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--model', type=str, default='lenet5', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset: mnist, cifar10, cifar100")
    parser.add_argument('--partition', type=str, default='iid', help='method of partitioning')
    parser.add_argument('--niid_beta', type=float, default=0.5, help='The parameter for non-iid data partitioning')
    parser.add_argument('--iid_beta', type=float, default=0.5, help='The parameter for iid data partitioning')
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--ntrials', type=int, default=1, help="the number of trials")
    parser.add_argument('--log_filename', type=str, default=None, help='The log file name')
    parser.add_argument('--p_train', type=float, default=1.0, help="Percentage of Train Data")
    parser.add_argument('--p_test', type=float, default=1.0, help="Percentage of Test Data")
    parser.add_argument('--datadir', type=str, default='../data/', help='data directory')
    parser.add_argument('--alg', type=str, default='cluster_fl', help='Algorithm')
    parser.add_argument('--logdir', type=str, default='../logs/', help='logs directory')
    parser.add_argument('--warmup_epoch', type=int, default=0, help="the number of pretrain local ep")

    ## newcomers
    parser.add_argument('--new_comer', action='store_true', help='if true, 80% clients are involved for FL training and 20% are evaluated afterwards')
    parser.add_argument('--ft_epoch', type=int, default=20, help="finetune epochs for new_comer")
        
    # clustering arguments for pacfl
    parser.add_argument('--pacfl_beta', type=float, default=4.6, help="PACFL clustering threshold")
    parser.add_argument('--pacfl_n_basis', type=int, default=3, help="PACFL number of basis per label")
    parser.add_argument('--pacfl_linkage', type=str, default='average', help="PACFL Type of Linkage for HC")
    
    ## FLIS
    parser.add_argument('--nclasses', type=int, default=10, help="number of classes")
    parser.add_argument('--flis_cluster_alpha', type=float, default=0.5, help="FLIS clustering threshold")
    
    ## IFCA
    parser.add_argument('--nclusters', type=int, default=2, help="Number of Clusters for IFCA")
    #parser.add_argument('--num_incluster_layers', type=int, default=2, help="Number of Clusters for IFCA")

    # pruning arguments for subfedavg
    parser.add_argument('--pruning_percent_subfedavg', type=float, default=5,
                        help="Pruning percent for layers (0-100)")
    parser.add_argument('--pruning_target_subfedavg', type=float, default=35,
                        help="Total Pruning target percentage (0-100)")
    parser.add_argument('--dist_thresh_subfedavg', type=float, default=0.0001,
                        help="threshold for fcs masks difference ")
    parser.add_argument('--acc_thresh_subfedavg', type=float, default=55,
                        help="accuracy threshold to apply the derived pruning mask")
    parser.add_argument('--ks', type=int, default=5, help='kernel size to use for convolutions')
    parser.add_argument('--in_ch', type=int, default=3, help='input channels of the first conv layer')
    # parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    # metavar='W', help='weight decay (default: 1e-4)')
    
    ## FedDF 
    parser.add_argument('--distill_lr', type=float, default=0.01, help="Distillation learning rate")
    parser.add_argument('--distill_T', type=float, default=1.0, help="Distillation Temprature")
    parser.add_argument('--distill_E', type=int, default=10, help="Distillation Epoch")
    parser.add_argument('--public_dataset', type=str, default='cifar100', help="Public Distillation Dataset cifar100")
    
    # algorithm specific arguments
    parser.add_argument('--pfedme_beta', type=float, default=1.0, help="pFedMe beta for global model update")
    parser.add_argument('--local_rep_ep', type=int, default=1, help="the number of local rep layers updates for FedRep")
    parser.add_argument('--n_models', type=int, default=3, help="number of mixture distributions M for FedEM")
    parser.add_argument('--mu', type=float, default=0.001, help="FedProx Regularizer")
    parser.add_argument('--alpha', type=float, default=0.25, help="alpha for APFL and FedDyn")
    parser.add_argument('--glob_layers', type=int, default=4, help='number of global or personal layers')
    parser.add_argument('--lam_ditto', type=float, default=0.8, help='Ditto parameter lambda')
    parser.add_argument('--alpha_apfl', type=float, default=0.75, help='APFL parameter Alpha')
    parser.add_argument('--use_project_head', action='store_true', help='whether use projection head or not')
    parser.add_argument('--temperature_moon', type=float, default=0.5, help='Moon parameter Temperature')
    parser.add_argument('--mu_moon', type=float, default=1.0, help='Moon parameter Mu')

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--is_print', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="printing frequency during training rounds")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--load_initial', type=str, default='', help='define initial model path')
    #parser.add_argument('--savedir', type=str, default='../save/', help='save directory')
    #parser.add_argument('--results_save', type=str, default='/', help='define fed results save folder')
    #parser.add_argument('--start_saving', type=int, default=0, help='when to start saving models')

    args = parser.parse_args()
    return args
