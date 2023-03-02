import os
import numpy as np
import torch
import torchvision.transforms as transforms
import time
import copy
import sys

from .datasetzoo import DatasetZoo

def load_data(datadir, dataset, p_train=1.0, p_test=1.0):
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset == 'mnist':
        dataobj_train = DatasetZoo(datadir, dataset='mnist', train=True, transform=transform, download=True, p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='mnist', train=False, transform=transform, download=True, p_data=p_test)
    elif dataset == 'usps':
        dataobj_train = DatasetZoo(datadir, dataset='usps', train=True, transform=transform, download=True, p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='usps', train=False, transform=transform, download=True, p_data=p_test)
    elif dataset == 'fmnist':
        dataobj_train = DatasetZoo(datadir, dataset='fmnist', train=True, transform=transform, download=True, p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='fmnist', train=False, transform=transform, download=True, p_data=p_test)
    elif dataset == 'cifar10':
        dataobj_train = DatasetZoo(datadir, dataset='cifar10', train=True, transform=transform, download=True, p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='cifar10', train=False, transform=transform, download=True, p_data=p_test)
    elif dataset == 'cifar100':
        dataobj_train = DatasetZoo(datadir, dataset='cifar100', train=True, transform=transform, download=True, p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='cifar100', train=False, transform=transform, download=True, p_data=p_test)
    elif dataset == 'svhn':
        dataobj_train = DatasetZoo(datadir, dataset='svhn', train=True, transform=transform, download=True, p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='svhn', train=False, transform=transform, download=True, p_data=p_test)
    elif dataset == 'stl10':
        dataobj_train = DatasetZoo(datadir, dataset='stl10', train=True, transform=transform, download=True, p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='stl10', train=False, transform=transform, download=True, p_data=p_test)
    elif dataset == 'celeba':
        dataobj_train = DatasetZoo(datadir, dataset='celeba', train=True, transform=transform, download=True, p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='celeba', train=False, transform=transform, download=True, p_data=p_test)
    elif dataset == 'tinyimagenet':
        dataobj_train = DatasetZoo(datadir, dataset='tinyimagenet', train=True, transform=transform, download=True, 
                                   p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='tinyimagenet', train=False, transform=transform, download=True, 
                                  p_data=p_test)
    elif dataset == 'femnist':
        dataobj_train = DatasetZoo(datadir, dataset='femnist', train=True, transform=transform, download=True, p_data=p_train)
        dataobj_test = DatasetZoo(datadir, dataset='femnist', train=False, transform=transform, download=True, p_data=p_test)
        #return X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
    
    x_train, y_train = dataobj_train.data, dataobj_train.target
    x_test, y_test = dataobj_test.data, dataobj_test.target

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_test  = np.array(y_test)
    
    return (x_train, y_train, x_test, y_test)

def load_tinyimagenet_data(datadir):
    print(datadir)
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/val/', transform=transform)
    
    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_celeba_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train =  celeba_train_ds.attr[:,gender_index:gender_index+1].reshape(-1)
    y_test = celeba_test_ds.attr[:,gender_index:gender_index+1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)

def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)

def partition_data(dataset, datadir, partition, num_users, niid_beta=0.5, iid_beta=0.5, p_train=1.0, p_test=1.0):
    #np.random.seed(2022)
    #torch.manual_seed(2022)
    dataset_stats = {'mnist': {'ntrain':60000, 'ntest':10000, 'nclasses':10},
                     'fmnist': {'ntrain':60000, 'ntest':10000, 'classes':10}, 
                     'usps': {'ntrain':5000, 'ntest':10000, 'classes':10}, 
                     'cifar10': {'ntrain':50000, 'ntest':10000, 'classes':10}, 
                     'cifar100': {'ntrain':50000, 'ntest':10000, 'classes':100}, 
                     'svhn': {'ntrain':70000, 'ntest':10000, 'classes':10}, 
                     'tinyimagenet': {'ntrain':60000, 'ntest':10000, 'classes':100},
                     'celeba': {'ntrain':60000, 'ntest':10000, 'classes':10}, 
                     'femnist': {'ntrain':60000, 'ntest':10000, 'classes':10}
                    }

    x_train, y_train, x_test, y_test = load_data(datadir, dataset, p_train, p_train)
    
#     idxs_train = np.arange(len(x_train))
#     idxs_test = np.arange(len(x_test))
    
#     perm_train = idxs_train  #np.random.permutation(idxs_train) 
#     perm_test  = idxs_test   #np.random.permutation(idxs_test) 
    
#     p_train1 = int(len(idxs_train)*p_train)
#     p_test1 = int(len(idxs_test)*p_test)
#     perm_train = perm_train[0:p_train1]
#     perm_test = perm_test[0:p_test1]
    
#     x_train = x_train[perm_train] 
#     y_train = y_train[perm_train]

#     x_test = x_test[perm_test] 
#     y_test = y_test[perm_test]
    
    idxs_train = np.arange(len(x_train))
    idxs_test = np.arange(len(x_test))
    
    partitions_train = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_test = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_train_stat = {}
    partitions_test_stat = {}
    
    n_train = y_train.shape[0]
    nclasses = 10
    if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
        nclasses = 2
    elif dataset in ['cifar100']:
        nclasses = 100
    elif dataset == 'tinyimagenet':
        nclasses = 200
            
    if partition == "iid":
        dataidx = np.random.permutation(n_train)
        idx_batch = np.array_split(dataidx, num_users)
        for j in range(num_users):
            partitions_train[j] = np.hstack([partitions_train[j], idx_batch[j]])
            
            dataidx = partitions_train[j]           
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_train_stat[j] = tmp
            
            partitions_test[j] = np.hstack([partitions_test[j], idxs_test])

            dataidx = partitions_test[j]
            unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_test_stat[j] = tmp
            
    elif partition == "iid-qskew":
        assert iid_beta > 0 
        
        dataidx = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(iid_beta, num_users))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(dataidx))
        proportions = (np.cumsum(proportions)*len(dataidx)).astype(int)[:-1]
        idx_batch = np.split(dataidx, proportions)
        
        for j in range(num_users):
            partitions_train[j] = np.hstack([partitions_train[j], idx_batch[j]])
            
            dataidx = partitions_train[j]           
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_train_stat[j] = tmp
            
            partitions_test[j] = np.hstack([partitions_test[j], idxs_test])

            dataidx = partitions_test[j]
            unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_test_stat[j] = tmp

    elif partition == "niid-labeldir":
        assert niid_beta > 0 
        
        min_size = 0
        min_require_size = 15
        #np.random.seed(2022)
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_users)]
            for k in range(nclasses):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(niid_beta, num_users))
                proportions = np.array([p * (len(idx_j) < n_train/num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        #### Assigning samples to each client         
        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            partitions_train[j] = np.hstack([partitions_train[j], idx_batch[j]])

            dataidx = partitions_train[j]           
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_train_stat[j] = tmp   

            for key in tmp.keys():
                dataidx = np.where(y_test==key)[0]
                partitions_test[j] = np.hstack([partitions_test[j], dataidx])

            dataidx = partitions_test[j]
            unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_test_stat[j] = tmp

    elif partition == "niid-labelskew" :
        assert  0 < niid_beta < nclasses
        assert niid_beta.is_integer()
        
        niid_beta = int(niid_beta)
        check=True
        while check:
            times={el:0 for el in np.arange(nclasses)}
            contain={j:None for j in range(num_users)}
            for j in range(num_users):
                ind_a = j%nclasses
                rand_labels = [ind_a]
                kk=1
                while (kk < niid_beta):
                    ind=np.random.choice(np.arange(nclasses), size=1, replace=False).tolist()[0]
                    if (ind not in rand_labels):
                        kk+=1
                        rand_labels.append(ind)
                #rand_labels = np.random.choice(clust, size=num, replace=False)
                contain[j] = rand_labels
                for el in rand_labels:
                    times[el]+=1
            check = not np.all([el>0 for el in times.values()])

        #### Assigning samples to each client 
        for el in range(nclasses):
            dataidx_test = np.where(y_test==el)[0]
            idx_k = np.where(y_train==el)[0]
            np.random.shuffle(idx_k)
            splits = np.array_split(idx_k, times[el])

            ids=0
            for j in range(num_users):
                if el in contain[j]:
                    partitions_train[j]=np.hstack([partitions_train[j], splits[ids]])
                    partitions_test[j] = np.hstack([partitions_test[j], dataidx_test])
                    ids+=1

        for j in range(num_users):
            dataidx = partitions_train[j]           
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_train_stat[j] = tmp    

            dataidx = partitions_test[j]
            unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_test_stat[j] = tmp
       
    elif partition == "niid-randomshard":
        assert  0 < niid_beta < nclasses
        assert niid_beta.is_integer()
        
        niid_beta = int(niid_beta)
        
        dict_train = np.vstack((idxs_train, y_train))
        dict_train = dict_train[:, dict_train[1, :].argsort()]
        sorted_idxs_train = dict_train[0, :]
        sorted_y_train = dict_train[1, :]
        
        len_shard = int(n_train/(niid_beta*num_users))
        
        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        shards = []
        for x in batch(sorted_idxs_train, len_shard):
            shards.append(x)
        rand_shards = np.random.permutation(shards)
        
#         a = [np.arange(i, i+len_shard).tolist() for i in range(0, n_train, len_shard)]
#         c = np.random.permutation(a)
#         idxs_batch = [np.concatenate([c[i], c[i + 1]]) for i in range(0, len(c) - 1, 2)]
        
        cnt = 0
        for j in range(num_users):
            for _ in range(niid_beta):
                partitions_train[j] = np.hstack([partitions_train[j], rand_shards[cnt]])
                cnt+=1
            
            dataidx = partitions_train[j]     
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_train_stat[j] = tmp   
            
            for key in tmp.keys():
                dataidx = np.where(y_test==key)[0]
                partitions_test[j] = np.hstack([partitions_test[j], dataidx])

            dataidx = partitions_test[j]
            unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            partitions_test_stat[j] = tmp
            
    else: 
        print('Partition Does Not Exist')
        sys.exit()
    
    return (partitions_train, partitions_test, partitions_train_stat, partitions_test_stat)

#################################################################