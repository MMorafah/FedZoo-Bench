# FedZoo-Bench

FedZoo-Bench, is an open-source library based on [PyTorch](https://pytorch.org/) that facilitates experimentation in federated learning by providing researchers with a comprehensive set of standardized and customizable features such as training, Non-IID data partitioning, fine-tuning, performance evaluation, fairness assessment, and generalization to newcomers, for both global and personalized FL approaches.

## Code organization

Most modules are put in the `src` directory. The main modules to run the code (which you may modify to customize your experiments) are organized as follows:

| Directory | Contents |
| --------- | ------- |
| `src/benchmarks` | Server modules and main training functions for each algorithm. |
| `src/client`     | Client modules. |
| `src/data`       | Dataset pre-processing and loaders. |
| `src/models`     | Neural network architectures. |
| `src/utils`      | Other utils and argument parsers. |

## Algorithms

9 Global FL algorithms and 15 Personalized FL algorithms are supported in this code.

**Global** Algorithms:

| Algorithm | Publication | Code |
| --------- | ----------- | ---- |
| FedAvg   | PMLR, 2017, https://arxiv.org/abs/1602.05629 | |
| FedProx  | MLSys, 2020, https://arxiv.org/abs/1812.06127 | https://github.com/litian96/FedProx |
| FedNova  | NeurIPS, 2020, https://arxiv.org/abs/2007.07481 | https://github.com/JYWa/FedNova |
| Scaffold | PMLR, 2020, https://arxiv.org/abs/1910.06378 | |
| MOON     | CVPR, 2021, https://arxiv.org/abs/2103.16257 | https://github.com/QinbinLi/MOON |
| FedBN    | ICLR, 2021, https://arxiv.org/abs/2102.07623 | https://github.com/med-air/FedBN |
| FedDyn   | ICLR, 2021, https://arxiv.org/abs/2111.04263 | https://github.com/alpemreacar/FedDyn |
| FedDF    | NeurIPS, 2020, https://arxiv.org/abs/2006.07242 | |
| FedAvgM  | https://arxiv.org/abs/1909.06335 | |
| FedavgM  | https://arxiv.org/abs/2002.06440 | |

**Personalized** Algorithms:

| Algorithm | Publication | Code |
| --------- | ----------- | ---- |
| Global + Fine-Tuning | | |
| LG                   | NeurIPS Workshop on Federated Learning, 2019, https://arxiv.org/abs/2001.01523 | https://github.com/pliang279/LG-FedAvg |
| Per-FedAvg           | NeurIPS, 2020, https://arxiv.org/abs/2002.07948 | |
| CFL                  | IEEE transactions on neural networks and learning systems, 2020, https://arxiv.org/abs/1910.01991 | https://github.com/felisat/clustered-federated-learning |
| IFCA                 | NeurIPS, 2020, https://arxiv.org/abs/2006.04088 | https://github.com/jichan3751/ifca |
| MTL                  | NeurIPS, 2017, https://arxiv.org/abs/1705.10467 | https://github.com/gingsmith/fmtl |
| Ditto                | ICML, 2021, https://arxiv.org/abs/2012.04221 | https://github.com/litian96/ditto |
| FedRep               | ICML, 2021, https://arxiv.org/abs/2102.07078 | https://github.com/lgcollins/FedRep |
| FedPer               | arXiv preprint, 2019, https://arxiv.org/abs/1912.00818 | |
| FedFOMO              | ICLR, 2021, https://arxiv.org/abs/2012.08565 | https://github.com/NVlabs/FedFomo |
| pFedMe               | NeurIPS, 2020, https://arxiv.org/abs/2006.08848 | https://github.com/CharlieDinh/pFedMe |
| FedEM                | NeurIPS, 2021, https://arxiv.org/abs/2108.10252 | https://github.com/omarfoq/FedEM |
| APFL                 | arXiv preprint, 2020, https://arxiv.org/abs/2003.13461 | |
| SubFedAvg            | ICDCSW, 2021, https://arxiv.org/abs/2105.00562 | https://github.com/MMorafah/Sub-FedAvg |
| PACFL                | arXiv preprint, 2022, https://arxiv.org/abs/2209.10526 | https://github.com/MMorafah/PACFL |
| FLIS                 | https://arxiv.org/abs/2112.07157 | `TODO: need verification` |
<!-- | HeteroFL             | ICLR, 2021, https://arxiv.org/abs/2010.01264 | https://github.com/dem123456789/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients | -->


## Datasets

We support 9 widely used datasets: `MNIST`, `CIFAR-10`, `CFIAR-100`, `USPS`, `SVHN`, `FMNIST`, `FEMNIST`, `Tiny-ImageNet`, and `STL-10`.

The datasets should be aligned as follows:
```
datadir
├── cifar-10-batches-py
│   └── ...
├── cifar-100-python
│   └── ...
├── fmnist
│   └── ...
...
```
and the argument `datadir` should point to this datadir.

Most dataests will be downloaded automatically if they are not in the directory except Tiny-ImageNet, which is available at http://cs231n.stanford.edu/tiny-imagenet-200.zip.

## Usage

We provide scripts to run the algorithms, which are put under `scripts/`. Here is an example to run the script:
```
cd scripts
bash fedavg.sh
```

Custom experiments can be performed by running `main.py` with customized arguments. For example:
```
python main.py --ntrials 3 ...
```

### General Arguments
The descriptions of general arguments are as follows:

| Parameter | Description |
| --------- | ----------- |
| ntrials      | The number of total runs. |
| rounds       | The number of communication rounds per run. |
| num_users    | The number of clients. |
| nclass       | Classes or shards per user. |
| nsample_pc   | The number of samples per class or shard for each client. |
| frac         | The fraction of clients updated per round. |
| local_ep     | The number of local training epochs. |
| local_bs     | Local batch size. |
| lr           | The learning rate for local models. |
| momentum     | The momentum for the optimizer. |
| model        | Network architecture. Options: `lenet5`, `simple-cnn-3`, `resnet9`, `resnet`, `vgg16`. |
| dataset      | The dataset for training and testing. Options are discussed above. |
| partition    | How datasets are partitioned. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns). |
| datadir      | The path of datasets. |
| logdir       | The path to store logs. |
| log_filename | The folder name for multiple runs. E.g., with `ntrials=3` and `log_filename=$trial`, the logs of 3 runs will be located in 3 folders named `1`, `2`, and `3`. |
| alg          | Federated learning algorithm. Options are discussed above. |
| niid_beta    | The parameter for non-iid data partitioning. |
| iid_beta     | The parameter for iid data partitioning. |
| gpu          | The IDs of GPU to use. E.g., 0. |
| print_freq   | The frequency to print training logs. E.g., with `print_freq=10`, training logs are displayed every 10 communication rounds. |

Algorithm specific arguments:
| Parameter | Description |
| --------- | ----------- |
| local_rep_ep    | The number of local rep layers updates for FedRep. |
| feddf_n         | The number of server-side model fusion: N. |
| glob_layers     | The number of global or personal layers. |

### FedProx
| Parameter | Description |
| --------- | ----------- |
| mu              | FedProx regularizer parameter. default=0.001. (float)|

### Ditto
| Parameter | Description |
| --------- | ----------- |
| lam_ditto       | Ditto parameter lambda. default=0.8. (float)|

### APFL
| Parameter | Description |
| --------- | ----------- |
| alpha_apfl      | APFL parameter Alpha. default=0.75. (float)|

### MOON
| Parameter | Description |
| --------- | ----------- |
| use_project_head | whether use projection head or not. (action store true)|
| temperature_moon | Moon parameter Temperature. default=0.5. (float)|
| mu_moon         | Moon parameter Mu. default=1.0. (float)|

### PACFL
| Parameter | Description |
| --------- | ----------- |
| pacfl_beta      | PACFL clustering threshold. |
| pacfl_n_basis   | PACFL number of basis per label. |
| pacfl_linkage   | PACFL Type of Linkage for HC. |
| nclasses        | The mumber of classes for PACFL. |

### FLIS
| Parameter | Description |
| --------- | ----------- |
| flis_cluster_alpha      | FLIS clustering threshold. default=0.5. (float) |
| nclasses        | The mumber of classes for public dataset. default=10. (int)|

### pFedMe
| Parameter | Description |
| --------- | ----------- |
| pfedme_beta     | pFedMe beta for global model update. default=1.0. (float)|

### IFCA
| Parameter | Description |
| --------- | ----------- |
| nclusters       | The number of Clusters for IFCA. default=2. (int)|

### Sub-FedAvg
| Parameter | Description |
| --------- | ----------- |
| pruning_percent_subfedavg | Pruning percent for layers (0-100) for subfedavg. default= 5. (float)|
| pruning_target_subfedavg | Total Pruning target percentage (0-100) for subfedavg. default= 35. (float)|
| dist_thresh_subfedavg | The threshold for fcs masks difference for subfedavg . default=0.0001. (float)|
| acc_thresh_subfedavg | The accuracy threshold to apply the derived pruning mask for subfedavg. default=55. (float)|

### FedEM 
| Parameter | Description |
| --------- | ----------- |
| n_models        | The number of mixture distributions M for FedEM. |

### Generalization to New Comers
For generalization to new comers experiment, please use the following arguments in your script. 

| Parameter | Description |
| --------- | ----------- |
| new_comer    | If passed in, 80% clients are involved for FL training and 20% are evaluated afterwards as new comers. |
| ft_epoch     | The number of finetune epochs for new comers. |

## Citation
If you find this repository useful, please cite our work:
```
@article{morafah2023practical,
  title={A Practical Recipe for Federated Learning Under Statistical Heterogeneity Experimental Design},
  author={Morafah, Mahdi and Wang, Weijia and Lin, Bill},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2023},
  publisher={IEEE}
}
```
## Contact
For any issues please feel free to submit an issue. We are also open to any contributions.
