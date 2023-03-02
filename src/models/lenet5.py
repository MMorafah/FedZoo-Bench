import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'LeNet5', 'LeNet5_3', 'LeNet5_TinyImagenet_3', 'LeNet5_MNIST', 'LeNet5_Container', 
    'LeNet5Mnist_SubFedAvg', 'LeNet5Cifar10_SubFedAvg', 'LeNet5Cifar100_SubFedAvg', 
    'LeNet5BNMnist_SubFedAvg', 'LeNet5BNCifar_SubFedAvg'
]


class LeNet5(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet5_3(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(LeNet5_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 48, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 16 * 3 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet5_TinyImagenet_3(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(LeNet5_TinyImagenet_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 48, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 3 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5_MNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(LeNet5_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5_Container(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(LeNet5_Container, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims

        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers

        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
############################# SUB-FEDAVG    

############### Defining MODELS For UnStructured Pruning 
class LeNet5Mnist_SubFedAvg(nn.Module):
    def __init__(self):
        super(LeNet5Mnist_SubFedAvg, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()  ## The original version in LG-FedAvg has the dropout,
        # but in our setup since we are doing pruning, we removed it 
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))   ## if dropout uncomment this line! 
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)   ## if dropout uncomment this line! 
        x = self.fc2(x)
        return x

class LeNet5Cifar10_SubFedAvg(nn.Module):
    def __init__(self):
        super(LeNet5Cifar10_SubFedAvg, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        #self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        #self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet5Cifar100_SubFedAvg(nn.Module):
    def __init__(self):
        super(LeNet5Cifar100_SubFedAvg, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        #self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
################## Defining MODELS For Structured Pruning   
class LeNet5BNMnist_SubFedAvg(nn.Module):
    def __init__(self, cfg=None, ks=5):
        super(LeNet5BNMnist_SubFedAvg, self).__init__()
        if cfg == None: 
            self.cfg = [10, 'M', 20, 'M'] 
        else: 
            self.cfg = cfg
            
        self.ks = ks 
        self.main = nn.Sequential()
        self.make_layers(self.cfg, True) 
        
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        self._initialize_weights()
    
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        idx_maxpool = 1 
        idx_bn = 1
        idx_conv = 1 
        idx_relu = 1
        for v in self.cfg:
            if v == 'M':
                layers += [('maxpool{}'.format(idx_maxpool), nn.MaxPool2d(kernel_size=2, stride=2))]
                idx_maxpool += 1 
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=self.ks)
                if batch_norm:
                    layers += [('conv{}'.format(idx_conv), conv2d), ('bn{}'.format(idx_bn), nn.BatchNorm2d(v)),
                               ('relu{}'.format(idx_relu), nn.ReLU(inplace=True))]
                    idx_bn += 1 
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                idx_conv += 1
                idx_relu += 1 
                in_channels = v
        
        [self.main.add_module(n, l) for n, l in layers]
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.main(x)
        #print(x.shape)
        #x = x.view(-1, self.cfg[-2] * self.ks * self.ks)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNet5BNCifar_SubFedAvg(nn.Module):
    def __init__(self, nclasses = 10, cfg=None, ks=5):
        super(LeNet5BNCifar_SubFedAvg, self).__init__()
        if cfg == None: 
            self.cfg = [6, 'M', 16, 'M'] 
        else: 
            self.cfg = cfg
    
        self.ks = ks 
        fc_cfg = [120, 84, 100]
        
        self.main = nn.Sequential()
        self.make_layers(self.cfg, True)        
        
        self.fc1 = nn.Linear(self.cfg[-2] * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nclasses)
        
        self._initialize_weights()
    
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        idx_maxpool = 1 
        idx_bn = 1
        idx_conv = 1 
        idx_relu = 1
        for v in self.cfg:
            if v == 'M':
                layers += [('maxpool{}'.format(idx_maxpool), nn.MaxPool2d(kernel_size=2, stride=2))]
                idx_maxpool += 1 
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=self.ks)
                if batch_norm:
                    layers += [('conv{}'.format(idx_conv), conv2d), ('bn{}'.format(idx_bn), nn.BatchNorm2d(v)),
                               ('relu{}'.format(idx_relu), nn.ReLU(inplace=True))]
                    idx_bn += 1 
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                idx_conv += 1
                idx_relu += 1 
                in_channels = v
        
        [self.main.add_module(n, l) for n, l in layers]
    
    def forward(self, x):
        #x = self.main.conv1(x)
        x = self.main(x)
        
        #print(x.shape)
        #print(self.cfg[2])
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        return

def updateBN(mymodel, args):
    for m in mymodel.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1       
    return