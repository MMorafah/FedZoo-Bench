import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F

# final update is not implemented
class Client_APFL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, global_model,
                 alpha, train_dl_local = None, test_dl_local = None, alpha_apfl=0.8):

        self.name = name
        self.net = model # local model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0
        self.count = 0
        self.save_best = True
        self.alpha_apfl = alpha_apfl

    def train(self, w_local=None, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        
        w_glob = copy.deepcopy(self.net.state_dict())
        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)
                
#                 for k in self.net.state_dict().keys():
#                     w_loc_new[k] = self.alpha_apfl*w_local[k] + (1-self.alpha_apfl)*w_glob[k]
                
                self.net.load_state_dict(w_glob)
                self.net.to(self.device)
                optimizer.zero_grad()
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                
                wt = copy.deepcopy(self.net.state_dict())
                
                self.net.load_state_dict(w_local)
                self.net.to(self.device)
                optimizer.zero_grad()
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                vt = copy.deepcopy(self.net.state_dict())
                for k in self.net.state_dict().keys():
                    w_local[k] = (self.alpha_apfl*vt[k]).to(self.device) + ((1-self.alpha_apfl)*wt[k]).to(self.device)
                    w_local[k] = w_local[k].to(self.device)
                
                w_glob = copy.deepcopy(wt)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return sum(epoch_loss)/len(epoch_loss), w_local, w_glob

    def get_state_dict(self):
        return self.global_model.state_dict()
    def get_best_acc(self):
        return self.acc_best
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy

    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy
