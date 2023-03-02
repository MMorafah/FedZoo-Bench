import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F

class Client_Moon(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_dl_local = None, test_dl_local = None, mu=0.001, temperature=0.5):

        self.name = name
        self.net = model
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
        self.mu = mu
        self.temperature = temperature
        self.prevnet = copy.deepcopy(self.net)

    def train(self, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        self.prevnet.to(self.device)
        self.prevnet.eval()

        global_net = copy.deepcopy(self.net).to(self.device).eval()
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        cos = torch.nn.CosineSimilarity(dim=-1)

        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)
                #labels = labels.view(images.size(0), -1)

                self.net.zero_grad()
                optimizer.zero_grad()
                ## Calculating loss1
                _, pro1, out = self.net(images)
                out = out.view(images.size(0), -1)
                #print(f'out: {out.shape}')
                #print(f'labels: {labels.shape}')
                loss1 = self.loss_func(out, labels)

                ## Calculating loss2
                _, pro2, _ = global_net(images)
                logits = cos(pro1, pro2).reshape(-1,1)
                
                self.prevnet.to(self.device)
                _, pro3, _ = self.prevnet(images)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
            
                logits /= self.temperature
                fake_labels = torch.zeros(images.size(0)).to(self.device).long()
                loss2 = self.mu * self.loss_func(logits, fake_labels)
                
                loss = loss1 + loss2
                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        self.prevnet = copy.deepcopy(self.net)
        self.prevnet.to('cpu')
        self.prevnet.eval()
        del global_net
        return sum(epoch_loss) / len(epoch_loss)

    def get_state_dict(self):
        return self.net.state_dict()
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

                _,_,output = self.net(data)
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

                _,_,output = self.net(data)
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

                _,_,output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy
