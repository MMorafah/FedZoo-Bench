import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F
from src.utils import *

class Client_FedEM(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_dl_local = None, test_dl_local = None, n_models=3):

        self.name = name
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
        self.n_models = n_models
        self.models = [model]
        for component in range(self.n_models - 1):
            new_model = copy.deepcopy(model)
            new_model.apply(weight_init)
            self.models.append(new_model)
        self.models_weights = torch.ones(n_models) / n_models

    def train(self, lr_factor = 1, is_print = False):
        with torch.no_grad():
            all_loss = []
            self.models_weights = self.models_weights.to(self.device)
            for component, model in enumerate(self.models):
                model.to(self.device)
                model.eval()
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.device), labels.to(self.device)
                    labels = labels.type(torch.LongTensor).to(self.device)

                    #optimizer.zero_grad()
                    log_probs = model(images)
                    loss = self.loss_func(log_probs, labels).squeeze()
                    batch_loss.append(loss)
                all_loss.append(batch_loss)

            # E-step
            self.samples_weights = F.softmax((torch.log(self.models_weights) - torch.Tensor(all_loss).to(self.device).T), dim=1).T

            # M-step
            self.models_weights = self.samples_weights.mean(dim=1)

        epoch_loss = []
        for iteration in range(self.local_ep):
            for component, model in enumerate(self.models):
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr * lr_factor, momentum=self.momentum, weight_decay=0)
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.device), labels.to(self.device)
                    labels = labels.type(torch.LongTensor).to(self.device)

                    optimizer.zero_grad()
                    log_probs = model(images)
                    loss = self.loss_func(log_probs, labels)
#                     loss = (loss.T @ self.samples_weights[batch_idx]) / loss.size(0)
                    loss = loss * self.samples_weights[component][batch_idx]
                    loss.backward()

                    optimizer.step()
                    batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)

    def get_state_dict(self, component=None):
        if component is not None:
            return self.models[component].state_dict()
        else:
            return [model.state_dict() for model in self.models]

    def get_best_acc(self):
        return self.acc_best

    def get_count(self):
        return self.count

    def set_state_dict(self, state_dicts):
        for model, state_dict in zip(self.models, state_dicts):
            model.load_state_dict(state_dict)

    def eval_test(self):
        criterion = nn.NLLLoss(reduction="none")
        test_loss = 0
        correct = 0
        with torch.no_grad():

            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                pred = 0
                for component in range(self.n_models):
                    model = self.models[component]
                    model.to(self.device)
                    model.eval()

                    output = model(data)
                    pred += self.models_weights[component] * F.softmax(output, dim=1)

                pred = torch.clamp(pred, min=0., max=1.)
                test_loss += criterion(torch.log(pred), target).sum().item()

                pred = pred.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_test_glob(self, glob_dl):
        criterion = nn.NLLLoss(reduction="none")
        test_loss = 0
        correct = 0
        with torch.no_grad():

            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                pred = 0
                for component in range(self.n_models):
                    model = self.models[component]
                    model.to(self.device)
                    model.eval()

                    output = model(data)
                    pred += self.models_weights[component] * F.softmax(output, dim=1)

                pred = torch.clamp(pred, min=0., max=1.)
                test_loss += criterion(torch.log(pred), target).sum().item()

                pred = pred.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_train(self):
        criterion = nn.NLLLoss(reduction="none")
        test_loss = 0
        correct = 0
        with torch.no_grad():

            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                pred = 0
                for component in range(self.n_models):
                    model = self.models[component]
                    model.to(self.device)
                    model.eval()

                    output = model(data)
                    pred += self.models_weights[component] * F.softmax(output, dim=1)

                pred = torch.clamp(pred, min=0., max=1.)
                test_loss += criterion(torch.log(pred), target).sum().item()

                pred = pred.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return train_loss, accuracy
