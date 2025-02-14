import pdb
from classifiers.ActionClassifier import ActionClassifier
from classifiers.ctrgcn.ctrgcn import TCN_GCN_unit, bn_init
from classifiers.ctrgcn.ntu_rgb_d import Graph
import math
import numpy as np
import os
import torch.optim as optim
import torch
from torch import nn
import copy
from torch.utils.tensorboard import SummaryWriter
import time
from datasets.dataloaders import *
from classifiers.utils import *

class CTRGCN(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader(args)
        self.createModel()
        self.steps = [35, 55]
        self.warmUpEpoch = 5
    def createModel(self):
        class Classifier(nn.Module):
            def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                         drop_out=0, adaptive=True):
                super(Classifier, self).__init__()

                if graph is None:
                    raise ValueError()
                else:
                    self.graph = Graph(labeling_mode='spatial')

                A = self.graph.A 

                self.num_class = num_class
                self.num_point = num_point
                self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

                base_channel = 64
                self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
                self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
                self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
                self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
                self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
                self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
                self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
                self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
                self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
                self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

                self.fc = nn.Linear(base_channel * 4, num_class)
                nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
                bn_init(self.data_bn, 1)
                if drop_out:
                    self.drop_out = nn.Dropout(drop_out)
                else:
                    self.drop_out = lambda x: x

                self.features = ''
                self.featureSize = base_channel * 4

            def forward(self, x,flag=False):
                if flag==True:
                    if len(x.shape) == 3:
                        N, T, VC = x.shape
                        x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
                    N, C, T, V, M = x.size()
                    x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
                    x = self.data_bn(x)
                    x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
                    x = self.l1(x)
                    out1=x
                    x = self.l2(x)
                    out2=x
                    x = self.l3(x)
                    out3=x
                    x = self.l4(x)
                    out4 = x
                    x = self.l5(x)
                    out5 = x
                    x = self.l6(x)
                    x = self.l7(x)
                    x = self.l8(x)
                    x = self.l9(x)
                    x = self.l10(x)
                    percept_out=[out1,out2,out3,out4,out5]

                    c_new = x.size(1)
                    x = x.view(N, M, c_new, -1)
                    x = x.mean(3).mean(1)
                    x = self.drop_out(x)

                    self.features = x
                    return self.fc(x),percept_out

                else:
                    if len(x.shape) == 3:
                        N, T, VC = x.shape
                        x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
                    N, C, T, V, M = x.size()

                    x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
                    x = self.data_bn(x)
                    x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
                    x = self.l1(x)
                    x = self.l2(x)
                    x = self.l3(x)
                    x = self.l4(x)
                    x = self.l5(x)
                    x = self.l6(x)
                    x = self.l7(x)
                    x = self.l8(x)
                    x = self.l9(x)
                    x = self.l10(x)

                    c_new = x.size(1)
                    x = x.view(N, M, c_new, -1)
                    x = x.mean(3).mean(1)
                    x = self.drop_out(x)

                    self.features = x
                    return self.fc(x)

        num_class = self.args.classNum
        num_point = 25
        if self.args.dataset == 'hdm05':
            num_person = 1
        elif self.args.dataset == 'ntu60' or self.args.dataset == 'ntu120':
            num_person = 2
        graph = 'classifiers.ctrgcn.ntu_rgb_d.Graph'
        graph_args = dict([('labeling_mode:', 'spatial')])
        in_channels = 3
        drop_out = 0
        adaptive = True

        # create the train data loader, if the routine is 'attack', then the data will be attacked in an Attacker
        if self.args.routine == 'train'  or  self.args.routine == 'adTrain' or  self.args.routine == 'DualBayesian'or self.args.routine == 'finetune' or self.args.routine == 'attack' \
                or self.args.routine == 'bayesianTrain':
            self.model = Classifier(num_class, num_point, num_person, graph, graph_args, in_channels, drop_out, adaptive)
        elif self.args.routine == 'test' or self.args.routine == 'gatherCorrectPrediction' \
                or self.args.routine == 'bayesianTest':
            self.model = Classifier(num_class, num_point, num_person, graph, graph_args, in_channels, drop_out, adaptive)
            self.model.eval()
        else:
            print("no model is created")

        self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/'

        if len(self.args.trainedModelFile) > 0:
            if len(self.args.adTrainer) == 0:
                self.model.load_state_dict(torch.load(self.retFolder + self.args.trainedModelFile,map_location='cuda:0'))
            else:
                if self.args.bayesianTraining:
                    self.model.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                                        self.args.baseClassifier + '/' + self.args.trainedModelFile,map_location='cuda:0'))
                else:
                    self.model.load_state_dict(
                        torch.load(self.retFolder + self.args.adTrainer + '/' + self.args.trainedModelFile))

        self.configureOptimiser()
        self.classificationLoss()
        self.model.to(device)
    def configureOptimiser(self):

        if self.args.optimiser == 'Adam':
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.learningRate, weight_decay=0.0001)
        elif self.args.optimiser == 'SGD':
            self.optimiser = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.args.learningRate,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=0.0004)

    def adjustLearningRate(self, epoch):

        if epoch < self.warmUpEpoch:
            lr = self.args.learningRate * (epoch + 1) / self.warmUpEpoch
        else:
            lr = self.args.learningRate * (
                    0.1 ** np.sum(epoch >= np.array(self.steps)))
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = lr
        return lr


    def classificationLoss(self):

        self.classLoss = torch.nn.CrossEntropyLoss()

    def setTrain(self):
        self.model.train()
    def setEval(self):
        self.model.eval()

    def modelEval(self, X, modelNo = -1):
        return self.model(X)

    def train(self):
        size = len(self.trainloader.dataset)

        bestLoss = np.infty
        bestValAcc = 0

        logger = SummaryWriter()
        startTime = time.time()
        valTime = 0

        for ep in range(self.args.epochs):
            epLoss = 0
            batchNum = 0
            self.adjustLearningRate(ep)
            for batch, (X, y) in enumerate(self.trainloader):
                batchNum += 1
                pred = self.model(X)
                loss = self.classLoss(pred, y)
                
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                epLoss += loss.detach().item()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch: {ep}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                    
            epLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, ep)
            if epLoss < bestLoss:
                if not os.path.exists(self.retFolder):
                    os.makedirs(self.retFolder)
                print(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                torch.save(self.model.state_dict(), self.retFolder + 'minLossModel.pth')
                bestLoss = epLoss
            print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")

            valStartTime = time.time()
            misclassified = 0
            self.model.eval()
            for v, (tx, ty) in enumerate(self.testloader):
                pred = torch.argmax(self.model(tx), dim=1)
                diff = (pred - ty) != 0
                misclassified += torch.sum(diff)
            acc = 1 - misclassified / len(self.testloader.dataset)
            logger.add_scalar('Loss/testing accuracy', acc, ep)
            self.model.train()
            if acc > bestValAcc:
                print(f"epoch: {ep} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")
                torch.save(self.model.state_dict(), self.retFolder + 'minValLossModel.pth')
                bestValAcc = acc
            valEndTime = time.time()
            valTime += (valEndTime - valStartTime)/3600

    
    def finetune(self):
        source_model_path = self.retFolder + '/minValLossModel.pth'
        pre_model_dic = torch.load(source_model_path)
        self.model.load_state_dict(pre_model_dic)

        print('------------------------------------------------------------------------------')
        print("the pretrained model has been loaded, finetune starts")
        mean_model = copy.deepcopy(self.model)
        sqmean_model = copy.deepcopy(self.model)

        self.optimiser = optim.SGD(self.model.parameters(), lr=self.args.learningRate, momentum=0.9, weight_decay=5e-4)

        size = len(self.trainloader.dataset)
        bestLoss = np.infty


        logger = SummaryWriter()
        startTime = time.time()
        valTime = 0

        n_ensembled = 0
        for epoch in range(self.args.epochs):
            bestValAcc = 0
            epLoss = 0
            batchNum = 0
            for batch, (X, y) in enumerate(self.trainloader):
                batchNum += 1
                self.model.train()
                pred = self.model(X)
                loss = nn.functional.cross_entropy(pred, y)

                # finetune
                self.optimiser.zero_grad()
                loss.backward()
                grad_normal = get_grad(self.model)
                norm_grad_normal = cat_grad(grad_normal).norm()  
                add_into_weights(self.model, grad_normal, gamma=+0.1 / (norm_grad_normal + 1e-20))
                loss_add = self.classLoss(self.model(X), y)
                self.optimiser.zero_grad()
                loss_add.backward()
                grad_add = get_grad(self.model)
                add_into_weights(self.model, grad_normal, gamma=-0.1 / (norm_grad_normal + 1e-20))
                self.optimiser.zero_grad()
                grad_new_dict = OrderedDict()
                for (name, g_normal), (_, g_add) in zip(grad_normal.items(), grad_add.items()):
                    grad_new_dict[name] = g_normal + (self.args.lam / 0.1) * (g_add - g_normal)
                assign_grad(self.model, grad_new_dict)
                self.optimiser.step()

                epLoss += loss.detach().item()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch: {epoch}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

                if ((epoch + 1) > self.args.swa_start
                        and ((epoch - self.args.swa_start) * len(self.trainloader) + batch) % (
                                ((self.args.epochs - self.args.swa_start) * len(
                                    self.trainloader)) // self.args.swa_n) == 0):
                    update_swag_model(self.model, mean_model, sqmean_model, n_ensembled)
                    n_ensembled += 1

        
            epLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, epoch)
            if epLoss < bestLoss:
                if not os.path.exists(self.retFolder):
                    os.makedirs(self.retFolder)
                print(f"epoch: {epoch} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                bestLoss = epLoss
            print(f"epoch: {epoch} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")

            torch.save({"state_dict": self.model.state_dict(),
                        "opt_state_dict": self.optimiser.state_dict(),
                        "epoch": epoch},
                       os.path.join(self.retFolder + 'ep.pt'))
            torch.save({"mean_state_dict": mean_model.state_dict(),
                        "sqmean_state_dict": sqmean_model.state_dict(),
                        "epoch": epoch},
                       os.path.join(self.retFolder + 'swag_ep.pt'))

            valStartTime = time.time()
            misclassified = 0
            self.model.eval()
            for v, (tx, ty) in enumerate(self.testloader):
                pred = torch.argmax(self.model(tx), dim=1)
                diff = (pred - ty) != 0
                misclassified += torch.sum(diff)
            acc = 1 - misclassified / len(self.testloader.dataset)
            logger.add_scalar('Loss/testing accuracy', acc, epoch)
            self.model.train()
            if acc > bestValAcc:
                print(f"epoch: {epoch} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")

                torch.save(self.model.state_dict(), self.retFolder + 'finetune.pth')

                bestValAcc = acc
            valEndTime = time.time()
            valTime += (valEndTime - valStartTime) / 3600

        
    def test(self):
        self.model.eval()
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            tx = tx.to(device)
            ty = ty.to(device)
            pred = torch.argmax(self.model(tx), dim=1)
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = pred.cpu()
            diff = (pred - ty) != 0
            misclassified += torch.sum(diff)
        acc = 1 - misclassified / len(self.testloader.dataset)
        unaccuracy = misclassified / len(self.testloader.dataset)
        print(f"unaccuracy: {unaccuracy:>4f}")
        print(f"accuracy: {acc:>4f}")
        return acc

    def collectCorrectPredictions(self):
        self.model.eval()
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            pred = torch.argmax(self.model(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.testloader.dataset.data[results.astype(bool)]
        adLabels = self.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")

        if not os.path.exists(self.retFolder):
            os.mkdir(self.retFolder)
        np.savez_compressed(self.retFolder+self.args.adTrainFile, clips=adData, classes=adLabels)

        return len(adLabels)