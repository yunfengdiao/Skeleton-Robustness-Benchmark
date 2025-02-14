import sys
from classifiers.ActionClassifier import ActionClassifier
from classifiers.stgcn.st_gcn import st_gcn
from classifiers.stgcn.utils.graph import Graph
import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
import copy
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from datasets.dataloaders import *
from classifiers.utils import *

 
class STGCN(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader(args)
        self.createModel()
        self.steps = [10, 40]
    def createModel(self):
        class Classifier(nn.Module):
            r"""Spatial temporal graph convolutional networks.

            Args:
                in_channels (int): Number of channels in the input data
                num_class (int): Number of classes for the classification task
                graph_args (dict): The arguments for building the graph
                edge_importance_weighting (bool): If ``True``, adds a learnable
                    importance weighting to the edges of the graph
                **kwargs (optional): Other parameters for graph convolution units

            Shape:
                - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
                - Output: :math:`(N, num_class)` where
                    :math:`N` is a batch size,
                    :math:`T_{in}` is a length of input sequence,
                    :math:`V_{in}` is the number of graph nodes,
                    :math:`M_{in}` is the number of instance in a frame.
            """

            def __init__(self, args, dataloader):
                super().__init__()

                self.dataShape = dataloader.dataset.data.shape

                in_channels = args.inputDim
                num_class = args.classNum
                edge_importance_weighting = args.edge_importance_weighting
    
                if args.dataset == 'hdm05':
                    graph='hdm05'
                elif args.dataset == 'ntu60' or args.dataset == 'ntu120'or args.dataset == 'ntu60_300':
                    graph= args.graph_layout
                self.graph = Graph(graph, args.graph_strategy)
                A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
                self.register_buffer('A', A)

                spatial_kernel_size = A.size(0)
                temporal_kernel_size = 9
                kernel_size = (temporal_kernel_size, spatial_kernel_size)
            
                self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
                self.data_bn.cuda()


                self.st_gcn_networks = nn.ModuleList((
                    st_gcn(in_channels, 64, kernel_size, 1, residual=False),
                    st_gcn(64, 64, kernel_size, 1, dropout = args.dropout),
                    st_gcn(64, 64, kernel_size, 1, dropout = args.dropout),
                    st_gcn(64, 64, kernel_size, 1, dropout = args.dropout),
                    st_gcn(64, 128, kernel_size, 2, dropout = args.dropout),
                    st_gcn(128, 128, kernel_size, 1, dropout = args.dropout),
                    st_gcn(128, 128, kernel_size, 1, dropout = args.dropout),
                    st_gcn(128, 256, kernel_size, 2, dropout = args.dropout),
                    st_gcn(256, 256, kernel_size, 1, dropout = args.dropout),
                    st_gcn(256, 256, kernel_size, 1, dropout = args.dropout),
                ))

                # initialize parameters for edge importance weighting
                if edge_importance_weighting:
                    self.edge_importance = nn.ParameterList([
                        nn.Parameter(torch.ones(self.A.size()))
                        for i in self.st_gcn_networks
                    ])
                else:
                    self.edge_importance = [1] * len(self.st_gcn_networks)
                    
                self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
                self.features = ''
                self.featureSize = 256
            def forward(self, x):
                # #our data is [batch_size, num_of_frames, dofs]
                # #so we first convert our data into the format STGCN uses
                # #Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
                # x = x.reshape((x.shape[0], x.shape[1], -1, 3, 1))
                # # x = x.permute(0, 2, 1)
                # # x = x.reshape((x.shape[0], 3, -1, x.shape[2], 1))
                # x = x.permute(0, 3, 1, 2, 4)

                # data normalization
                N, C, T, V, M = x.size()
                x = x.permute(0, 4, 3, 1, 2).contiguous()
                x = x.view(N * M, V * C, T)

                # A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
                # in_channels = self.temp
                # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
                # self.data_bn.cuda()

                x = self.data_bn(x)
                x = x.view(N, M, V, C, T)
                x = x.permute(0, 1, 3, 4, 2).contiguous()
                x = x.view(N * M, C, T, V)

                # forward
                for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
                    x, _ = gcn(x, self.A * importance)

                # global pooling
                x = F.avg_pool2d(x, x.size()[2:])
                x = x.view(N, M, -1, 1, 1).mean(dim=1)
                self.features = x
                # prediction
                x = self.fcn(x)
                x = x.view(x.size(0), -1)

                return x

            def extract_feature(self, x):
                # #our data is [batch_size, num_of_frames, dofs]
                # #so we first convert our data into the format STGCN uses
                #
                # x = x.reshape((x.shape[0], x.shape[1], -1, 3, 1))
                # # x = x.permute(0, 2, 1)
                # # x = x.reshape((x.shape[0], 3, -1, x.shape[2], 1))
                # x = x.permute(0, 3, 1, 2, 4)

                # data normalization
                N, C, T, V, M = x.size()
                x = x.permute(0, 4, 3, 1, 2).contiguous()
                x = x.view(N * M, V * C, T)
                x = self.data_bn(x)
                x = x.view(N, M, V, C, T)
                x = x.permute(0, 1, 3, 4, 2).contiguous()
                x = x.view(N * M, C, T, V)

                # forward
                for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
                    x, _ = gcn(x, self.A * importance)

                _, c, t, v = x.size()
                feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

                # prediction
                x = self.fcn(x)
                output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

                return output, feature

        # create the train data loader, if the routine is 'attack', then the data will be attacked in an Attacker
        if self.args.routine == 'train' or  self.args.routine == 'adTrain' or  self.args.routine == 'DualBayesian' or self.args.routine == 'attack' or self.args.routine == 'finetune'\
             or self.args.routine == 'bayesianTrain':
            self.model = Classifier(self.args, self.trainloader)
        elif self.args.routine == 'test' or self.args.routine == 'gatherCorrectPrediction' \
                or self.args.routine == 'bayesianTest':
            self.model = Classifier(self.args, self.testloader)
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
                    weight_decay=0.0001)

    def adjustLearningRate(self, epoch):
        if self.args.optimiser == 'SGD':
            if self.steps:
                lr = self.args.learningRate * (
                    0.1**np.sum(epoch >= np.array(self.steps)))
                for param_group in self.optimiser.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            else:
                self.lr = self.args.learningRate
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