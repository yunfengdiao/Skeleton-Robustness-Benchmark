

from classifiers.ActionClassifier import ActionClassifier
import numpy as np
import os
from torch import nn
import copy
import torch.optim as optim
from classifiers.utils import *
import time
from torch.utils.tensorboard import SummaryWriter
from datasets.dataloaders import *

class ThreeLayerMLP(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader(args)
        self.createModel()
        self.steps = [10, 40]

    def createModel(self):
        class Classifier(nn.Module):
            def __init__(self,args, dataloader):
                super().__init__()

                self.dataShape = dataloader.dataset.data.shape
                self.flatten = nn.Flatten()
                self.mlpstack = nn.Sequential(
                    nn.Linear(self.dataShape[1] * self.dataShape[2], 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, dataloader.dataset.classNum),
                    nn.ReLU()
                )
            def forward(self, x):
                x = self.flatten(x)
                logits = self.mlpstack(x)
                return logits

        if self.args.routine == 'train'or  self.args.routine == 'adTrain'or self.args.routine == 'DualBayesian'or self.args.routine == 'finetune'or self.args.routine == 'attack' \
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
            if len(self.args.args.adTrainer) == 0:
                self.model.load_state_dict(torch.load(self.retFolder + self.args.trainedModelFile))
            else:
                if self.args.args.bayesianTraining:
                    self.model.load_state_dict(torch.load(self.args.retFolder + self.args.dataset + '/' +
                                        self.args.args.baseClassifier + '/' + self.args.trainedModelFile))
                else:
                    self.model.load_state_dict(
                        torch.load(self.retFolder + self.args.args.adTrainer + '/' + self.args.trainedModelFile))

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




