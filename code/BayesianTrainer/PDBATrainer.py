from BayesianTrainer.Trainer import Trainer
from classifiers.loadClassifiers import loadClassifier
from optimisers.optimisers import SGAdaHMC
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time
from BayesianTrainer.utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PDBATrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.classifier = loadClassifier(args)
        self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + self.args.baseClassifier + '/' + self.args.adTrainer + '/'
    
        if not os.path.exists(self.retFolder):
            os.makedirs(self.retFolder)

        self.configureOptimiser()
    def configureOptimiser(self):
        if not self.args.bayesianTraining:
            self.optimiser= SGAdaHMC(self.classifier.model.parameters(),config=dict(lr=self.args.learningRate, alpha=0, gamma=0.01, L=30, T=1e-5, tao=2, C=1))


    def modelEval(self, X, modelNo = -1):

        if self.args.bayesianTraining:
            pred = self.classifier.modelEval(X, modelNo)
        else:
            pred = self.classifier.model(X)
        return pred

   
    def bayesianTrain(self):
        size = len(self.classifier.trainloader.dataset)
        bestLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestClfLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestValClfAcc = [0 for i in range(self.args.bayesianModelNum)]
        logger = SummaryWriter()
        
        #burn-in for classifier
        print(f"burn-in training {self.args.burnIn} epochs")
        for i in range(self.args.burnIn):
            for m in range(self.args.bayesianModelNum):
                for batch, (X, y) in enumerate(self.classifier.trainloader):
                    pred = self.modelEval(X, m)
                    loss = self.classifier.classLoss(pred, y)
                    # Backpropagation
                    self.classifier.optimiserList[m].zero_grad()
                    loss.backward()
                    self.classifier.optimiserList[m].step()

        startTime = time.time()
        valTime = 0
        
        for ep in range(self.args.epochs):
            epLoss = np.zeros(self.args.bayesianModelNum)
            epClfLoss = np.zeros(self.args.bayesianModelNum)

            for i in range(self.args.bayesianModelNum):
                batchNum = 0
                for batch, (X, y) in enumerate(self.classifier.trainloader):
                    batchNum += 1

                    lossPYX = torch.nn.CrossEntropyLoss()(self.modelEval(X, i), y)

                    loss = lossPYX

                    epLoss[i] += loss.detach().item()
                    epClfLoss[i] += lossPYX.detach().item()

                    # Backpropagation
                    self.classifier.optimiserList[i].zero_grad()
                    loss.backward()
                    self.classifier.optimiserList[i].step()
                    
                    if (batchNum - 1) % 20 == 0:
                        loss, current = loss.detach().item(), batch * len(X)
                        print(f"epoch: {ep}/{self.args.epochs} model: {i} loss: {loss:>7f}  lossPYX: {lossPYX:>6f} [{current:>5d}/{size:>5d}]")

                
                # save a model if the best training loss so far has been achieved.
                epLoss[i] /= batchNum
                epClfLoss[i] /= batchNum
                logger.add_scalar(('Loss/train/model%d' % i), epLoss[i], ep)
                logger.add_scalar(('Loss/train/clf/model%d' % i), epClfLoss[i], ep)
                if epLoss[i] < bestLoss[i]:
                    print(f"epoch: {ep} model: {i} per epoch average training loss improves from: {bestLoss[i]} to {epLoss[i]}")
                    bestLoss[i] = epLoss[i]

                if epClfLoss[i] < bestClfLoss[i]:
                    print(f"epoch: {ep} model: {i} per epoch average training clf loss improves from: {bestClfLoss[i]} to {epClfLoss[i]}")
                    bestClfLoss[i] = epClfLoss[i]
                    
                valStartTime = time.time()
               
                self.classifier.setEval(modelNo=i)

                misclassified = 0
                for v, (tx, ty) in enumerate(self.classifier.testloader):
                    pred = torch.argmax(self.modelEval(tx, i), dim=1)
                    diff = (pred - ty) != 0
                    misclassified += torch.sum(diff)

                valClfAcc = 1 - misclassified / len(self.classifier.testloader.dataset)
                
                if valClfAcc > bestValClfAcc[i]:
                    print(f"epoch: {ep} model: {i} per epoch average clf validation acc improves from: {bestValClfAcc[i]} to {valClfAcc}")
                    modelFile = self.retFolder + '/' + str(i) + '_minValLossAppendedModel.pth'
                    torch.save(self.classifier.modelList[i].model.state_dict(), modelFile)
                    bestValClfAcc[i] = valClfAcc

                valEndTime = time.time()
                valTime += (valEndTime - valStartTime) / 3600
                self.classifier.setTrain(modelNo=i)
            print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")
    
        return

    def Dual_Bayesian(self):
        size = len(self.classifier.trainloader.dataset)
        bestLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestClfLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestValClfAcc = [0 for i in range(self.args.bayesianModelNum)]
        logger = SummaryWriter()
       
        # burn-in for classifier
        print(f"burn-in training {self.args.burnIn} epochs")
        for i in range(self.args.burnIn):
            for m in range(self.args.bayesianModelNum):
                for batch, (X, y) in enumerate(self.classifier.trainloader):
                    pred = self.modelEval(X, m)
                    loss = self.classifier.classLoss(pred, y)
                    # Backpropagation
                    self.classifier.optimiserList[m].zero_grad()
                    loss.backward()
                    self.classifier.optimiserList[m].step()

        startTime = time.time()
        valTime = 0
        n_ensembled = 0
        for ep in range(self.args.epochs):
            epLoss = np.zeros(self.args.bayesianModelNum)
            epClfLoss = np.zeros(self.args.bayesianModelNum)

            for i in range(self.args.bayesianModelNum):
                batchNum = 0
                for batch, (X, y) in enumerate(self.classifier.trainloader):
                    batchNum += 1

                    pred = self.modelEval(X, i)

                    lossPYX = torch.nn.CrossEntropyLoss()(pred, y)

                    loss = lossPYX

                    epLoss[i] += loss.detach().item()
                    epClfLoss[i] += lossPYX.detach().item()

                    # Backpropagation
                    self.classifier.optimiserList[i].zero_grad()
                    loss.backward()
                    grad_normal = get_grad(self.classifier.modelList[i].model)
                    norm_grad_normal = cat_grad(grad_normal).norm() 
                    add_into_weights(self.classifier.modelList[i].model, grad_normal, gamma=+0.1 / (norm_grad_normal + 1e-20))
                    loss_add = self.classifier.classLoss(self.modelEval(X, i), y)

                    self.classifier.optimiserList[i].zero_grad()
                    loss_add.backward()
                    grad_add = get_grad(self.classifier.modelList[i].model)
                    add_into_weights(self.classifier.modelList[i].model, grad_normal, gamma=-0.1 / (norm_grad_normal + 1e-20))
                    self.classifier.optimiserList[i].zero_grad()

                    grad_new_dict = OrderedDict()
                    for (name, g_normal), (_, g_add) in zip(grad_normal.items(), grad_add.items()):
                        grad_new_dict[name] = g_normal + (self.args.lam / 0.1) * (g_add - g_normal)

                    assign_grad(self.classifier.modelList[i].model, grad_new_dict)
                    self.classifier.optimiserList[i].step()

                    if ((ep + 1) > self.args.swa_start
                            and ((ep- self.args.swa_start) * len(self.classifier.trainloader) + batch) % (
                                    ((self.args.epochs - self.args.swa_start) * len(
                                        self.classifier.trainloader)) // self.args.swa_n) == 0):
                        update_swag_model(self.classifier.modelList[i], self.classifier.mean_model_list[i], self.classifier.sqmean_model_list[i], n_ensembled)

                        n_ensembled += 1
                    if (batchNum-1) % 20 == 0:
                        loss, current = loss.detach().item(), batch * len(X)
                        print(f"epoch: {ep}/{self.args.epochs}  loss: {loss:>7f}, lossPYX: {lossPYX:>6f}")
                
                retFolder_swag = self.retFolder + '/Dual_Bayesian/'
                if not os.path.exists(retFolder_swag):
                    os.makedirs(retFolder_swag)

                torch.save({"state_dict": self.classifier.modelList[i].state_dict(),
                                "opt_state_dict":  self.classifier.optimiserList[i].state_dict(),
                                "epoch": ep},
                               os.path.join(retFolder_swag + 'multi_ep_%i.pt' % (i)))
                torch.save({"mean_state_dict": self.classifier.mean_model_list[i].state_dict(),
                                "sqmean_state_dict": self.classifier.sqmean_model_list[i].state_dict(),
                                "epoch": ep},
                               os.path.join(retFolder_swag + 'multi_swag_ep_%i.pt' % (i)))
            
                epLoss[i] /= batchNum
                epClfLoss[i] /= batchNum
                logger.add_scalar(('Loss/train/model%d' % i), epLoss[i], ep)
                logger.add_scalar(('Loss/train/clf/model%d' % i), epClfLoss[i], ep)
              
                print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")

                valStartTime = time.time()
               
                self.classifier.setEval(modelNo=i)

                misclassified = 0
                for v, (tx, ty) in enumerate(self.classifier.testloader):
                    pred = torch.argmax(self.modelEval(tx, i), dim=1)
                    diff = (pred - ty) != 0
                    misclassified += torch.sum(diff)

                valClfAcc = 1 - misclassified / len(self.classifier.testloader.dataset)
                print(f"epoch: {ep} clf validation accuracy: {valClfAcc}")
                valEndTime = time.time()
                valTime += (valEndTime - valStartTime) / 3600
        return


