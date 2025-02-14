from AdversarialTrainers.AdversarialTrainer import AdversarialTrainer
from classifiers.loadClassifiers import loadClassifier
from Attackers.SMART import SmartAttacker
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time
from AdversarialTrainers.utils import *
class TRADES(AdversarialTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.attacker = SmartAttacker(args)
        self.classifier = loadClassifier(args)
        if self.args.classifier == 'ExtendedBayesian':
            self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + self.args.baseClassifier + '/' + self.args.adTrainer + '/'
        else:
            self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + self.args.adTrainer + '/'
        if not os.path.exists(self.retFolder):
            os.makedirs(self.retFolder)


    def setTrain(self):
        self.classifier.model.train()
    def setEval(self):
        self.classifier.model.eval()

    def modelEval(self, X, modelNo = -1):
        return self.classifier.model(X)

    def adversarialTrain(self):
        size = len(self.classifier.trainloader.dataset)
        beta = 6
        bestLoss = np.infty
        bestValAcc = 0
        bestValAcc_adv = 0

        logger = SummaryWriter(log_dir=self.retFolder + 'run/')  # save in specific path
        log_folder = self.retFolder + 'log/'  # log.txt
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        log = create_logger(log_folder,'train',level='info')
        print_args(self.args,log)
        startTime = time.time()
        valTime = 0
        criterion_kl = nn.KLDivLoss(size_average=False)
        epsilon = self.args.clippingThreshold
        step_size = 0.0001
        for ep in range(self.args.rep,self.args.epochs):
            epLoss = 0
            batchNum = 0
            self.classifier.adjustLearningRate(ep)
            #print(f"epoch: {ep} GPU memory allocated: {torch.cuda.memory_allocated(1)}")
            for batch, (X, y) in enumerate(self.classifier.trainloader):
                X = X.to(device)
                y = y.to(device)
                batchNum += 1
                # Compute prediction and loss
                batch_size = len(X)
                self.setEval()
                x_adv = X.clone().detach()
                x_adv = x_adv.detach() + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
                x_adv = torch.clamp(x_adv, 0, 1).detach()
                for _ in range(self.args.attackep):
                    x_adv.requires_grad_()
                    with torch.enable_grad():
                        loss_kl = criterion_kl(F.log_softmax(self.classifier.model(x_adv), dim=1),
                                               F.softmax(self.classifier.model(X), dim=1))
                    grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                    x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)
                self.classifier.setTrain()
                x_adv = Variable(x_adv, requires_grad=False)
                pred_adv = self.classifier.model(x_adv)
                #pred = self.classifier.model(X)
                # calculate robust loss
                logits = self.classifier.model(X)
                loss_natural = F.cross_entropy(logits, y)
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(self.classifier.model(x_adv), dim=1),
                                                                F.softmax(self.classifier.model(X), dim=1))
                loss = loss_natural + beta * loss_robust
                epLoss += loss.detach().item()
                # Backpropagation
                self.classifier.optimiser.zero_grad()
                loss.backward()
                self.classifier.optimiser.step()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch: {ep}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

            #print(f"epoch: {ep} GPU memory allocated after one epoch: {torch.cuda.memory_allocated(1)}")
            # save a model if the best training loss so far has been achieved.
            epLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, ep)
            log.info('advLoss %.3f/train %d' % (epLoss, ep))
            if epLoss < bestLoss:
                if not os.path.exists(self.retFolder):
                    os.makedirs(self.retFolder)
                #print(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                log.info(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                torch.save(self.classifier.model.state_dict(), self.retFolder + 'minLossModel_adtrained.pth')
                bestLoss = epLoss
            #print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")
            log.info(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")
            if ep % 5 == 0:
                # run validation and save a model if the best validation loss so far has been achieved.
                valStartTime = time.time()
                misclassified = 0
                misclassified_adv = 0
                self.classifier.model.eval()
                for v, (tx, ty) in enumerate(self.classifier.testloader):
                    tx = tx.to(device)
                    ty = ty.to(device)
                    tx_adv = self.attacker.attack_batch(tx,ty)
                    pred_adv = torch.argmax(self.classifier.model(tx_adv), dim=1)
                    pred = torch.argmax(self.classifier.model(tx), dim=1)
                    diff = (pred - ty) != 0
                    diff_adv = (pred_adv - ty) != 0
                    misclassified += torch.sum(diff)
                    misclassified_adv += torch.sum(diff_adv)
                acc = 1 - misclassified / len(self.classifier.testloader.dataset)
                acc_adv = 1 - misclassified_adv / len(self.classifier.testloader.dataset)
                logger.add_scalar('Loss/testing accuracy', acc, ep)
                logger.add_scalar('Loss/testing robust accuracy', acc_adv, ep)
                log.info('test robust acc %.3f, test acc %.3f /val %d' % (acc_adv, acc, ep))
                self.classifier.model.train()
                if acc > bestValAcc:
                    #print(f"epoch: {ep} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")
                    log.info(f"epoch: {ep} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")
                    torch.save(self.classifier.model.state_dict(), self.retFolder + 'minValLossModel.pth')
                    bestValAcc = acc
                if acc_adv > bestValAcc_adv:
                    #print(f"epoch: {ep} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")
                    log.info(f"epoch: {ep} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")
                    torch.save(self.classifier.model.state_dict(), self.retFolder + 'minValLossModel_adtrained.pth')
                    bestValAcc_adv = acc_adv
                valEndTime = time.time()
                valTime += (valEndTime - valStartTime)/3600
    def bayesianAdversarialTrain(self):
        size = len(self.classifier.trainloader.dataset)
        beta = 6
        bestLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestValAcc = [0 for i in range(self.args.bayesianModelNum)]
        bestValAcc_adv = [0 for i in range(self.args.bayesianModelNum)]
        logger = SummaryWriter(log_dir=self.retFolder + 'run/')  # save in specific path
        log_folder = self.retFolder + 'log/'  
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        log = create_logger(log_folder,'train',level='info')
        print_args(self.args,log)
        startTime = time.time()
        valTime = 0
        criterion_kl = nn.KLDivLoss(size_average=False)
        epsilon = self.args.clippingThreshold
        step_size = epsilon / self.args.attackep
        for ep in range(self.args.rep,self.args.epochs):
            epLoss = np.zeros(self.args.bayesianModelNum)
 
            for i in range(self.args.bayesianModelNum):
                batchNum = 0
                for batch, (X, y) in enumerate(self.classifier.trainloader):
                    batchNum += 1
                    # Compute prediction and loss
                    batch_size = len(X)
                    self.classifier.setEval(modelNo=i)
                    x_adv = X.detach() + 0.0001 * torch.randn(X.shape).cuda().detach()
                    for _ in range(self.args.attackep):
                        x_adv.requires_grad_()
                        with torch.enable_grad():
                            loss_kl = criterion_kl(F.log_softmax(self.classifier.modelEval(x_adv,i), dim=1),
                                                   F.softmax(self.classifier.modelEval(X,i), dim=1))
                        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                        x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
                    self.classifier.setTrain(modelNo=i)
                    x_adv = Variable(x_adv, requires_grad=False)

                    # calculate robust loss
                    logits = self.classifier.modelEval(X,i)
                    loss_natural = F.cross_entropy(logits, y)
                    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(self.classifier.modelEval(x_adv,i), dim=1),
                                                                    F.softmax(self.classifier.modelEval(X,i), dim=1))
                    loss = loss_natural + beta * loss_robust
                    epLoss[i] += loss.detach().item()
                    # Backpropagation
                    self.classifier.optimiserList[i].zero_grad()
                    loss.backward()
                    self.classifier.optimiserList[i].step()
                    loss, current = loss.item(), batch * len(X)
                    if batch % 10 == 0:
                        print(f"epoch: {ep}/{self.args.epochs} model: {i}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


                epLoss[i] /= batchNum
                logger.add_scalar('Loss/train', epLoss[i], ep)
                log.info('advLoss %.3f/train %d' % (epLoss[i], ep))
                if epLoss[i] < bestLoss[i]:
                    if not os.path.exists(self.retFolder):
                        os.makedirs(self.retFolder)
                    
                    log.info(f"epoch: {ep}  model: {i} per epoch average training loss improves from: {bestLoss[i]} to {epLoss[i]}")
                    torch.save(self.classifier.modelList[i].model.state_dict(), self.retFolder + '/' + str(i)+ '_minLossModel_adtrained.pth')
                    bestLoss[i] = epLoss[i]
                log.info(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")
                if ep % 5 == 0 and ep >= 90:
                    misclassified = 0
                    misclassified_adv = 0
                    self.classifier.setEval(modelNo=i)
                    for v, (tx, ty) in enumerate(self.classifier.testloader):
                        tx_adv = self.attacker.attack_batch(tx,ty)
                        pred_adv = torch.argmax(self.classifier.modelEval(tx_adv,i), dim=1)
                        pred = torch.argmax(self.classifier.modelEval(tx,i), dim=1)
                        diff = (pred - ty) != 0
                        diff_adv = (pred_adv - ty) != 0
                        misclassified += torch.sum(diff)
                        misclassified_adv += torch.sum(diff_adv)
                        if v > 10:
                            break
                    acc = 1 - misclassified / ((v+1)*batch_size)
                    acc_adv = 1 - misclassified_adv / ((v+1)*batch_size)
                    logger.add_scalar('Loss/testing accuracy', acc, ep)
                    logger.add_scalar('Loss/testing robust accuracy', acc_adv, ep)
                    log.info('test robust acc %.3f, test acc %.3f /val %d/model %d' % (acc_adv, acc, ep, i))

                    if acc > bestValAcc[i]:
                        log.info(f"epoch: {ep} model: {i} per epoch average validation accuracy improves from: {bestValAcc[i]} to {acc}")
                        torch.save(self.classifier.modelList[i].model.state_dict(), self.retFolder + '/' + str(i)+'_minValLossModel.pth')
                        bestValAcc[i] = acc
                    if acc_adv > bestValAcc_adv[i]:
                        log.info(f"epoch: {ep} per epoch average validation accuracy improves from: {bestValAcc[i]}  to {acc}")
                        torch.save(self.classifier.modelList[i].model.state_dict(), self.retFolder+'/' + str(i)+ '_minValLossModel_adtrained.pth')
                        bestValAcc_adv[i] = acc_adv

                    self.classifier.setTrain(modelNo=i)