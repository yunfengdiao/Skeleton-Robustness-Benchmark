from AdversarialTrainers.AdversarialTrainer import AdversarialTrainer
from classifiers.loadClassifiers import loadClassifier
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time
from AdversarialTrainers.utils import *

class EBMATrainer(AdversarialTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.classifier = loadClassifier(args)

        self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + self.args.baseClassifier + '/'
        if args.bayesianTraining:
            self.replayBufferList = [[] for i in range(args.bayesianModelNum)]
        else:
            if len(args.bufferSamples) > 0:
                self.replayBuffer = torch.FloatTensor(np.load(args.bufferSamples)['clips'])
            else:
                self.replayBuffer = []

        if not os.path.exists(self.retFolder):
            os.makedirs(self.retFolder)

        self.configureOptimiser()

    def configureOptimiser(self):

        if not self.args.bayesianTraining:
            self.optimiser = torch.optim.Adam(self.classifier.model.parameters(), lr=self.args.learningRate, weight_decay=0.0001)

    def modelEval(self, X, modelNo = -1):

        if self.args.bayesianTraining:
            pred = self.classifier.modelEval(X, modelNo)
        else:
            pred = torch.nn.ReLU()(self.classifier.model(X))
        return pred

    def initRandom(self, X):
        return torch.FloatTensor(np.random.uniform(-1, 1, X.shape).astype('float32'))

    def sampleP0(self, X, rb = -1, y=None):
        #currently, we do not do conditioned sampling, y is not used
        if rb == -1:
            replayBuffer = self.replayBuffer
        else:
            replayBuffer = self.replayBufferList[rb]
        if len(replayBuffer) == 0:
            return self.initRandom(X), []
        self.args.bufferSize = len(replayBuffer) if y is None else len(replayBuffer)
        inds = torch.randint(0, self.args.bufferSize, (self.args.batchSize,))

        bufferSamples = replayBuffer[inds]
        randomSamples = self.initRandom(X)
        choose_random = (torch.rand(self.args.batchSize) < self.args.reinitFreq).float()[:, None, None]
        samples = choose_random * randomSamples + (1 - choose_random) * bufferSamples
        return samples, inds

    def sampleX(self, X, rb = -1, y=None):
        if rb == -1:
            replayBuffer = self.replayBuffer
        else:
            replayBuffer = self.replayBufferList[rb]
        if not self.args.bayesianTraining:
            self.classifier.model.eval()
        # # get batch size
        # bs = self.args.batchSize if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        initSample, bufferInds = self.sampleP0(X, rb)
        x_k = torch.autograd.Variable(initSample, requires_grad=True).to(device)
        # sgld
        for k in range(self.args.samplingStep):
            f_prime = torch.autograd.grad(torch.logsumexp(self.modelEval(x_k), 1).sum(), [x_k], retain_graph=True)[0]
            x_k.data += self.args.sgldLr * f_prime + np.sqrt(self.args.sgldLr*2) * self.args.sgldStd * torch.randn_like(x_k)
        if not self.args.bayesianTraining:
            self.classifier.model.train()

        final_samples = x_k.detach()
        # update replay buffer
        if len(replayBuffer) > 0:
            replayBuffer[bufferInds] = final_samples.cpu()
        return final_samples

    def boneLengths(self, x):
        x = self.reshapeData(x)
        if len(x.shape) == 3:
            jpositions = torch.reshape(x, (x.shape[0], x.shape[1], -1, 3))

            boneVecs = jpositions - jpositions[:, :, self.classifier.trainloader.dataset.parents, :]

            boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1) + 1.e-10)
        elif len(x.shape) == 4:
            jpositions = torch.reshape(x, (x.shape[0], x.shape[1], -1, 3, x.shape[3]))

            boneVecs = jpositions - jpositions[:, :, self.classifier.trainloader.dataset.parents, :, :]

            boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-2) + 1.e-10)
        else:
            boneLengths = []
            print('unknown data structure for bone lengths computation')

        return boneLengths

    def xXTildeLoss(self, x, xTilde, refBoneLengths, drv_order=3):
        loss_drv = 0
        x_k = x.clone()
        x_tilde_k = xTilde.clone()

        #bone length loss
        boneLengths = self.boneLengths(xTilde)

        boneLengthsLoss = torch.nn.functional.mse_loss(boneLengths, refBoneLengths)

        x_k = self.reshapeData(x_k)
        x_tilde_k = self.reshapeData(x_tilde_k)

        for k in range(drv_order):
            x_k = x_k[:, 1:] - x_k[:, :-1]
            x_tilde_k = x_tilde_k[:, 1:] - x_tilde_k[:, :-1]

            loss_drv += torch.nn.functional.mse_loss(x_k, x_tilde_k)

        return loss_drv + boneLengthsLoss

    def log_xTilde_x(self, x, xTilde, y, beta, refBoneLengths, drv_order=3, modelNo = -1):
        loss_motion = self.xXTildeLoss(x, xTilde, refBoneLengths, drv_order)

        # There are three possible formulations
        # 1. logp(xTilde | x, y)
        # logits = self.modelEval(xTilde, modelNo)
        # prob = 0
        # for i in range(len(y)):
        #     prob += logits[i, y[i]]
        # prob /= len(y)
        # return -prob + beta * loss_motion

        # 2. \sum_y logp(xTilde | x)
        #prob = torch.logsumexp(self.modelEval(xTilde), 0).sum()
        #return -prob + beta * loss_motion

        #use cross-entropy
        loss = torch.nn.CrossEntropyLoss()(self.modelEval(xTilde, modelNo), y)

        return -(loss + beta * loss_motion)

    def sampleXTilde(self, x, y, refBoneLengths):
        if not self.args.bayesianTraining:
            self.classifier.model.eval()
        # based on data x, generate perturbed data x_tilde
        if self.args.perturbThreshold > 0:
            xTilde = x.cpu() + np.random.uniform(-self.args.perturbThreshold, self.args.perturbThreshold, x.shape).astype('float32')
        else:
            xTilde = np.copy(x)

        x_tilde_k = torch.autograd.Variable(xTilde, requires_grad=True).to(device)
        for k in range(self.args.samplingStep):
            f_prime = torch.autograd.grad(self.log_xTilde_x(x, x_tilde_k, y, self.args.drvWeight, refBoneLengths), [x_tilde_k], retain_graph=True)[0]
            x_tilde_k.data += self.args.sgldLr * f_prime + np.sqrt(self.args.sgldLr*2) * self.args.sgldStd * torch.randn_like(x_tilde_k)

        if not self.args.bayesianTraining:
            self.classifier.model.train()

        return x_tilde_k.detach()

    def reshapeData(self, x, toNative=True):
        #ntu format is N, C, T, V, M (batch_no, channel, frame, node, person)
        if toNative:
            x = x.permute(0, 2, 3, 1, 4)
            x = x.reshape((x.shape[0], x.shape[1], -1, x.shape[4]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], -1, 3, x.shape[4]))
            x = x.permute(0, 3, 1, 2, 4)
        return x

    def adversarialTrain(self):

        size = len(self.classifier.trainloader.dataset)

        bestLoss = np.infty
        bestClfLoss = np.infty
        bestValClfAcc = 0
        logger = SummaryWriter()

        #burn-in for classifier
        print(f"burn-in training {self.args.burnIn} epochs")
        for i in range(self.args.burnIn):
            for batch, (X, y) in enumerate(self.classifier.trainloader):
                pred = self.modelEval(X)
                loss = self.classifier.classLoss(pred, y)
                # Backpropagation
                self.classifier.optimiser.zero_grad()
                loss.backward()
                self.classifier.optimiser.step()

        startTime = time.time()
        valTime = 0
        for ep in range(self.args.epochs):
            epLoss = 0
            epClfLoss = 0
            batchNum = 0

            for batch, (X, y) in enumerate(self.classifier.trainloader):
                refBoneLengths = self.boneLengths(X)
                batchNum += 1


                # sample x and compute logP(X)

                XSamples = self.sampleX(X)

                logPX = self.modelEval(X).mean()

                logPXSamples = self.modelEval(XSamples).mean()

                lossLogPX = -(logPX - logPXSamples)

                # compute logP(x_tilde|X, y)

                XTilde = self.sampleXTilde(X, y, refBoneLengths)

                lossPYXTilde = torch.nn.CrossEntropyLoss()(self.modelEval(XTilde), y)
                lossXXTilde = self.xXTildeLoss(X, XTilde, refBoneLengths)

                lossPXTildeXY = lossPYXTilde + self.args.drvWeight * lossXXTilde

                lossPYX = torch.nn.CrossEntropyLoss()(self.modelEval(X), y)

                loss = self.args.xWeight*lossLogPX + self.args.xTildeWeight*lossPXTildeXY + self.args.clfWeight * lossPYX

                epLoss += loss.detach().item()
                epClfLoss += lossPYX.detach().item()

                # Backpropagation
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if (batchNum-1) % 20 == 0:
                    loss, current = loss.detach().item(), batch * len(X)
                    print(f"epoch: {ep}/{self.args.epochs}  loss: {loss:>7f}  lossLogPX: {lossLogPX:>6f}, lossPXTildeXY: {lossPXTildeXY:>6f}, lossPYX: {lossPYX:>6f}"
                          f"(logPX: {logPX:>6f} logPXSamples: {logPXSamples:>6f} lossPYXTilde: {lossPYXTilde:>7f} lossXXTilde: {lossXXTilde:>6f}) [{current:>5d}/{size:>5d}]")

            # save a model if the best training loss so far has been achieved.
            epLoss /= batchNum
            epClfLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, ep)
            logger.add_scalar('Loss/train/clf', epClfLoss, ep)
            if epLoss < bestLoss:
                print(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                bestLoss = epLoss


            if epClfLoss < bestClfLoss:
                print(f"epoch: {ep} per epoch average training clf loss improves from: {bestClfLoss} to {epClfLoss}")
                bestClfLoss = epClfLoss
            print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")
            # run validation and save a model if the best validation loss so far has been achieved.
            valStartTime = time.time()

            misclassified = 0
            self.classifier.model.eval()
            for v, (tx, ty) in enumerate(self.classifier.testloader):
                pred = torch.argmax(self.modelEval(tx), dim=1)
                diff = (pred - ty) != 0
                misclassified += torch.sum(diff)

            self.classifier.model.train()

            valClfAcc = 1 - misclassified / len(self.classifier.testloader.dataset)
            if valClfAcc > bestValClfAcc:
                print(f"epoch: {ep} clf validation accuracy improves from: {bestValClfAcc} to {valClfAcc}")
                modelFile = self.retFolder + '/minValLossModel_adtrained.pth'
                torch.save(self.classifier.model.state_dict(), modelFile)
                bestValClfAcc = valClfAcc
            valEndTime = time.time()
            valTime += (valEndTime - valStartTime) / 3600
        return
    def bayesianAdversarialTrain(self):

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
                    refBoneLengths = self.boneLengths(X)

                    # sample x and compute logP(X)

                    XSamples = self.sampleX(X, i)

                    logPX = self.modelEval(X, i).mean()

                    logPXSamples = self.modelEval(XSamples, i).mean()

                    lossLogPX = -(logPX - logPXSamples)

                    # compute logP(x_tilde|X, y)

                    XTilde = self.sampleXTilde(X, y, refBoneLengths)

                    lossPYXTilde = torch.nn.CrossEntropyLoss()(self.modelEval(XTilde, i), y)
                    lossXXTilde = self.xXTildeLoss(X, XTilde, refBoneLengths)

                    lossPXTildeXY = lossPYXTilde + self.args.drvWeight * lossXXTilde

                    lossPYX = torch.nn.CrossEntropyLoss()(self.modelEval(X, i), y)

                    loss = self.args.xWeight*lossLogPX + self.args.xTildeWeight*lossPXTildeXY + self.args.clfWeight * lossPYX

                    epLoss[i] += loss.detach().item()
                    epClfLoss[i] += lossPYX.detach().item()

                    # Backpropagation
                    self.classifier.optimiserList[i].zero_grad()
                    loss.backward()
                    self.classifier.optimiserList[i].step()

                    if (batchNum - 1) % 20 == 0:
                        loss, current = loss.detach().item(), batch * len(X)
                        print(f"epoch: {ep}/{self.args.epochs} model: {i} loss: {loss:>7f}  lossLogPX: {lossLogPX:>6f}, lossPXTildeXY: {lossPXTildeXY:>6f}, lossPYX: {lossPYX:>6f} [{current:>5d}/{size:>5d}]")
                        print(f"epoch: {ep}/{self.args.epochs} model: {i} (logPX: {logPX:>6f} logPXSamples: {logPXSamples:>6f} lossPYXTilde: {lossPYXTilde:>7f} lossXXTilde: {lossXXTilde:>6f}) [{current:>5d}/{size:>5d}]")

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
                # run validation and save a model if the best validation loss so far has been achieved.

                self.classifier.setEval(modelNo = i)

                misclassified = 0
                for v, (tx, ty) in enumerate(self.classifier.testloader):
                    pred = torch.argmax(self.modelEval(tx, i), dim=1)
                    diff = (pred - ty) != 0
                    misclassified += torch.sum(diff)

                valClfAcc = 1 - misclassified / len(self.classifier.testloader.dataset)

                if valClfAcc > bestValClfAcc[i]:
                    print(f"epoch: {ep} model: {i} per epoch average clf validation acc improves from: {bestValClfAcc[i]} to {valClfAcc}")
                    modelFile = self.retFolder + '/' + str(i) + '_minValLossAppendedModel_adtrained.pth'
                    torch.save(self.classifier.modelList[i].model.state_dict(), modelFile)
                    bestValClfAcc[i] = valClfAcc

                valEndTime = time.time()
                valTime += (valEndTime - valStartTime) / 3600
                self.classifier.setTrain(modelNo = i)
            print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")

        return