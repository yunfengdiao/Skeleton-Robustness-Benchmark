import os
import torch
from Attackers.Attacker import ActionAttacker
from classifiers.loadClassifiers import loadClassifier
import torch as K
import numpy as np
from Attackers.utils import MyAdam, create_logger,print_args,device
import os
import time

class SmartAttacker(ActionAttacker):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.name = 'SMART'
        self.perpLossType = args.perpLoss
        self.classWeight = args.classWeight
        self.reconWeight = args.reconWeight
        self.boneLenWeight = args.boneLenWeight
        self.attackType = args.attackType
        self.epochs = args.epochs
        self.updateRule = args.updateRule
        self.updateClip = args.clippingThreshold
        self.deltaT = 1 / 30
        self.topN = 3
        self.alpha = self.updateClip / self.epochs
        self.refBoneLengths = []
        self.optimizer = ''
        self.classifier = loadClassifier(args)
        if isinstance(self.classifier,torch.nn.DataParallel):
            self.classifier = self.classifier.module
        if len(self.args.baseClassifier) > 0:
            self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/' + self.args.baseClassifier + '/' + self.name + '/'
        else:
            self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/' + self.name + '/'
        if not os.path.exists(self.retFolder):
            os.makedirs(self.retFolder)

        self.jointWeights = torch.Tensor([[[0.02, 0.02, 0.02, 0.02, 0.02,
                                      0.02, 0.02, 0.02, 0.02, 0.02,
                                      0.04, 0.04, 0.04, 0.04, 0.04,
                                      0.02, 0.02, 0.02, 0.02, 0.02,
                                      0.02, 0.02, 0.02, 0.02, 0.02]]]).to(device)

    def boneLengths(self, data):

        jpositions = K.reshape(data, (data.shape[0], data.shape[1], -1, 3))

        boneVecs = jpositions - jpositions[:, :, self.classifier.trainloader.dataset.parents, :] + 1e-8

        boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))

        return boneLengths

    def boneLengthLoss (self, parentIds, adData, refBoneLengths):

        jpositions = K.reshape(adData, (adData.shape[0], adData.shape[1], -1, 3))


        boneVecs = jpositions - jpositions[:, :, self.classifier.trainloader.dataset.parents, :] + 1e-8

        boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))

        boneLengthsLoss = K.mean(
            K.sum(K.sum(K.square(boneLengths - refBoneLengths), axis=-1), axis=-1))
        return boneLengthsLoss

    def accLoss (self, adData, refData, jointWeights = None):
        refAcc = (refData[:, 2:, :] - 2 * refData[:, 1:-1, :] + refData[:, :-2, :]) / self.deltaT / self.deltaT

        adAcc = (adData[:, 2:, :] - 2 * adData[:, 1:-1, :] + adData[:, :-2, :]) / self.deltaT / self.deltaT

        if jointWeights == None:
            return K.mean(K.sum(K.sum(K.square(adAcc - refAcc), axis=-1), axis=-1), axis=-1)
        else:
            return K.mean(K.sum(K.sum(K.square(adAcc - refAcc), axis=-1), axis=-1)*jointWeights, axis=-1)

    def perceptualLoss(self, refData, adData, refBoneLengths):


        elements = self.perpLossType.split('_')

        if elements[0] == 'l2' or elements[0] == 'l2Clip':

            diffmx = K.square(refData - adData),
            squaredLoss = K.sum(K.reshape(K.square(refData - adData), (refData.shape[0], refData.shape[1], 25, -1)),
                                axis=-1)

            weightedSquaredLoss = squaredLoss * self.jointWeights

            squareCost = K.sum(K.sum(weightedSquaredLoss, axis=-1), axis=-1)

            oloss = K.mean(squareCost, axis=-1)



        elif elements[0] == 'lInf':
            squaredLoss = K.sum(K.reshape(K.square(refData - adData), (refData.shape[0], refData.shape[1], 25, -1)),
                                axis=-1)

            weightedSquaredLoss = squaredLoss * self.jointWeights

            squareCost = K.sum(weightedSquaredLoss, axis=-1)

            oloss = K.mean(K.norm(squareCost, p=np.inf, dim=0))

        else:
            print('warning: no reconstruction loss')
            return

        if len(elements) == 1:
            return oloss

        elif elements[1] == 'acc-bone':

            jointAcc = self.accLoss(adData, refData)

            boneLengthsLoss = self.boneLengthLoss(self.classifier.trainloader.dataset.parents, adData, refBoneLengths)

            return self.boneLenWeight * boneLengthsLoss + (1 - self.boneLenWeight) * ((1 - self.reconWeight) * oloss + self.reconWeight * jointAcc)

    def targetAttack(self, labels, targettedClasses=[]):
        if len(targettedClasses) <= 0:
            flabels = torch.randint(0, self.args.classNum, labels.shape)
        else:
            flabels = targettedClasses
        return flabels

    def untargetAttack(self, labels):

        flabels = labels

        return flabels

    def foolRateCal(self, rlabels, flabels, logits = None):
        hitIndices = []
        if self.attackType == 'untarget':
            for i in range(0, len(flabels)):
                if flabels[i] != rlabels[i]:
                    hitIndices.append(i)

        elif self.attackType == 'target':
            for i in range(0, len(flabels)):
                if flabels[i] == rlabels[i]:
                    hitIndices.append(i)

        return len(hitIndices) / len(flabels) * 100

    def getUpdate(self, grads, input):

        if self.updateRule == 'gd':
            self.learningRate = 0.01

            return input - grads * self.learningRate

        elif self.updateRule == 'Adam':
            if not hasattr(self, 'Adam'):
                self.Adam = MyAdam()
            return self.Adam.get_updates(grads, input)


    def reshapeData(self, x, toNative=True):
        if toNative:
            x = x.permute(0, 2, 3, 1, 4)
            x = x.reshape((x.shape[0], x.shape[1], -1, x.shape[4]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], -1, 3, x.shape[4]))
            x = x.permute(0, 3, 1, 2, 4)
        return x
    def attack(self):
        self.classifier.setEval()
        overallFoolRate = 0
        overallepochs = 0
        num = int(self.args.epochs/100)
        overallFoolRate_iterations = np.zeros((num,1))
        batchTotalNum = 0
        if len(self.args.adTrainer) > 0:
            log_folder = self.retFolder + 'log_attack/'+self.args.adTrainer # log.txt
        else:
            log_folder = self.retFolder + 'log_attack/'  # log.txt
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        log = create_logger(log_folder, 'train', level='info')
        print_args(self.args, log)
        startTime = time.time()
        image_list = []
        label_list = []
        adimage_list = []
        adlabel_list = []

        for batchNo, (tx, ty) in enumerate(self.classifier.trainloader):

            labels = ty
            if self.attackType == 'target':
                flabels = self.targetAttack(labels).to(device)
            elif self.attackType == 'untarget':
                flabels = self.untargetAttack(labels).to(device)
            else:
                print('specified targetted attack, no implemented')
                return

            adData = tx.clone()
            adData.requires_grad = True
            maxFoolRate = np.NINF
            batchTotalNum += 1
            foolrate_hundreds = np.zeros((num,1))
            cgs = torch.zeros_like(adData).to(device)

            for ep in range(self.classifier.args.epochs):
                adData.requires_grad = True
                pred = self.classifier.modelEval(adData)

                predictedLabels = torch.argmax(pred, axis=1)


                if self.attackType == 'untarget':
                    classLoss = -torch.nn.CrossEntropyLoss()(pred, flabels)
                else:
                    classLoss = torch.nn.CrossEntropyLoss()(pred, flabels)

                adData.grad = None
                classLoss.backward()  
                cgs = adData.grad

                if len(tx.shape) > 3:
                    convertedData = self.reshapeData(tx)
                    convertedAdData = self.reshapeData(adData)
                    if len(convertedData.shape) > 3:
                        percepLoss = 0
                        for i in range(convertedData.shape[-1]):
                            boneLengths = self.boneLengths(convertedData[:, :, :, i])
                            percepLoss += self.perceptualLoss(convertedData[:, :, :, i], convertedAdData[:, :, :, i], boneLengths)
                    else:
                        boneLengths = self.boneLengths(convertedData)
                        percepLoss = self.perceptualLoss(convertedData, convertedAdData, boneLengths)
                else:
                    boneLengths = self.boneLengths(tx)
                    percepLoss = self.perceptualLoss(tx, adData, boneLengths)
                adData.grad = None
                percepLoss.backward()
                pgs = adData.grad

                if ep % 50 == 0:
                    print(f"Iteration {ep}/{self.classifier.args.epochs}, batchNo {batchNo}/{int(len(self.classifier.trainloader.dataset) / self.args.batchSize)}: Class Loss {classLoss.item():>9f}, Perceptual Loss: {percepLoss.item():>9f}")
                if self.attackType == 'untarget':
                    foolRate = self.foolRateCal(ty, predictedLabels)
                elif self.attackType == 'target':
                    foolRate = self.foolRateCal(flabels, predictedLabels)
                else:
                    print('specified targetted attack, no implemented')
                    return

                if ep % 100 == 0:
                    ith = int(ep / 100)
                    foolrate_hundreds[ith] = foolRate
                if maxFoolRate < foolRate:
                    print('foolRate Improved! Iteration %d/%d, batchNo %d: Class Loss %.9f, Perceptual Loss: %.9f, Fool rate:%.2f' % (
                        ep, self.classifier.args.epochs, batchNo, classLoss.item(), percepLoss.item(), foolRate))
                    maxFoolRate = foolRate

                if maxFoolRate >= 100:
                    divide = ep // 100
                    all = self.classifier.args.epochs // 100
                    for ith in range(divide,all):
                        foolrate_hundreds[ith] = maxFoolRate
                    break

                cgsView = cgs.view(cgs.shape[0], -1)
                cgsnorms = torch.norm(cgsView, dim=1) + 1e-18
                cgsView /= cgsnorms[:, np.newaxis]

                pgsView = pgs.view(pgs.shape[0], -1)
                pgsnorms = torch.norm(pgsView, dim=1) + 1e-18
                pgsView /= pgsnorms[:, np.newaxis]
                
                temp = self.getUpdate(cgs.detach() * self.classWeight + pgs.detach() * (1 - self.classWeight), adData.detach()).to(device).detach()
                missedIndices = []

                if self.attackType == 'untarget':
                    for i in range(len(ty)):
                        if ty[i] == predictedLabels[i]:
                            missedIndices.append(i)
                elif self.attackType == 'target':
                    for i in range(len(ty)):
                        if flabels[i] != predictedLabels[i]:
                            missedIndices.append(i)

                tempCopy = adData.detach()
                if self.updateClip > 0:
                    updates = temp[missedIndices].detach() - adData[missedIndices].detach()
                    for ci in range(updates.shape[0]):
                        updateNorm = torch.norm(updates[ci])
                        if updateNorm > self.updateClip:
                            updates[ci] = updates[ci] * 5*self.updateClip / updateNorm
                    tempCopy[missedIndices] += updates
                else:
                    tempCopy[missedIndices] = temp[missedIndices]
                
                delta = torch.clamp(adData - tx, min=-self.updateClip, max=self.updateClip)
                adData = (tx + delta).detach()
               
            folder = '%sAttack_%s_Clipsize%.3f_ep%d/' % (
            self.attackType,self.name,self.updateClip, self.epochs)
            path = self.retFolder + folder
            if not os.path.exists(path):
                os.mkdir(path)
            print(path)

            label_list.append(flabels.detach().cpu())
            image_list.append(tx.detach().cpu())

            pred = self.classifier.modelEval(adData)
            predictedLabels = torch.argmax(pred, axis=1)
            adlabel_list.append(predictedLabels.detach().cpu())
            adimage_list.append(adData.detach().cpu())

            overallFoolRate += maxFoolRate
            overallFoolRate_iterations += foolrate_hundreds
            overallepochs += (ep + 1)
            log.info('batchNo %d: ' % (batchNo))
            log.info(f"Per hundreds fool rate is {overallFoolRate_iterations / batchTotalNum}")
            log.info(f"Current fool rate is {overallFoolRate / batchTotalNum}")
            log.info(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600} hours")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            if batchTotalNum > 30:
                break

        image_list = torch.cat(image_list, 0)
        label_list = torch.cat(label_list, 0)
        adimage_list = torch.cat(adimage_list, 0)
        adlabel_list = torch.cat(adlabel_list, 0)

        save_path1 = self.retFolder + folder + 'original_data.pt'
        torch.save((image_list, label_list), save_path1)

        save_path2 = self.retFolder + folder + 'visual_adData.pt'
        torch.save((adimage_list, adlabel_list), save_path2)

        save_path3 = self.retFolder + folder + 'transfer_adv_data.pt'
        torch.save((adimage_list, label_list), save_path3)

        print('end')
        print(f"Overall fool rate is {overallFoolRate/batchTotalNum}")
        print(f"Time for generating an adversarial sample is {(time.time() - startTime) / (60 * batchTotalNum * self.args.batchSize)} mins")
        print(f"Overall epochs is {overallepochs / batchTotalNum}")

        return overallFoolRate/batchTotalNum

    def attack_batch(self,tx,ty):

        self.classifier.setEval()

        overallFoolRate = 0
        batchTotalNum = 0

        labels = ty
        if self.attackType == 'target':
            flabels = self.targetAttack(labels).to(device)
        elif self.attackType == 'untarget':
            flabels = self.untargetAttack(labels).to(device)
        else:
            print('specified targetted attack, no implemented')
            return

        adData = tx.clone().detach()
        adData.requires_grad = True
        minCl = np.PINF
        maxFoolRate = np.NINF
        batchTotalNum += 1
        updates = torch.zeros_like(adData).to(device)

        for ep in range(self.args.attackep):
            # compute the classification loss and gradient
            adData.requires_grad = True
            pred = self.classifier.modelEval(adData)
            predictedLabels = torch.argmax(pred, axis=1)

            # computer the classfication loss gradient

            if self.attackType == 'untarget':
                classLoss = -torch.nn.CrossEntropyLoss()(pred, flabels)
            else:
                classLoss = torch.nn.CrossEntropyLoss()(pred, flabels)

            adData.grad = None
            classLoss.backward()
            cgs = adData.grad

            if len(tx.shape) > 3:
                convertedData = self.reshapeData(tx)
                convertedAdData = self.reshapeData(adData)
                if len(convertedData.shape) > 3:
                    percepLoss = 0
                    for i in range(convertedData.shape[-1]):
                        boneLengths = self.boneLengths(convertedData[:, :, :, i])
                        percepLoss += self.perceptualLoss(convertedData[:, :, :, i], convertedAdData[:, :, :, i],
                                                          boneLengths)
                else:
                    boneLengths = self.boneLengths(convertedData)
                    percepLoss = self.perceptualLoss(convertedData, convertedAdData, boneLengths)
            else:
                boneLengths = self.boneLengths(tx)
                percepLoss = self.perceptualLoss(tx, adData, boneLengths)

            adData.grad = None
            percepLoss.backward()
            pgs = adData.grad

            cgsView = cgs.view(cgs.shape[0], -1)
            pgsView = pgs.view(pgs.shape[0], -1)

            cgsnorms = torch.norm(cgsView, dim=1) + 1e-18
            pgsnorms = torch.norm(pgsView, dim=1) + 1e-18

            cgsView /= cgsnorms[:, np.newaxis]
            pgsView /= pgsnorms[:, np.newaxis]

            temp = self.getUpdate(cgs.detach() * self.classWeight + pgs.detach() * (1 - self.classWeight), adData.detach()).to(device).detach()

            missedIndices = []

            if self.attackType == 'untarget':
                for i in range(len(ty)):
                    if ty[i] == predictedLabels[i]:
                        missedIndices.append(i)
            elif self.attackType == 'target':
                for i in range(len(ty)):
                    if flabels[i] != predictedLabels[i]:
                        missedIndices.append(i)

            tempCopy = adData.detach()
            if self.updateClip > 0:

                updates = temp[missedIndices].detach() - adData[missedIndices].detach()
                for ci in range(updates.shape[0]):

                    updateNorm = torch.norm(updates[ci])
                    if updateNorm > self.updateClip:
                        updates[ci] = updates[ci] * 5 * self.updateClip / updateNorm  # l_inifty
                tempCopy[missedIndices] += updates
            else:
                tempCopy[missedIndices] = temp[missedIndices]
            delta = torch.clamp(adData - tx, min=-self.updateClip, max=self.updateClip)
            adData = (tx + delta).detach()
        overallFoolRate += maxFoolRate
        print(f"Current fool rate is {overallFoolRate / batchTotalNum}")

        adData = adData.detach()
        del updates
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        return adData

