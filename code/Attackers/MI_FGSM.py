import sys
import pdb
import os
import torch
from Attackers.Attacker import ActionAttacker
from classifiers.loadClassifiers import loadClassifier
import torch as K
import numpy as np
from Attackers.utils import MyAdam, create_logger,print_args,device
import os
import time

class MIAttacker(ActionAttacker):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.name = 'MI-FGSM'
        self.attackType = args.attackType
        self.epochs = args.epochs
        self.updateRule = args.updateRule
        self.updateClip = args.clippingThreshold
        self.topN = 3
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
    
        decay_factor = 0.1
        g=0.0

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
                g = decay_factor * g + cgs
                cgs=g.detach()

                if ep % 50 == 0:
                    print(f"Iteration {ep}/{self.classifier.args.epochs}, batchNo {batchNo}/{int(len(self.classifier.trainloader.dataset) / self.args.batchSize)}: Class Loss {classLoss.item():>9f}")

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
                    print('foolRate Improved! Iteration %d/%d, batchNo %d: Class Loss %.9f, Fool rate:%.2f' % (
                        ep, self.classifier.args.epochs, batchNo, classLoss.item(), foolRate))
                    maxFoolRate = foolRate

                if maxFoolRate >= 100:
                    divide = ep // 100
                    all = self.classifier.args.epochs // 100
                    for ith in range(divide,all):
                        foolrate_hundreds[ith] = maxFoolRate
                    break

                cgs = cgs / torch.mean(torch.abs(cgs), dim=(1, 2, 3, 4), keepdim=True)

                adData = adData.detach() - 0.01 * cgs
                delta = torch.clamp(adData - tx, min=-self.updateClip, max=self.updateClip)
                adData = (tx + delta).detach()

            folder = '%sAttack_%s_Clipsize%.3f_ep%d/' % (
            self.attackType,self.name,self.updateClip, self.epochs)
            path = self.retFolder + folder
            if not os.path.exists(path):
                os.mkdir(path)

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
    