import copy
from classifiers.ActionClassifier import ActionClassifier
import numpy as np
import os
from classifiers.STGCN import STGCN
from classifiers.CTRGCN import CTRGCN
from classifiers.MSG3D import MSG3D, MSG3D_ensemble
from classifiers.AGCN import AGCN, AGCN_ensemble
from classifiers.ThreeLayerMLP import ThreeLayerMLP
from classifiers.SkateFormer import SkateFormer
from classifiers.STTFormer import STTFormer
from classifiers.FRHEAD import FRHEAD
from datasets.dataloaders import *
class EnsembleModel(ActionClassifier):
    def __init__(self, args,mode='mean'):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader(args)
        self.classifier = args.classifier.split(',')
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(self.classifier)
        self.mode = mode
        self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + args.classifier + '/' 
        self.classifiers=[]
        for model in self.classifier:
            if model == 'STGCN':
                args.classifier='STGCN'
                classifier = STGCN(args)
            elif model == '3layerMLP':
                args.classifier='3layerMLP'
                classifier = ThreeLayerMLP(args)
            elif model == 'MSG3D_joint':
                args.classifier='MSG3D_joint'
                classifier = MSG3D(args)
            elif model == 'MSG3D_bone':
                args.classifier='MSG3D_bone'
                classifier = MSG3D(args)
            elif model == 'MSG3D':
                args.classifier='MSG3D'
                classifier = MSG3D_ensemble(args)
            elif model == 'AGCN_joint':
                args.classifier='AGCN_joint'
                classifier = AGCN(args)
            elif model == 'AGCN_bone':
                args.classifier='AGCN_bone'
                classifier = AGCN(args)
            elif model == 'AGCN':
                args.classifier='AGCN'
                classifier = AGCN_ensemble(args)
            elif model == 'CTRGCN':
                args.classifier='CTRGCN'
                classifier = CTRGCN(args)
            elif model == 'SkateFormer':
                args.classifier='SkateFormer'
                classifier = SkateFormer(args)
            elif model == 'STTFormer':
                args.classifier='STTFormer'
                classifier = STTFormer(args)
            elif model == 'FRHEAD':
                args.classifier='FRHEAD'
                classifier = FRHEAD(args)

            self.classifiers.append(classifier)
    def forward(self, x):
            outputs = []
            for model in self.classifiers:
                outputs.append(model(x))
            outputs = torch.stack(outputs, dim=0)
            if self.mode == 'mean':
                outputs = torch.mean(outputs, dim=0)
                return outputs
            elif self.mode == 'ind':
                return outputs
            else:
                raise NotImplementedError
        
    def modelEval(self, x):
        pred = torch.zeros((len(x), self.classifiers[0].args.classNum), dtype=torch.float).to(device)
        for model in self.classifiers:
            pred += model.model(x)
        return pred

    def classificationLoss(self):
        self.classLoss = torch.nn.CrossEntropyLoss()


    def setEval(self):
        for model in self.classifiers:
            model.model.eval()
   
    def test(self):
        misclassified = 0
        results = np.empty(len(self.classifiers[0].testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.classifiers[0].testloader):
            tx = tx.to(device)
            ty = ty.to(device)
            predY = self.modelEval(tx)
            predY = torch.argmax(predY, dim=1)
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = predY.cpu()
            diff = (predY - ty) != 0
            misclassified += torch.sum(diff)
        error = misclassified / len(self.classifiers[0].testloader.dataset)
        print(f"accuracy: {1-error:>4f}")
        return

    def collectCorrectPredictions(self):
        results = np.empty(len(self.classifiers[0].testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.classifiers[0].testloader):
            pred = torch.argmax(self.modelEval(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.classifiers[0].testloader.dataset.data[results.astype(bool)]
        adLabels = self.classifiers[0].testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")
        path = self.retFolder
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(path + self.args.adTrainFile, clips=adData, classes=adLabels)

        return len(adLabels)

