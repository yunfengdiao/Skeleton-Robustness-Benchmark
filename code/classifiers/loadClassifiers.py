from classifiers.ActionClassifier import ClassifierArgs
from classifiers.STGCN import STGCN
from classifiers.CTRGCN import CTRGCN
from classifiers.BayesianClassifier import ExtendedBayesianClassifier,ExtendedBayesianClassifier_ensemble
from classifiers.EnsembleModel import EnsembleModel
from classifiers.MSG3D import MSG3D, MSG3D_ensemble
from classifiers.AGCN import AGCN, AGCN_ensemble
from classifiers.ThreeLayerMLP import ThreeLayerMLP
from classifiers.SkateFormer import SkateFormer
from classifiers.STTFormer import STTFormer
from classifiers.FRHEAD import FRHEAD
def loadClassifier(args):
    classifier = ''
    if args.ensemble:
         #args.model = args.model.split(',')
         classifier=EnsembleModel(args)
    else:
        if args.classifier == 'STGCN':
            classifier = STGCN(args)
        elif args.classifier == '3layerMLP':
            classifier = ThreeLayerMLP(args)
        elif args.classifier == 'CTRGCN':
            classifier = CTRGCN(args)
        elif args.classifier == 'MSG3D_joint' or args.classifier == 'MSG3D_bone':
            classifier = MSG3D(args)
        elif args.classifier == 'MSG3D':
            classifier = MSG3D_ensemble(args)
        elif args.classifier == 'ExtendedBayesian':
            if args.baseClassifier == 'MSG3D' or args.baseClassifier == 'AGCN':
                classifier = ExtendedBayesianClassifier_ensemble(args)
            else:
                classifier = ExtendedBayesianClassifier(args)
        elif args.classifier == 'AGCN_joint' or args.classifier == 'AGCN_bone':
            classifier = AGCN(args)
        elif args.classifier == 'AGCN':
            classifier = AGCN_ensemble(args)
        elif args.classifier == 'SkateFormer':
            classifier = SkateFormer(args)
        elif args.classifier == 'STTFormer':
            classifier = STTFormer(args)
        elif args.classifier == 'FRHEAD':
            classifier = FRHEAD(args)
        else:
            print('No classifier created')

    return classifier