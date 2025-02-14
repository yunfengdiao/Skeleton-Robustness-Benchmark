from numpy.distutils.fcompiler import str2bool
from classifiers.loadClassifiers import loadClassifier
from Attackers.loadAttackers import loadAttacker
from BayesianTrainer.loadTrainers import loadTrainer
from AdversarialTrainers.loadATrainers import loadATrainer

def parameteSettingForOptimizers(ap):
    ap.add_argument("-lr", "--learningRate", type=float, required=False, help="to specify an adversarial attacker",
                    default=5e-3)

def parameteSettingForAttackers(ap):

    ap.add_argument("-attacker", "--attacker", type=str, required=False, help="to specify an adversarial attacker",
                    default='TASAR')
    ap.add_argument("-at", "--attackType", type=str, required=False, help="to specify the type of attack",
                    default='untarget')
    ap.add_argument("-pl", "--perpLoss", type=str, required=False, help="to specify the perceptual loss",
                    default='l2_acc-bone')
    ap.add_argument("-ur", "--updateRule", type=str, required=False,
                    help="to specify the optimisation method for adversarial attack", default='gd')
    ap.add_argument("-cw", "--classWeight", type=float, required=False, help="to specify the weight for classification loss",
                    default=0.6) 
    ap.add_argument("-rw", "--reconWeight", type=float, required=False, help="to specify the weight for reconstruction loss",
                    default=0.4)
    ap.add_argument("-blw", "--boneLenWeight", type=float, required=False, help="to specify the weight for bone length loss",
                    default=0.7)
    ap.add_argument("-cp", "--clippingThreshold", type=float, required=False, help="set up the clipping threshold in update",
                    default=0.01)
    ap.add_argument("--transfer_attack", type=bool, required=False, help="whether to perform transfer attack",
                    default=False)
    ap.add_argument("--transfer_path", type=str, required=False, help="path", default='')
    ap.add_argument('--ensemble',type=bool, default=False, help='enable ensemble attack')




def parameteSettingForAdTrainers(ap):
    ap.add_argument("-adTrainer", "--adTrainer", type=str, required=False, help="to specify an adversarial trainer",
                    default='')
    ap.add_argument("-BayesianTrainer", "--BayesianTrainer", type=str, required=False, help="to specify an Bayesian trainer",
                    default='')
    ap.add_argument("-aep", "--attackep", type=int, required=False, help="to specify the number of epochs for training",
                    default=20)
    ap.add_argument("-rep", "--rep", type=int, required=False, help="resume epoches",
                    default=0)
    ap.add_argument("--burnIn", type=int, required=False, help="training the classifier in normal training mode for -burnIn steps",
                    default=0)
    ap.add_argument("--samplingStep", type=int, required=False, help="sampling step for SGLD", default=10)
    ap.add_argument("--drvWeight", type=float, required=False, help="weight for the motion derivatives", default=1e-3)
    ap.add_argument("--bufferSize", type=int, required=False, help="the replay buffer size for SGLD sampling",default=100)
    ap.add_argument("--reinitFreq", type=float, required=False, help="the re-init probability of SGLD sampling",default=.05)
    ap.add_argument("--sgldLr", type=float, required=False, help="the learning rate of SGLD",default=0.05)
    ap.add_argument("--sgldStd", type=float, required=False, help="the standard deviation of the noise in SGLD",default=5e-3)
    ap.add_argument("--bufferSamples", type=str, required=False, help="buffered data sample file", default='')
    ap.add_argument("--perturbThreshold", type=float, required=False, help="perturbation threshold during p(x_tilde|x)", default=0.05)
    ap.add_argument("--xWeight", type=float, required=False, help="weight for logp(x)", default=0.3)
    ap.add_argument("--clfWeight", type=float, required=False, help="weight for logp(y|x)", default=1)
    ap.add_argument("--xTildeWeight", type=float, required=False, help="weight for logp(x_tilde|x, y)", default=0.1)
    ap.add_argument("--bayesianTraining", type=bool, required=False, help="flag for Bayesian Adversarial Training", default=False)
    ap.add_argument("--bayesianModelNum", type=int, required=False, help="the number of model to train",
                    default=5)



def parameteSettingForClassifiers(ap):
    ap.add_argument("-path", "--dataPath", type=str, required=False, help="folder to load or save data")
    ap.add_argument("-retPath", "--retPath", type=str, required=False, help="folder to load or save data")
    ap.add_argument("-trainFile", "--trainFile", type=str, required=False, help="the training data file under --dataPath. "
                    "this is the training data when training the classifier, and the data samples to attack when "
                                                                      "attacking the classifier")
    ap.add_argument("-testFile", "--testFile", type=str, required=False, help="the test data file under --dataPath used for "
                                                                    "training the classifier")
    ap.add_argument("-trainedModelFile", "--trainedModelFile", type=str, required=False,
                    help="the pre-trained weight file, under --retPath", default='')

    ap.add_argument("-trainedAppendedModelFile", "--trainedAppendedModelFile", type=str, required=False,
                    help="the pre-trained appended model weight file, under --retPath", default='')

    ap.add_argument("-ep", "--epochs", type=int, required=False, help="to specify the number of epochs for training",
                    default=80)
    ap.add_argument("-cn", "--classNum", type=int, required=False, help="to specify the number of classes")
    ap.add_argument("-bs", "--batchSize", type=int, required=False, help="to specify the number of classes",
                    default=64)
    ap.add_argument("-adTrainFile", "--adTrainFile", type=str, required=False,
                    help="the file name of the correctly classified data samples for adversarial attack",
                    default='adClassTrain.npz')
    ap.add_argument("-optimiser", "--optimiser", type=str, required=False, help="choose the opimiser",default='SGD')

    #set up for STGCN, for running the original code
    ap.add_argument("-inputDim", "--inputDim", type=int, required=False, help="STGCN: the input channel number", default=3)
    ap.add_argument("-dropout", "--dropout", type=float, required=False, help="STGCN: the dropout rate",
                    default=0.5)
    ap.add_argument("-eiw", "--edge_importance_weighting", type=bool, required=False, help="STGCN: put weighting on edges",
                    default=True)
    ap.add_argument("-glayout", "--graph_layout", type=str, required=False, help="STGCN: the graph layout, specific to dataset",
                    default='ntu-rgb+d')
    ap.add_argument("-gstrategy", "--graph_strategy", type=str, required=False, help="STGCN: the graph strategy",
                    default='spatial')
    ap.add_argument("-baseClassifier", "--baseClassifier", type=str, required=False, help="STGCN: the graph strategy",
                    default='')



def parameteSettingForTASAR(ap):
    ap.add_argument("--lam", type=float, default=1e-3) 
    ap.add_argument("--swa_start", type=int, default=0)
    ap.add_argument("--swa_n", type=int, default=300)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--n_models", type=int, default=3)
    ap.add_argument("--flag1", type=bool, default=False)
    ap.add_argument("--flag2", type=bool, default=False)
    ap.add_argument("--temporal", type=bool, default=False)

if __name__ == '__main__':

    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("-r", "--routine", type=str, required=False, help="program to run",
                    choices=["train", "test", "gatherCorrectPrediction", "attack", "adTrain", "bayesianTrain", "bayesianTest","finetune","DualBayesian","AR"])
    ap.add_argument("-classifier", "--classifier", type=str, required=False, help="choose the classifier to train/test/attack",
                    )
    ap.add_argument("-dataset", "--dataset", type=str, required=False, help="choose the dataset",
                    choices=["hdm05", "ntu60", "ntu120"])

    parameteSettingForOptimizers(ap)
    parameteSettingForClassifiers(ap)
    parameteSettingForAttackers(ap)
    parameteSettingForAdTrainers(ap)
    parameteSettingForTASAR(ap)

  
    args = ap.parse_args()
    routine = args.routine
    
    if routine == 'train':
        classifier = loadClassifier(args)
        classifier.train()
    if routine == 'finetune':
        classifier = loadClassifier(args)
        classifier.finetune()

    elif routine == 'test':
        classifier = loadClassifier(args)
        classifier.test()

    elif routine == 'gatherCorrectPrediction':
        classifier = loadClassifier(args)
        classifier.collectCorrectPredictions()
    
    elif routine == 'bayesianTrain':
        adTrainer = loadTrainer(args)
        adTrainer.bayesianTrain()
        
    elif routine == 'DualBayesian':
        adTrainer = loadTrainer(args)
        adTrainer.Dual_Bayesian()

    elif routine == 'bayesianTest':
        classifier = loadClassifier(args)
        classifier.test()
        
    elif routine == 'adTrain':
        adTrainer = loadATrainer(args)
        if args.adTrainer == 'EBMATrainer':
            adTrainer.bayesianAdversarialTrain()
        else:
            adTrainer.adversarialTrain()
        
    elif routine == 'attack':   
        attacker = loadAttacker(args)
        attacker.attack()
    else:
        print('nothing happened')
