# Skeleton-Robustness-Benchmark(RobustBenchHAR)

RobustBenchHAR is a pytorch framework to boost and evaluate the adversarial transferability for Human skeletal behavior recognition. Official code for the paper:

> **ICLR2025 TASAR: Transfer-based Attack on Skeletal Action Recognition**

<div style="text-align: center;">
<img src="./figs/highlevel.png" alt="High-Level" style="width: 80%; display: inline-block;">
</div>

Key Features of RobustBenchHAR:
+ **A benchmark for evaluating existing transfer-based attacks in human Activity Recognition (HAR)**: RobustBenchHAR ensembles existing transfer-based attacks including several types and fairly evaluates various transfer-based attacks under the same setting.
+ **Evaluate the robustness of various models and datasets**: RobustBenchHAR provides a plug-and-play interface to verify the robustness of models on different data sets.
+ **A summary of transfer-based attacks and defenses**: RobustBenchHAR reviews numerous transfer-based attacks and adversarial defenses, making it easy to get the whole picture of transfer-based attacks for practitioners.

## Usage
The detailed benchmark setting are available in "TASAR: Transfer-based Attack on Skeletal Action Recognition". The complete usage is as follows:

**For any action on the ensemble models, add parameters “--ensemble True”**

## Train

```
python main.py -classifier STGCN --routine train --dataset hdm05 --trainFile classTrain.npz --testFile classTest.npz --dataPath ../data/ --retPath ../results/ -ep 80 -cn 65 -bs 64 -lr 0.1
```

Training the classifier will generate two models by default: the minimal loss model minLossModel.pth and the minimal validation loss model minValLossModel.pth under --result

## Test

```
python main.py -classifier STGCN --routine test --dataset hdm05 --testFile classTest.npz --trainedModelFile minValLossModel.pth --dataPath ../data/ -retPath ../results/ -cn 65 -bs 64
```

## collect all the correctly recognised samples for attack

```
python main.py -classifier STGCN --routine gatherCorrectPrediction --dataset hdm05 --testFile classTest.npz --trainedModelFile minValLossModel.pth --dataPath ../data/ -retPath ../results/ -cn 65
```

This process by default generates a file called adClassTrain.npz under --retFolder, which contains all the correctly recognised samples for attack. If acting on the ensemble models:

```
python main.py -classifier STGCN,CTRGCN,MSG3D --routine gatherCorrectPrediction --dataset hdm05 --testFile classTest.npz --trainedModelFile minValLossModel.pth --dataPath ../data/ -retPath ../results/ -cn 65 --ensemble True
```

## Attack(untarget by default)

## Gradient-based

1. **[I-FGSM](https://arxiv.org/abs/1412.6572)**,  **[MI-FGSM](https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf)**, **[SMART](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Understanding_the_Robustness_of_Skeleton-Based_Action_Recognition_Under_Adversarial_Attack_CVPR_2021_paper.pdf)**, **[MIG](https://openaccess.thecvf.com/content/ICCV2023/papers/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.pdf)**, **[CIASA](https://ieeexplore.ieee.org/abstract/document/9302639)**

```
python main.py -classifier STGCN --routine attack -attacker [FGSM, MI-FGSM, SMART, MIG, CIASA] --trainedModelFile minValLossModel.pth -cp 0.01 --dataset hdm05 --trainFile adClassTrain.npz --testFile classTest.npz --dataPath ../results/ -retPath ../results/ -cn 65 --epochs 200 --batchSize 16 --attackType untarget
```

1. **[I-FGSM](https://arxiv.org/abs/1412.6572)**

```
python main.py -classifier STGCN --routine attack -attacker FGSM --trainedModelFile minValLossModel.pth -cp 0.01 --dataset hdm05 --trainFile adClassTrain.npz --testFile classTest.npz --dataPath ../results/ -retPath ../results/ -cn 65 --epochs 200 --batchSize 16 --attackType untarget
```


## Input transformation-based

1. **[DIM](https://arxiv.org/abs/1803.06978)**

```
python main.py -classifier STGCN --routine attack -attacker Augment --trainedModelFile minValLossModel.pth -cp 0.01 --dataset hdm05 --trainFile adClassTrain.npz --testFile classTest.npz --dataPath ../results/ -retPath ../results/ -cn 65 --epochs 200 --batchSize 16 --attackType untarget
```

## Ensemble-based

1. **[ENSEMBLEA](https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf)**

```
python main.py -classifier STGCN,CTRGCN,MSG3D --routine attack -attacker ENSEMBLE --trainedModelFile minValLossModel.pth -cp 0.01 --dataset hdm05 --trainFile adClassTrain.npz --testFile classTest.npz --dataPath ../results/ -retPath ../results/ -cn 65 --epochs 200 --batchSize 16 --attackType untarget --ensemble True
```

2. **[SVRE](https://arxiv.org/pdf/2111.10752)**

```
python main.py -classifier STGCN,CTRGCN --routine attack -attacker SVRE --trainedModelFile minValLossModel.pth -cp 0.01 --dataset hdm05 --trainFile adClassTrain.npz --testFile classTest.npz --dataPath ../results/ -retPath ../results/ -cn 65 --epochs 200 --batchSize 16 --attackType untarget --ensemble True
```

3. **[BA](https://arxiv.org/pdf/2302.05086)**
   Before starting the attack, you first need to fine-tune to get the surrogate model by:

```
python main.py -classifier STGCN --routine finetune --dataset hdm05 --trainFile classTrain.npz --testFile classTest.npz --dataPath ../data/ --retPath ../results/ -ep 20 -cn 65 -bs 64 -lr 0.01
```

Then attack by:

```
python main.py -classifier STGCN --routine attack -attacker BA --trainedModelFile minValLossModel.pth -cp 0.01 --dataset hdm05 --trainFile adClassTrain.npz --testFile classTest.npz --dataPath ../results/ -retPath ../results/ -cn 65 --epochs 200 --batchSize 16 --attackType untarget
```

4. **TASAR**

You first need to post-train the pre-trained model by:

```
python main_TASAR.py -classifier ExtendedBayesian --baseClassifier STGCN --routine bayesianTrain --dataset hdm05 -adTrainer PDBATrainer --trainFile classTrain.npz --testFile classTest.npz --dataPath ../data/ --retPath ../results/ -ep 5 -cn 65 --trainedModelFile minValLossModel.pth -bs 32 -lr 2e-3 --bayesianModelNum 3
```

Then performethe The Dual Bayesian sampling in post-train models

```
python main_TASAR.py -classifier ExtendedBayesian --baseClassifier STGCN --routine DualBayesian --dataset hdm05 -adTrainer PDBATrainer --trainFile classTrain.npz --testFile classTest.npz --dataPath ../data/ --retPath ../results/ -ep 5 -cn 65 --trainedModelFile minValLossModel.pth -bs 32 -lr 2e-3 --bayesianModelNum 3 --trainedAppendedModelFile yes
```

Meanwhile, You also need to calculate the timing parameters of the data for **motion gradient**.

```
python main_TASAR.py -classifier ExtendedBayesian --baseClassifier STGCN --routine AR --attacker TASAR --trainedModelFile minValLossModel.pth --adTrainer PDBATrainer --updateRule Adam --dataset hdm05 --trainFile adClassTrain.npz --testFile classTest.npz --dataPath ../data/ -retPath ../results/ -cn 65 --epochs 200 --batchSize 8 --trainedAppendedModelFile yes --bayesianModelNum 3
```

Fianlly, attack by:

```
python main_TASAR.py -classifier ExtendedBayesian --baseClassifier STGCN --routine attack --attacker TASAR --trainedModelFile minValLossModel.pth --adTrainer PDBATrainer --updateRule Adam --dataset hdm05 --trainFile adClassTrain.npz --testFile classTest.npz --dataPath ../data/ -retPath ../results/ -cn 65 --epochs 200 --batchSize 8 --trainedAppendedModelFile yes --bayesianModelNum 3  -cp 0.01 --attackType untarget --temporal True

```

## Defense by Adversarial Training

1. **[TRADES](https://proceedings.mlr.press/v97/zhang19p.html)**

```
python main.py -classifier STGCN --routine adTrain --dataset hdm05 --trainFile classTrain.npz --testFile classTest.npz --dataPath ../data/ --retPath ../results/ -ep 80 -cn 65 -bs 64 -lr 0.1 --adTrainer TRADES 
```

2. **[BEAT](https://ojs.aaai.org/index.php/AAAI/article/view/25352)**

```
python main.py -classifier ExtendedBayesian --baseClassifier STGCN --routine adTrain --dataset hdm05 --trainFile classTrain.npz --testFile classTest.npz --dataPath ../data/ --retPath ../results/ -ep 80 -cn 65 -bs 64 -lr 0.1 --adTrainer EBMATrainer
```

## Transfer-based attack test

When the attack is complete, the generated adversarial examples will be saved in the folder corresponding to the attack name, you can execute transfer-based attack test.

```
python main.py -classifier STGCN --routine test --transfer_attack True --dataset ntu60 --trainedModelFile minValLossModel.pth --dataPath ../data/ -retPath ../results/ -cn 60 -bs 16 --transfer_path {adv_examples_path}
```

## Warning
The code has not been exhaustively tested. You need to run it at your own risk. The author will try to actively maintain it and fix reported bugs but this can be delayed.

## Citation
```
@article{tasar,
  title={TASAR: Transfer-based Attack on Skeletal Action Recognition},
  author={Diao, Yunfeng and Wu, Baiqi and Zhang, Ruixuan and Liu, Ajian and Wei, Xingxing and Wang, Meng and Wang, He},
  journal={arXiv preprint arXiv:2409.02483},
  year={2024}
}
```
