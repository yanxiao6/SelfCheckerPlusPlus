# SelfChecker++

Code release of the paper "Self-Checking Deep Neural Networks for Anomalies and Adversaries in Deployment" submitted to IEEE Transactions on Dependable and Secure Computing.

## Introduction

This work is extended from our previous work SelfChecker. This paper presents a novel self-checking system, SelfChecker++, to address the challenges raised by both unintended anomalies and intentional adversaries. Similar to SelfChecker, SelfChecker++ (1) triggers an alarm if most internal layer features of the model are inconsistent with the final prediction, and (2) provides advice in the form of an alternative prediction. However, unlike SelfChecker, SelfChecker++ (1) relaxes the assumption of SelfChecker that the training and validation datasets come from a distribution similar to that of the inputs that the DNN model will face in deployment, and (2) makes the model immune to adversarial attacks in which an adversary crafts adversarial inputs. We designed SelfChecker++ by introducing a GAN-based (Generative Adversarial Network) transformation technique to learn the distribution of the training data to differentiate samples from a different distribution, and synthesize a sample from the latent space (avoiding the processing of any potential adversarial samples). 

## Repo structure

- `utils.py` - Util functions for log.
- `main_kde.py` - Obtain density functions for the combination of classes and layers and inferred classes.
- `kdes_generation.py` - Contain functions for generating density functions and inferred classes.
- `layer_selection_agree.py` - Layer selection for alarm.
- `layer_selection_condition.py` - Layer selection for advice.
- `layer_selection_condition_neg.py` - Layer selection for advice.
- `sc.py` - Alarm and advice analysis.
- `gan_trans.py` - GAN-based transformation to generate new input given original one triggering an alarm.
- `models/` - Folder contains pre-trained models.
- `tmp/` - Folder saving density functions and inferred classes.

## Dependencies
```bash
conda env create -f sc.yml
conda activate sc
```

## How to run

### SelfChecker
- We prepare a pre-trained model ConvNet on CIFAR-10: python sc.py
- To run the whole project: bash exe.sh

### GAN-based transformation
```bash
mkdir -p classifier/{cifar10/{conv,resnet},mnist/{conv,resnet},fmnist/{conv,resnet}}
```
Ensure all conv and resnet sub directories in classifier has the following three files:
- Respective Classifier Model (e.g. model_cifar10_conv.h5)
- Respective False Positive Index (e.g. FP_idx.npy)
- Respective True Positive Index (e.g. TP_idx.npy)
```bash
nohup python3 -u gan_trans.py &> run_name.out & # run_name refers to run_name specified in gan_trans.py's trans_config dictionary
```
>Before running gan_trans.py, you are able to specify the required config in gan_trans.py's trans_config dictionary





