# TapWeight

The code is an implementation of the TapWeight framework.

The molreweight repo contains the code of applying TapWeight to molecule property prediction.

The nlureweight repo contains the code of applying TapWeight to natural language understanding.

## Molecule Property Prediction

The code is based on the Imagemol [repo](https://github.com/HongxinXiang/ImageMol). To run the scripts, please follow instructions in processing data in Imagemol and put data path into the script.

To run the TapWeight framework, use the reweight_cls.sh script for classification tasks and reweight_reg.sh for regression tasks. 

We also provided an implementation of bi-level optimization framework (reweight_blo.py)  mentioned in ablation study section.  

## Natural Language Understanding

The code is based on the SimCSE [repo](https://github.com/princeton-nlp/SimCSE). To run the scripts, please follow instructions in processing data in SimCSE and put data path into the script.

To run the TapWeight framework, use the main.sh script.

## Retraining

TapWeight is a task-adaptive pretraining framework, so it is necessary to perform finetuning after running TapWeight to achieve optimal performance. For the two application above, implementation of finetuning can be found in their original repo. 

For example, the Imagemol repo contains code and instructions on how to finetune a checkpoint on a custom downstream dataset. To conduct retraining for TapWeight, simply replace the Imagemol checkpoint with TapWeight checkpoint.   