## Tutorials

This repository contains Python scripts to discover step by step the basic usage of DeepPhysX_Sofa. 

This tutorial only defines dummy objects so that users become familiar with the API and 
what is required to run training and prediction sessions.

### Description

Python scripts should be explored in the following order:
* **#01** `environment.py`: how to define a SofaEnvironment
* **#02** `network.py`: how to define a Network
* **#03** `configuration.py`: how to configure SofaEnvironment, Network and Dataset
* **#04** `runSofa.py`: how to run the SofaEnvironment as a SOFA scene in the GUI
* **#05** `dataGeneration.py`: how to only generate a Dataset
* **#06** `offlineTraining.py`: how to launch a training session with an existing dataset
* **#07** `onlineTraining.py`: how to generate a Dataset and launch a training session simultaneously
* **#08** `prediction.py`: how to launch a prediction session with a trained Network
