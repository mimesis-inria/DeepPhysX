## Tutorial

This repository is a first simple example where we train a neural network to predict the behavior of a mass-spring 
system, depending on the stiffness of the spring, the mass of the cube and the friction. 

> :warning: **PyTorch usage**: This session uses the Fully Connected implementation from DeepPhysX.Torch 
package, make sure it is installed if you want to run training scripts.


### Description

Python scripts should be explored in the following order:
* **#00** `scenario.py`: an interactive simulation to understand the context (out of the scope of DPX)
* **#01** `environment.py`: the definition of our training Environment
* **#02** `configuration.py`: the configuration of our Environment, Network architecture and Database
* **#03** `dataset.py`: the dataset generation
* **#04** `training.py`: the training session from the generated dataset
* **#05** `prediction.py`: the prediction session with a trained Network
* **#06** `interactive.py`: 
