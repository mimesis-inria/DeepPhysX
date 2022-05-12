## Features review

This repository contains examples focused on training a Fully Connected Network to predict the mean value of a vector. 
These examples illustrate the main features of DeepPhysX framework.

> :warning: **PyTorch usage**: This working session uses the Fully Connected implementation from DeepPhysX_Torch 
package, please make sure it is installed if you want to run training scripts.

### Content

Examples of an Environment: how to create it, how to send and receive data, 
how to initialize and update visualization data, how to send requests.
* `Environment.py`: Implementation of the Data producer

Examples of Dataset generation: how to create the pipeline, how to use multiprocessing 
to speed up the data production.
* `dataGeneration_single.py`: Launch the data generation pipeline.
* `dataGeneration_multi.py`: Compare data production between single process and multiprocess.

Examples of Training sessions: how to create and launch the pipeline.
* `offlineTraining.py`: Train the Network with an existing Dataset.
* `onlineTraining.py`: Train the Network while producing the Dataset simultaneously and visualizing predictions.

Examples of Prediction sessions: how to create and launch the pipeline.
* `prediction.py`: Run predictions from trained Network in the Environment.
* `gradienDescent.py`: A funny example to visualize the gradient descent process.
