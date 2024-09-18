## Features review

This repository contains examples focused on training a Fully Connected Network to predict the mean value of a vector. 
The same application is implemented in CORE examples, this is the SOFA compatible version of the application.
These examples illustrate the main features of DeepPhysX framework.

> :warning: **PyTorch usage**: This working session uses the Fully Connected implementation from DeepPhysX_Torch 
package, please make sure it is installed if you want to run training scripts.

### Content

Examples of a SofaEnvironment: how to create it, how to send and receive data, how to initialize and update 
visualization data, how to send requests.
* `Environment/EnvironmentSofa.py`: SofaEnvironment compatible with SOFA GUI.
                                    Create scene graph and define behavior.
* `Environment/EnvironmentDataset.py`: SofaEnvironment compatible with DataGeneration pipeline.
                                       Initialize and update visualization data.
* `Environment/EnvironmentTraining.py`: SofaEnvironment compatible with Training pipeline.
                                        Apply predictions, initialize and update visualization data.
* `Environment/EnvironmentPrediction.py`: SofaEnvironment compatible with SOFA GUI and Prediction pipeline.
                                          Apply predictions.

Example of SOFA GUI running: how to create a SofaEnvironment and launch the SOFA GUI.
* `runSofa.py`: Launch the SofaEnvironment in the SOFA GUI.

Example of Dataset generation: how to create the pipeline, how to use multiprocessing to speed up the data production.
* `dataGeneration.py`: Launch the data generation pipeline.

Examples of Training sessions: how to create and launch the pipeline.
* `offlineTraining.py`: Train the Network with an existing Dataset.
* `onlineTraining.py`: Train the Network while producing the Dataset simultaneously and visualizing predictions.

Examples of Prediction sessions: how to create and launch the pipeline.
* `prediction.py`: Run predictions from trained Network in the Environment.
