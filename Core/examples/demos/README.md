## Demos

This repository contains higher level applications. 
The understanding of these sessions will be easier after studying the content of "Tutorial" and "Features" repositories.

> :warning: **PyTorch usage**: This working session uses the Fully Connected implementation from DeepPhysX_Torch 
package, please make sure it is installed if you want to run training scripts.

> Note: Trained networks and training data where produced with available demos in **DeepPhysX_Sofa** package. 

### Armadillo, Beam, Liver

The objective of these three sessions (Armadillo, Beam & Liver) is to enable a neural network to predict deformations 
from forces applied on the object.

Each demo will contain the following scripts:
* `download.py`: Allow to automatically download the missing demo data. Launching this script is not mandatory.
* `prediction.py`: Launch the prediction session with `Environment/<Demo>.py`.
* `interactive.py`: Launch an interactive prediction session with `Environment/<Demo>Interative.py`.

### Description

#### FEM models

The simulated objects have a different **hyper-elastic material** depending on the session (Saint Venant–Kirchhoff or 
Neo-Hookean, with different coefficients). 
They are subject to **static** deformations imposed by forces applied to their surface.
Given a specific external force applied to the surface mesh of the object, the corresponding sparse grid will be
deformed with regard to the associated physical law.

#### Dataset

The goal is to train a **neural network** to replace the **static solver** of the simulation, ie predicting 
deformations from external forces.
Training data were produced with a large number of sample from the FEM simulation.

Then, the **inputs** of the network are the **external surface forces**, while the associated **ground truths** are the
resulting **volume deformations**.

Training data will not have the same dimension depending on the network architecture:
* With a **Fully Connected** network, input and output data must be passed as raw data (given an object with *n* surface 
  nodes, the forces vector will be reshaped from (*n*, 3) to (3*n*, 1); given an object with *m* volume nodes, the 
  displacement vector will be reshaped from (*m*, 3) to (3*m*, 1)).
* With a **UNet** network, input and output data must be passed as an n-dimensional grid. Then, the object is immersed 
  in a larger regular grid to interpolate the forces vector dans the displacement vector on it.

#### Prediction session

Once the neural network is trained, it will be able to **predict the deformations** of the object given an external 
force.
First, external forces are randomly computed and transformed to match the network input shape (either reshaped to a raw 
vector, either interpolated to a regular grid).
Then, the network predicts a deformation which is transformed to match the sparse grid shape (either reshaped from raw 
vector, either interpolated from regular grid).
Finally, this displacement vector will be applied on the rest position of the sparse grid to visualize the corresponding 
deformation.
