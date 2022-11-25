## DeepPhysX

![logo](docs/source/_static/image/logo.png)

### Interfacing AI with simulation

The **DeepPhysX** project provides Python packages allowing users to easily interface their **numerical simulations**
with **learning algorithms**.

**DeepPhysX** provides a **Core** package with no dependency on any **simulation** or **AI** framework.
Then other packages are compatible with this **Core** and a specific **simulation** or **AI** framework:
* [DeepPhysX.Sofa](https://github.com/mimesis-inria/DeepPhysX_Sofa) as simulation packages;
* [DeepPhysX.Torch](https://github.com/mimesis-inria/DeepPhysX_Torch) as AI packages.


### Features

**DeepPhysX** is a full Python3 projects with 3 main features:
* Generate a dataset with synthetic data from numerical simulations;
* Train an artificial neural network with a synthetic dataset;
* Use the predictions of a trained network in a numerical simulation.

The full list of features is detailed in the [**documentation**](https://deepphysx.readthedocs.io).


### Quick install

The project was initially developed using [SOFA](https://www.sofa-framework.org/) as the **simulation package** and
[PyTorch](https://pytorch.org/) as the **AI framework**. 
Thus, **DeepPhysX** is mainly designed for these frameworks, but obviously **other frameworks** can also be used.
The packages corresponding to these frameworks will therefore be used for the default installation.

The easiest way to install is using `pip`, but there are a several way to install and configure a **DeepPhysX**
environment (refer to the [**documentation**](https://deepphysx.readthedocs.io/en/latest/presentation/install.html) 
for further instructions).

``` bash
$ pip install DeepPhysX             # Install default package
$ pip install DeepPhysX.Sofa        # Install simulation package
$ pip install DeepPhysX.Torch       # Install AI package
```


### Demos

**DeepPhysX** includes a set of detailed tutorials, examples and demos.
As these scripts are producing data, they cannot be run in the python site-packages, thus they should be run locally.
Use the *command line interface* to get the examples or to run **interactive demos**:

``` bash
$ DPX --get             # Get the full example repository locally
$ DPX --run <demo>      # Run one of the demo scripts
```

|          **Armadillo**<br>`DPX -r armadillo`          |          **Beam**<br>`DPX -r beam`          |          **Liver**<br>`DPX -r liver`          |
|:-----------------------------------------------------:|:-------------------------------------------:|:---------------------------------------------:|
| ![armadillo](docs/source/_static/image/armadillo.png) | ![beam](docs/source/_static/image/beam.png) | ![liver](docs/source/_static/image/liver.png) |
