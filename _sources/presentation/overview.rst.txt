Overview
========

.. _overview-packages:

Packages
--------

.. role:: core
.. role:: simu
.. role:: ai

The **DeepPhysX** project is divided into several **Python packages**: a :core:`CORE` package which is able to
communicate with a :simu:`SIMULATION` package and an :ai:`AI` package.
This way, the :core:`CORE` has no dependencies neither to a :simu:`SIMULATION` framework nor an :ai:`AI` framework,
and any of those frameworks can be compatible with **DeepPhysX**.

.. figure:: ../_static/image/overview_packages.png
    :alt: overview_packages.png
    :align: center

    Package organization

Core
""""

This package rules the **data flow** and the **communication** between the AI components and the simulation components.
Data flow also involves the **storage** and **loading** of datasets and neural networks, the **management** of the
visualization tool and the **analysis** of the training sessions.
This package is named :guilabel:`DeepPhysX.Core`.

.. admonition:: Dependencies

    NumPy, Tensorboard, Vedo, SimulationSimpleDatabase

Simulation
""""""""""

A **simulation package** provides a **DeepPhysX** compatible API for a **specific simulation framework**.
For **DeepPhysX**, each simulation package dedicated to a specific simulation framework (written in python or providing
python bindings).
Thus, each package will be named according to the framework with a common prefix, for instance
:guilabel:`DeepPhysX.Simulation`.

.. admonition:: Available simulation packages

    :guilabel:`DeepPhysX.Sofa` designed for :SOFA:`SOFA <>`

AI
""

An **AI package** provides a **DeepPhysX** compatible API for a **specific AI framework**.
In the same way, each learning package is dedicated to a specific AI Python framework.
Thus, each package will be named according to the framework with the same common prefix, for instance
:guilabel:`DeepPhysX.AI`.

.. admonition:: Available learning packages

    :guilabel:`DeepPhysX.Torch` designed for :PyTorch:`PyTorch <>`


Architecture
------------

This section describes both the links between the components of :core:`CORE` package and the links to :ai:`AI` and
:simu:`SIMULATION` packages.

Users might use one of the provided *Pipelines* for their **data generation**, their **training session** or their
**prediction session**.
These *Pipelines* trigger a **loop** which defines the number of samples to produce, the number of epochs to perform
during a training session or the number of steps of prediction.

.. figure:: ../_static/image/overview_components.png
    :alt: overview_architecture.png
    :width: 80%
    :align: center

    Components architecture

The *Pipeline* will involve several components (data producers and data consumers), but the *Pipeline* will always
communicate with their *Manager* first.
A main *Manager* will provide the *Pipeline* an intermediary with all the existing *Managers*:

:``DatabaseManager``: It will manage the *Database* component to create **storage** partitions, to fill these partitions
 with the synthetic training data produced by the *Environment* and to **reload** an existing *Database* for training or
 prediction sessions.

 .. note::
    If training and data generation are done simultaneously (by default for the training *Pipeline*), the *Database*
    can be built only during the first epoch and then reloaded for the remaining epochs.

:``EnvironmentManager``: It will manage the *Environment* (the numerical simulation) component to **create** it, to
 trigger **steps** of simulations, to **produce** synthetic training data, to provide **predictions** of the network
 if required, and to finally **shutdown** the *Environment*.

 .. note::
    This *Manager* can communicate directly with a single *Environment* or with a *Server* which shares information
    with several *Environments* in multiprocessing launched as *Clients* through a custom TCP-IP protocol (see
    :ref:`dedicated section <environment-tcpip>`).

.. note::
    The two above *Managers* are managed by the ``DataManager`` since both the *Environment* and the *Database*
    components provide training data to the *Network*.
    This ``DataManager`` is the one who decides if data should be requested from the *Environment* or from the
    *Database* depending on the current state of the *Pipeline* and on the components configurations.

:``NetworkManager``: It will manage several objects to **train** or **exploit** your *Network*:

 * The *Network* to produce a **prediction** from an input, to **save** a set of parameters or to **reload** one.
 * The *Optimizer* to compute the **loss** value and to **optimize** the parameters of the *Network*.
   This component uses existing loss functions and optimizers in the chosen AI framework.
 * The *DataTransformation* to **convert** the type of training data sent from *Environment* to a compatible type for
   the AI framework you use and vice versa, to **transform** training data before a prediction, before the loss
   computation and before sending the prediction to the *Environment*.

 .. note::
    The above components are designed to be easily inherited and upgradable if the content of AI packages is not enough.
    Users are thus free to define their own *Network* architecture, to create a custom loss or optimizer to feed the
    *Optimizer* and to compute the required tensor transformations in the *DataTransformation*.

:``StatsManager``: It will manage the **analysis** of the **evolution** of a training session.
 These analytical data will be saved in an **event log file** interpreted by :Tensorboard:`Tensorboard <>`.

 .. note::
    Usual curves will be automatically provided in the board (such as the evolution of the loss value, the smoothed
    mean and the variance of this loss value per batch and per epoch), but other custom fields can be added and filled
    as well.

.. warning::
    It is not possible to use the default *Network* and *Environment* provided in the :core:`CORE` package, since they
    are not implemented at all.
    The reason is that you need to choose an :ai:`AI` and a :simu:`SIMULATION` Python framework to implement them.
    The aim of **DeepPhysX** additional packages is to provide a compatible implementation both for **DeepPhysX**
    and these frameworks.

    Example
     If you choose :PyTorch:`PyTorch <>` as your :ai:`AI` framework, you can use or implement a *TorchNetwork* which
     inherits from both the :core:`CORE` *Network* and ``Torch.nn.module`` (available in :guilabel:`DeepPhysX.Torch`).

     If you choose :SOFA:`SOFA <>` as your :simu:`SIMULATION` framework, you can implement a *SofaEnvironment* which
     inherits from both the :core:`CORE` *Environment* and ``Sofa.Core.Controller`` (available in
     :guilabel:`DeepPhysX.Sofa`).
