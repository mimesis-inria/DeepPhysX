About
=====

Features
--------

The purpose of the **DeepPhysX** framework is to provide an interface between **Deep Learning algorithms** and
**numerical simulations**.

This **full Python3** project brings several **pipelines**, allowing the user to:

 * **Generate a dataset** with synthetic data from numerical simulations;
 * **Train an artificial neural network** with a synthetic dataset;
 * **Use the predictions of a trained network** inside a numerical simulation.

.. note::
    The **dataset generation** and the **training** pipelines can be done **simultaneously**.

.. list-table:: Classical pipelines (left), DeepPhysX pipelines (right)
    :widths: 34 66

    * - .. image:: ../_static/image/about_classical_pipelines.png
            :alt: about_classical_pipelines.png
      - .. image:: ../_static/image/about_dpx_pipelines.png
            :alt: about_dpx_pipelines.png

About **dataset** components, the project has the following features:

 * Automatic training dataset **storage** and **loading** with **multiple files management**;
 * Dataset **shuffle** and **normalization**;
 * Multiple dataset **modes**: Training, Validation, Prediction;
 * **Customizable** dataset fields.

About **simulation** components, the project has the following features:

 * **Data generation** achieved by several simulations running in **multiprocessing** with a client-server architecture;
 * Operation with **internal data**, from the **dataset** or from the **neural network**;
 * Increased **interactions** with other components (dataset, neural network, visualizer);
 * Check the **validity** of the training **data**.

About **neural network** components, the project has the following features:

 * Automatic **storage** and **loading** of networks during training;
 * **Customizable data transformations** at each step (forward pass, optimization, prediction apply);
 * **Customizable optimization process** with training data;
 * Already **implemented architectures**: FC, UNet.

**Additional tools** are also provided:

 * A **visualization** Factory to **init**, **update** and **render** the simulated objects (written with
   :Vedo:`Vedo <>`);
 * An customizable analysis of the **evolution** of the **training session** (written with
   :Tensorboard:`Tensorboard <>`).


.. _about-working-sessions:

Working sessions
----------------

A **DeepPhysX** pipeline is associated with a **working session**. The repository is **automatically managed** and
typically contains the following **tree structure**:

 :``Dataset``: Contains the **dataset partitions** and a **json description file**;
 :``Network``: Contains the saved **neural networks**;
 :``Stats``: Contains a **Tensorboard file** which gathers **training analysis**;
 :``info.txt``: A **text description file** which gathers the **configurations** of all the **involved components**.

.. list-table:: Working session repository
    :width: 80%
    :align: center

    * - .. image:: ../_static/image/about_tree.png
            :alt: about_working_session.png
