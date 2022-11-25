Install
=======

.. role:: core
.. role:: simu
.. role:: ai

It is recommended to read the :ref:`overview-packages` section first to understand the packages organisation.

| The :core:`CORE` package :guilabel:`DeepPhysX.Core` is always installed by default.
| Using a :simu:`SIMULATION` package is recommended but not required.
| At least one :ai:`AI` package is mandatory (default one is :guilabel:`DeepPhysX.Torch`).


Prerequisites
-------------

**DeepPhysX** packages have the following dependencies :

.. table::
    :widths: 20 20 10 30

    +-----------------------------+-------------------------------+--------------+------------------------------------------+
    | **Package**                 | **Dependency**                | **Type**     | **Install**                              |
    +=============================+===============================+==============+==========================================+
    | :guilabel:`DeepPhysX_Core`  | :Numpy:`Numpy <>`             | **Required** | ``pip install numpy``                    |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :Vedo:`Vedo <>`               | **Required** | ``pip install vedo``                     |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :Tensorboard:`Tensorboard <>` | **Required** | ``pip install tensorboard``              |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :SSD:`SSD <>`                 | **Required** | ``pip install SimulationSimpleDatabase`` |
    +-----------------------------+-------------------------------+--------------+------------------------------------------+
    | :guilabel:`DeepPhysX_Sofa`  | :SOFA:`SOFA Framework <>`     | **Required** | :SOFAI:`Follow instructions <>`          |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :SP3:`SofaPython3 <>`         | **Required** | :SP3I:`Follow instructions <>`           |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :Caribou:`Caribou <>`         | Optional     | :CaribouI:`Follow instructions <>`       |
    +-----------------------------+-------------------------------+--------------+------------------------------------------+
    | :guilabel:`DeepPhysX_Torch` | :PyTorch:`PyTorch <>`         | **Required** | ``pip install torch``                    |
    +-----------------------------+-------------------------------+--------------+------------------------------------------+

.. note::
    :guilabel:`DeepPhysX.Sofa` has a dependency to :Caribou:`Caribou <>` to run the demo scripts from
    ``Examples/SOFA/Demo`` since implemented simulations involve some of its components.

Install
-------

Install with pip
""""""""""""""""

**DeepPhysX** packages are all registered on `PyPi <https://pypi.org/project/DeepPhysX>`_.
They can easily be installed with ``pip``:

.. code-block:: bash

    $ pip3 install DeepPhysX
    $ pip3 install DeepPhysX.Sofa
    $ pip3 install DeepPhysX.Torch

Then, you should be able to run:

.. code-block:: bash

    $ pip3 show DeepPhysX
    $ pip3 show DeepPhysX.Sofa
    $ pip3 show DeepPhysX.Torch

.. code-block:: python

    import DeepPhysX.Core
    import DeepPhysX.Sofa
    import DeepPhysX.Torch

Install from sources
""""""""""""""""""""

**DeepPhysX** packages must have the following architecture.
These detailed instructions will build such an installation:

.. code-block::

    DeepPhysX
     ├── Core
     ├── Sofa
     └── Torch


Start by cloning the **DeepPhysX** source code from its Github repository in a directory named ``DeepPhysX/Core``:

.. code-block:: bash

    $ mkdir DeepPhysX
    $ cd DeepPhysX
    $ git clone https://github.com/mimesis-inria/DeepPhysX.git Core

Then, you can add compatibility layers to your **DeepPhysX** environment and install packages:

* **Option 1 (recommended):** run one of the ``setup_.py`` scripts that handle the installation of all packages

    * Use ``setup_user.py`` to install and manage packages with ``pip`` as non-editable.

      .. code-block:: bash

            $ python3 setup_user.py

    * Use ``setup_dev.py`` to link packages in the site-packages.

      .. code-block:: bash

            $ python3 setup_dev.py set

  .. note::
      Both scripts will asks the packages to install and will **automatically clone** missing packages.

* **Option 2:** clone the corresponding Github repositories in the created ``DeepPhysX`` directory, then install
  packages manually.

  .. code-block:: bash

        # Clone compatibility layers
        $ git clone https://github.com/mimesis-inria/DeepPhysX.Sofa.git Sofa
        $ git clone https://github.com/mimesis-inria/DeepPhysX.Torch.git Torch

        # Install packages manually
        $ cd Core ; pip3 install .
        $ cd ../Sofa ; pip3 install .
        $ cd ../Torch ; pip3 install .

Finally, you should be able to run:

.. code-block:: bash

    # If installed with pip
    $ pip3 show DeepPhysX
    $ pip3 show DeepPhysX.Sofa
    $ pip3 show DeepPhysX.Torch

.. code-block:: python

    # In both cases
    import DeepPhysX.Core
    import DeepPhysX.Sofa
    import DeepPhysX.Torch
