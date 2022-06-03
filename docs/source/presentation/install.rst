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

    +-----------------------------+-------------------------------+--------------+-------------------------------------+
    | **Package**                 | **Dependency**                | **Type**     | **Install**                         |
    +=============================+===============================+==============+=====================================+
    | :guilabel:`DeepPhysX_Core`  | :Numpy:`Numpy <>`             | **Required** | ``pip install numpy``               |
    |                             +-------------------------------+--------------+-------------------------------------+
    |                             | :Vedo:`Vedo <>`               | **Required** | ``pip install vedo``                |
    |                             +-------------------------------+--------------+-------------------------------------+
    |                             | :Tensorboard:`Tensorboard <>` | **Required** | ``pip install tensorboard``         |
    +-----------------------------+-------------------------------+--------------+-------------------------------------+
    | :guilabel:`DeepPhysX_Sofa`  | :SOFA:`SOFA Framework <>`     | **Required** | :SOFAI:`Follow instructions <>`     |
    |                             +-------------------------------+--------------+-------------------------------------+
    |                             | :SP3:`SofaPython3 <>`         | **Required** | :SP3I:`Follow instructions <>`      |
    |                             +-------------------------------+--------------+-------------------------------------+
    |                             | :Caribou:`Caribou <>`         | Optional     | :CaribouI:`Follow instructions <>`  |
    +-----------------------------+-------------------------------+--------------+-------------------------------------+
    | :guilabel:`DeepPhysX_Torch` | :PyTorch:`PyTorch <>`         | **Required** | ``pip install torch``               |
    +-----------------------------+-------------------------------+--------------+-------------------------------------+

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

    pip3 install DeepPhysX
    pip3 install DeepPhysX_Sofa
    pip3 install DeepPhysX_Torch

Then, you should be able to run:

.. code-block:: bash

    pip3 show DeepPhysX
    pip3 show DeepPhysX_Sofa
    pip3 show DeepPhysX_Torch

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
    $ git clone https://github.com/mimesis/deepphysx.git Core
    $ cd Core

Specify which packages to install by running the configuration script.
This way, all the packages are gathered in a single installation.

.. code-block:: bash

    $ python3 config.py
    >   Available AI packages : ['Torch']
    >    >> Installing package Torch (y/n): yes
    >
    >   Available Simulation packages : ['Sofa']
    >    >> Installing package Sofa (y/n): yes
    >
    >   Applying following configuration:
    >     * DeepPhysX.Core:  True (default)
    >     * DeepPhysX.Torch: True
    >     * DeepPhysX.Sofa:  True
    >   Confirm (y/n): yes
    >   Configuration saved in 'config.json'


.. note::
    Configuration script will **automatically clone** missing packages.

Finally, install the defined packages:

* by using ``pip`` to install and manage them as non-editable

  .. code-block:: bash

      $ pip3 install .

* by running ``dev.py`` to link them as editable in the site-packages

  .. code-block:: bash

      $ python3 dev.py set

Then, you should be able to run:

.. code-block:: bash

    # If installed with pip
    $ pip show DeepPhysX

.. code-block:: python

    # In both cases
    import DeepPhysX.Core
    import DeepPhysX.Sofa
    import DeepPhysX.Torch
