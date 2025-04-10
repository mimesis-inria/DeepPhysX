Install
=======

Prerequisites
-------------

**DeepPhysX** packages have the following dependencies :

.. table::
    :widths: 20 20 10 30

    +-----------------------------+-------------------------------+--------------+------------------------------------------+
    | **Package**                 | **Dependency**                | **Type**     | **Install**                              |
    +=============================+===============================+==============+==========================================+
    | :guilabel:`DeepPhysX     `  | :Numpy:`Numpy <>`             | **Required** | ``pip install numpy``                    |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :PyTorch:`PyTorch <>`         | **Required** | ``pip install torch``                    |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :Tensorboard:`Tensorboard <>` | **Required** | ``pip install tensorboard``              |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :SSD:`SSD <>`                 | **Required** | :SSD:`Follow instructions <>`            |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :SimRender:`SimRender <>`     | **Required** | :SimRender:`Follow instructions <>`      |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :SOFA:`SOFA Framework <>`     | Optional     | :SOFAI:`Follow instructions <>`          |
    |                             +-------------------------------+--------------+------------------------------------------+
    |                             | :Caribou:`Caribou <>`         | Optional     | :CaribouI:`Follow instructions <>`       |
    +-----------------------------+-------------------------------+--------------+------------------------------------------+


Install
-------

Install with pip
""""""""""""""""

**DeepPhysX** packages are all registered on `PyPi <https://pypi.org/project/DeepPhysX>`_.
They can easily be installed with ``pip``:

.. code-block:: bash

    $ pip install git+https://github.com/mimesis-inria/DeepPhysX.git

Then, you should be able to run:

.. code-block:: bash

    $ pip show DeepPhysX

.. code-block:: python

    import DeepPhysX

Install from sources
""""""""""""""""""""

**DeepPhysX** can also be installed from sources for developers:

.. code-block:: bash

    $ git clone https://github.com/mimesis-inria/DeepPhysX.git
    $ cd DeepPhysX
    $ pip install -e .
