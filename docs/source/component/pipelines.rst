CORE - Using a Pipeline
=======================

Dedicated code in :guilabel:`Core/Pipelines` module.

General Policy
--------------

Several *Pipelines* are available with **DeepPhysX**, allowing the user to:

    * **Generate** synthetic data from simulations → ``DataGeneration``
    * **Train** artificial neural networks with synthetic data → ``Training``
    * Use the **predictions** of trained networks inside a simulation → ``Prediction``

A *Pipeline* is always associated with a :ref:`working session <about-working-sessions>`, whether it already exists or
whether it is automatically created when the *Pipeline* is launched.

The main policy for creating and using a *Pipeline* is to first define a *Configuration* for each component involved by
this *Pipeline*.
*Configurations* can then be mandatory for *Dataset*, *Network* and *Environment* if one wants to create a *Pipeline*.
Once these *Configurations* are defined, the *Pipeline* can be created and launched.

.. note::
    More details are provided for each *Configuration* in dedicated sections.


Pipeline - Data generation
--------------------------

The *DataGeneration* will only involve an *Environment* and a *Dataset*, so this *Pipeline* requires the corresponding
*Configurations*.

As the purpose of this *Pipeline* is only to create synthetic data, the working session will always be created at the
same time.
The name of the repository to create can be provided as a parameter.

Furthermore, users have to define which data to save and how much :

    * The number of **batches** to produce
    * The number of **samples** in a single batch

.. warning::
    This pipeline does not involve a *Network*, the ``get_prediction`` request from Environment is then disabled.
    Trying to launch this request it will lead to an error, so make sure there are no such requests in the data
    production procedure in your *Environment*.

.. highlight:: python

See following example::

    # Import BaseDataGeneration and Config objects
    from DeepPhysX.Core.Pipelines.BaseDataGeneration import BaseDataGeneration
    ...

    # Define configs
    dataset_config = ...
    environment_config = ...

    # Create the pipeline
    data_generator = BaseDataGeneration(session_dir='sessions',
                                        session_name='my_data_generation',
                                        dataset_config=dataset_config,
                                        environment_config=environment_config,
                                        batch_nb=500,
                                        batch_size=16)

    # Launch the pipeline
    data_generator.execute()


Pipeline - Training
-------------------

The *Training* can involve an *Environment*, a *Dataset* and a *Network*, so this *Pipeline* might require the
corresponding *Configurations*.
There are several ways to use this pipeline:

    **Training a Network from scratch**
        To train a *Network* from scratch, the *Training* requires the whole set of *Configurations*.
        A new working session will be created, whose name can be set as a parameter.

    **Training a Network with an existing Dataset**
        Training a new *Network* with an existing *Dataset* is considered as creating a new working session.
        The path to the *Dataset* to use has to be provided as a parameter.
        Using an *Environment* is not mandatory since the training data can already have the right format to feed the
        *Network*.
        If some data computation must be performed between the *Dataset* and the *Network*, an *Environment* can be
        created with the specific *Configuration* (see more in the dedicated section).

    **Training a Network from an existing Network state**
        Training from an existing *Network* state can be done both in an existing session or in a new session.
        If you want to work in the same session, you have to configure the *Training* to do so, otherwise a new working
        session will be automatically created.
        In the same session, a new set of trained parameters will be added in the ``network`` repository, either trained
        with data from an external *Dataset* (whose path must be provided) or with data from the *Environment* (whose
        *Configuration* must be provided).

The last parameters to set in the *Trainer* are:

    * The number of **epochs** the training loop should complete during the session
    * The number of **batches** used during a single epoch
    * The number of **samples** in a single batch

.. note::
    By default, the training data will be produced inside the *Environment* during the first epoch and then re-used
    from the *Dataset* for the remaining epochs.
    If you always need to use data from the *Environment*, you can specify this in its *Configuration*.

.. highlight:: python

See following example::

    # Import BaseTraining and Config objects
    from DeepPhysX.Core.Pipelines.BaseTraining import BaseTraining
    ...

    # Define configs
    dataset_config = ...
    environment_config = ...
    network_config = ...

    # Create the pipeline
    trainer = BaseTraining(session_dir='sessions',
                           session_name='my_training',
                           dataset_config=dataset_config,
                           environment_config=env_config,
                           network_config=net_config,
                           epoch_nb=100,
                           batch_nb=500,
                           batch_size=16)

    # Launch the pipeline
    trainer.execute()


Pipeline - Prediction
---------------------

The *Prediction* always requires a *Network* to compute predictions and an *Environment* to apply them, so this
*Pipeline* will always require the corresponding *Configurations*.

This *Pipeline* always works with an existing working session, no new sessions can be created within a *Prediction*.
The path to the session is therefore required, assuming that it contains a trained *Network*.

The *Prediction* can either run a specified **number of steps** or run an **infinite loop**.

A *Dataset* configuration can be provided.
In this case, the *Prediction* can record prediction data.
Each sample computed during the prediction phase will then be added to the *Dataset* in dedicated partitions.
With a *Dataset*, the *Prediction* can also load its data to **replay** stored samples.

.. highlight:: python

See following example::

    # Import BasePrediction and Config objects
    from DeepPhysX.Core.Pipelines.BasePrediction import BasePrediction
    ...

    # Define configs
    dataset_config = ...
    environment_config = ...
    network_config = ...

    # Create the pipeline
    runner = BasePrediction(session_dir='sessions',
                            session_name='my_training',
                            dataset_config=dataset_config,
                            environment_config=env_config,
                            network_config=net_config,
                            step_nb=-1)

    # Launch the pipeline
    runner.execute()

