CORE - Using a Dataset
======================

Dedicated code in :guilabel:`Core/Dataset` module.

Behavior
--------

**DeepPhysX** comes with its own *Dataset* management system.
The synthetic data produced in *Environments* is stored as Numpy arrays in a dedicated repository named ``dataset`` in
the training session.
Data is stored as partitions: these partitions correspond to the different *Dataset* modes (training data, test data,
prediction data) and can be multiple for each data field in order not to exceed the maximum size of the current amount
of loaded data.
Each partition will have a unique name: ``<session_name>_<dataset_mode>_<field>_<index>.npy``.

.. figure:: ../_static/image/database.png
    :alt: database.png
    :align: center
    :width: 80%

    Dataset repository organisation.

When adding a batch to the *Dataset*, a new partition is created for each data field if the current *Dataset* size
exceeds the threshold.
The batch is then appended to the *Dataset* for each data field.
Default *Dataset* fields are inputs and outputs, but users can add any data to the *Dataset* from *Environment* using
``additional_in_dataset`` or ``additional_out_dataset`` (see :ref:`dedicated section <environment-implementation>`).
Each field must always be filled at each batch.
A ``dataset.json`` file gathers information about the produced dataset.

When loading data from an existing *Dataset*, the repository is loaded first.
If there is a single partition for each field, only those partitions are loaded into the *Dataset*.
Otherwise, a proportional part of each partition will be loaded each time.
Batches of data are accessed in read order (random or not) until the read cursor reaches the end, triggering either the
reloading of a single partition or the loading of the subsequent slices of partitions.


Configuration
-------------

A *Dataset* is almost always used in DeepPhysX pipelines and requires a *DatasetConfig*.
This *Configuration* object role is to bring together all the options to configure the *Dataset* and to create it with
``create_dataset`` method.

Here is a description of attributes related to *Dataset* configuration.

.. list-table::
    :width: 100%
    :widths: 15 85

    * - ``dataset_class``
      - *Dataset* class from which an instance will be created (*BaseDataset* by default).

    * - ``dataset_dir``
      - Path to an existing *Dataset* repository if this repository needs to be loaded or completed.

    * - ``partition_size``
      - Maximum size (in Gb) of the total *Dataset* object.

    * - ``shuffle_dataset``
      - Specify if the loading order is random or not (True by default).

    * - ``use_mode``
      - Specify the *Dataset* mode between "Training", "Validation" and "Running".

    * - ``normalize_data``
      - If True, normalization parameters are computed from training data and applied to any loaded data.

.. highlight:: python

See following example::

    # Import DatasetConfig
    from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig

    # Create the config
    dataset_config = BaseDatasetConfig(partition_size=1,
                                       shuffle_dataset=True,
                                       use_mode='Training',
                                       normalize_data=True)
