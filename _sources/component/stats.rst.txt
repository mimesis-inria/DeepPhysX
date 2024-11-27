CORE - Using the Stats
======================

How to Use
----------

During the training session, some optimizations analysis are performed and stored as a readable file for
:Tensorboard:`Tensorboard <>`.
This file is automatically filled at each *Network* optimization step in the ``stats`` repository of the current working
directory.

Tensorboard will be automatically launched for the current training session in a web browser.
To open a previous training session analysis, launch Tensorboard with:

.. code-block:: bash

    $ cd <working_session>/stats
    $ tensorboard --logdir .
    >> Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
       TensorBoard x.x.x at http://localhost:6006/ (Press CTRL+C to quit)

Then copy-paste the provided link into a web browser.

Adding fields
-------------

By default, six data fields will be stored:

* The **value** of the **loss** function at each batch / epoch
* The **mean** of the **loss** function at each batch / epoch over the last 50 elements
* The **variance** of the **loss** function at each batch / epoch

Custom data fields can be set in Tensorboard as well.
After an optimization step, the *Optimization* class will by default return a dictionary with a single item named
``loss`` containing the current loss value.
Then, the *Trainer* will add every item contained in this dictionary to the *StatsManager*.
To fill this dictionary with custom fields, a custom *Optimization* class must be implemented where the
``transform_loss`` method will fill that dictionary (see :ref:`dedicated section <network-optimization>`).
This new field will have the same name as the item and take each value given in this item at each optimization step.

.. highlight:: python

See following example::

    # Import BaseOptimization
    from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization

    # Create a new Optimization class
    class MyOptimization(BaseOptimization):

        # Override the transform_loss method
        def transform_loss(self, data):
            return {'loss': self.loss_value     # Default loss value
                    'new_field': ...}           # Additional field
