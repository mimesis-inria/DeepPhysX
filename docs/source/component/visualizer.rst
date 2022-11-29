CORE - Using a Visualizer
=========================

Dedicated code in :guilabel:`Core/Visualizer` module.

.. _visualizer-howto:

How to use
----------

**DeepPhysX** provides a visualization tool written with :Vedo:`Vedo <>` (a Python library based on Numpy and VTK)
called *VedoVisualizer*, using the implementation form the :SSD:`SSD <>` library.
This *Visualizer* brings several advantages:

* Users can add any component of the simulation in the *Visualizer*;
* This *Visualizer* is compatible with every *Pipeline*;
* Parallel running *Environments* are rendered in the same window with sub-windows;
* A *Factory* is created with each *Environment* so that users can access templates to define visualization data.

Objects are created using the ``add_<object>`` methods of the *Factory* in the *Environment*.
These methods require the fields detailed above (common fields and object-specific fields).
These objects must be added in the ``init_visualization`` method of the *Environment*, which is automatically called to
create the objects within the *Factory*.

Objects are updated using the ``update_<object>`` methods of the *Factory* in the *Environment*.
These methods require the object index (indices follow the order of creation) and a the updated data fields.
The *Factory* will only use templates to create an updated objects dictionary which must be sent with the request
``update_visualization`` to update the view.


| **Visual Objects Parameters**
| All the available objects and their parameters are listed on the
  :SSDd:`SSD documentation <Core/Rendering/factory.html#vedofactory>`.


Configuration
-------------

Configuring a *Visualizer* is very simple, since the only option to change is the ``visualizer`` field in the
*EnvironmentConfig*.
If set to None, no *Visualizer* will be created, even if the *Environment* uses its *Factory* to create and update
visualization data.
It must be set to *VedoVisualizer* to activate the visualization tool.

.. highlight:: python

See following example::

    # Import EnvironmentConfig and Visualizer
    from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
    from DeepPhysX_Core.Visualization.VedoVisualizer import VedoVisualizer

    # Create the config
    env_config = BaseEnvironmentConfig(environment_class=MyEnvironment,
                                       visualizer=VedoVisualizer)
