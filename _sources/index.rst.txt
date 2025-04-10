DeepPhysX
=========

Interfacing AI with simulation
------------------------------

The **DeepPhysX** project provides Python packages allowing users to easily interface their **numerical simulations**
with **learning algorithms**.

**DeepPhysX** is mainly designed for :SOFA:`SOFA <>` and :PyTorch:`PyTorch <>` frameworks, but other simulation
frameworks can also be used.

The project is closely linked to the :SSD:`SSD <>` and :SimRender:`SimRender <>` external Python libraries.


Let's get started
-----------------

To better understand the architecture of **DeepPhysX**, reading the above sections is **highly recommended**:

 * :doc:`about` → A quick summary of DeepPhysX features.

Then, to start using **DeepPhysX**, please refer to the :doc:`install` section to configure your own
DeepPhysX installation.
A `tutorial <https://github.com/mimesis-inria/DeepPhysX/tree/master/examples/tutorial>`_ is provided to learn how to
use the package.

Finally, to go further in understanding some components of **DeepPhysX**, refer to the associated section.


.. toctree::
    :caption: PRESENTATION
    :maxdepth: 2
    :hidden:

    About     <about.rst>
    Install   <install.rst>

.. toctree::
    :caption: API
    :maxdepth: 1
    :hidden:

    Pipelines   <api/pipelines.rst>
    Database     <api/database.rst>
    Simulation    <api/simulation.rst>
    Network       <api/network.rst>
