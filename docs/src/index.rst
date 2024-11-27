DeepPhysX
=========

Interfacing AI with simulation
------------------------------

The **DeepPhysX** project provides Python packages allowing users to easily interface their **numerical simulations**
with **learning algorithms**.

**DeepPhysX** is mainly designed for :SOFA:`SOFA <>` and :PyTorch:`PyTorch <>` frameworks, but other simulation and AI
frameworks can also be used.

The project is closely linked to the :SSD:`SSD <>` external Python library.


Let's get started
-----------------

To better understand the architecture of **DeepPhysX**, reading the above sections is **highly recommended**:

 * :doc:`presentation/about` → A quick summary of DeepPhysX features.
 * :doc:`presentation/overview` → The project architecture and packages.

Then, to start using **DeepPhysX**, please refer to the :doc:`presentation/install` section to configure your own
DeepPhysX installation.
Some examples are provided in each ``Example`` repository to learn how to use the package, with
a 3 leveled complexity:

 * **Tutorial**: A walkthrough of **DeepPhysX** to learn the basics.
 * **Features**: A full session reviewing all features of **DeepPhysX** via a simple example.
 * **Demos**: Demonstration sessions with high-level applications.

Finally, to go further in understanding some components of **DeepPhysX**, refer to the associated section.


.. toctree::
    :caption: PRESENTATION
    :maxdepth: 2
    :hidden:

    About     <presentation/about.rst>
    Overview  <presentation/overview.rst>
    Install   <presentation/install.rst>

.. toctree::
    :caption: COMPONENTS
    :maxdepth: 2
    :hidden:

    Pipelines    <component/pipelines.rst>
    Environment  <component/environment.rst>
    Network      <component/network.rst>
    Dataset      <component/dataset.rst>
    Visualizer   <component/visualizer.rst>
    Stats        <component/stats.rst>

.. toctree::
    :caption: API
    :maxdepth: 1
    :hidden:

    Core   <api/core.rst>
    Sofa   <api/sofa.rst>
    Torch  <api/torch.rst>
