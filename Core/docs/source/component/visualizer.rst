CORE - Using a Visualizer
=========================

Dedicated code in :guilabel:`Core/Visualizer` module.

.. _visualizer-howto:

How to use
----------

**DeepPhysX** provides a visualization tool written with :Vedo:`Vedo <>` (a Python library based on Numpy and VTK)
called *VedoVisualizer*.
This *Visualizer* brings several advantages:

* Users can add any component of the simulation in the *Visualizer*;
* This *Visualizer* is compatible with every *Pipeline*;
* Parallel running *Environments* are rendered in the same window with sub-windows;
* A *Factory* is created with each *Environment* so that users can access templates to define visualization data.

Objects are created using the ``add_object`` method of the *Factory* in the *Environment*.
This method requires the object type name and a dictionary containing the required fields detailed above (common fields
and object-specific fields).
These objects must be defined in the ``send_visualization`` method of the *Environment*, which must return the objects
dictionary of the *Factory*.

Objects are updated using the ``update_object_dict`` method of the *Factory* in the *Environment*.
This method requires the object index (indices follow the order of creation) and a dictionary containing the updated
data fields.
The *Factory* will only use templates to create an updated objects dictionary which must be sent with the request
``update_visualization`` to update the view.

| **General parameters**
| Visual objects share default data fields that could also be filled at init and are all optional:

.. list-table::
    :width: 95%
    :widths: 21 11 11 57
    :header-rows: 1

    * - Field
      - Init
      - Update
      - Description

    * - ``c``
      - Optional
      - Unnecessary
      - Opacity of the object between 0 and 1.

    * - ``alpha``
      - Optional
      - Unnecessary
      - Marker object.

    * - ``at``
      - Optional
      - Unnecessary
      - Sub-window in which the object will be rendered.

        Set to -1 by default, meaning a new window is created for the object.

        Advise: set to ``self.instance_id`` to gather objects from an *Environment* in the same sub-window.

    * - ``colormap``
      - Optional
      - Unnecessary
      - Name of color palette that samples a continuous function between two end colors.

    * - ``scalar_field``
      - Optional
      - Unnecessary
      - List of scalar values to set individual points or cell color.

    * - ``scalar_field_name``
      - Optional
      - Unnecessary
      - Name of the scalar field.

| **Visual Objects Parameters**
| A list of templates are available in the *Factory* to initialize and update a list of objects.
  Here is a description of available objects and the required data fields:

* Create a :VedoObject:`Mesh <mesh.html#vedo.mesh.Mesh>`:

  .. list-table::
      :width: 95%
      :widths: 21 11 11 57
      :header-rows: 1

      * - Field
        - Init
        - Update
        - Description

      * - ``positions``
        - **Required**
        - **Required**
        - List of vertices.
          Updated position vector must always have the same size.

      * - ``cells``
        - **Required**
        - Unnecessary
        - List of connections between vertices.

      * - ``computeNormals``
        - Optional
        - Unnecessary
        - Compute cells and points normals at creation.

* Create a :VedoObject:`Point Cloud <pointcloud.html#vedo.pointcloud.Points>`:

  .. list-table::
      :width: 95%
      :widths: 21 11 11 57
      :header-rows: 1

      * - Field
        - Init
        - Update
        - Description

      * - ``positions``
        - **Required**
        - **Required**
        - List of vertices.
          Updated position vector must always have the same size.

      * - ``r``
        - Optional
        - Optional
        - Radius of points.

* Create a :VedoObject:`Marker <shapes.html#vedo.shapes.Marker>` (single point with associated symbol):

  .. list-table::
      :width: 95%
      :widths: 21 11 11 57
      :header-rows: 1

      * - Field
        - Init
        - Update
        - Description

      * - ``positions``
        - **Required**
        - **Required**
        - Position of the Marker.

      * - ``symbol``
        - **Required**
        - Unnecessary
        - Associated symbol.

      * - ``s``
        - Optional
        - Unnecessary
        - Radius of symbol.

      * - ``filled``
        - Optional
        - Unnecessary
        - Fill the shape or only draw outline.

* Create a :VedoObject:`Glyph <shapes.html#vedo.shapes.Glyph>` (point cloud with oriented markers):

  .. list-table::
      :width: 95%
      :widths: 21 11 11 57
      :header-rows: 1

      * - Field
        - Init
        - Update
        - Description

      * - ``positions``
        - **Required**
        - **Required**
        - Position of the Markers.

      * - ``glyphObj``
        - **Required**
        - Unnecessary
        - Marker object.

      * - ``orientationArray``
        - **Required**
        - Unnecessary
        - List of orientation vectors.

      * - ``scaleByScalar``
        - Optional
        - Unnecessary
        - Glyph is scaled by the scalar field.

      * - ``scaleByVectorSize``
        - Optional
        - Unnecessary
        - Glyph is scaled by the size of the orientation vectors.

      * - ``scaleByVectorComponents``
        - Optional
        - Unnecessary
        - Glyph is scaled by the components of the orientation vectors.

      * - ``colorByScalar``
        - Optional
        - Unnecessary
        - Glyph is colored based on the colormap and the scalar field.

      * - ``colorByVectorSize``
        - Optional
        - Unnecessary
        - Glyph is colored based on the size of the orientation vectors.

      * - ``tol``
        - Optional
        - Unnecessary
        - Minimum distance between two Glyphs.

* Create :VedoObject:`3D Arrows <shapes.html#vedo.shapes.Arrows>`:

  .. list-table::
      :width: 95%
      :widths: 21 11 11 57
      :header-rows: 1

      * - Field
        - Init
        - Update
        - Description

      * - ``positions``
        - **Required**
        - **Required**
        - Start points of the arrows.

      * - ``vectors``
        - **Required**
        - **Required**
        - Vector that must represent the arrows.

      * - ``res``
        - Optional
        - Unnecessary
        - Arrows visual resolution.

* Change window parameters

  .. list-table::
      :width: 95%
      :widths: 21 11 11 57
      :header-rows: 1

      * - Field
        - Init
        - Update
        - Description

      * - ``objects_id``
        - **Required**
        - Unnecessary
        - Indices of objects to set in this particular window.

      * - ``title``
        - Optional
        - Unnecessary
        - Title of the window.

      * - ``axes``
        - Optional
        - Unnecessary
        - Type of axes to show.

      * - ``sharecam``
        - Optional
        - Unnecessary
        - If True (default), all subwindows will share the same camera parameters.

      * - ``interactive``
        - Optional
        - Unnecessary
        - If True (default), the window will be interactive.


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
    from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer

    # Create the config
    env_config = BaseEnvironmentConfig(environment_class=MyEnvironment,
                                       visualizer=VedoVisualizer)
