from typing import List, Dict, Union, Tuple, Any
from vedo import Mesh, Glyph, Marker, Points, Arrows, Plotter

from DeepPhysX_Core.Visualizer.VedoObjectFactories.MarkerFactory import MarkerFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.MeshFactory import MeshFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.GlyphFactory import GlyphFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.PointsFactory import PointsFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.WindowFactory import WindowFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.ArrowsFactory import ArrowsFactory

ObjectDescription = Dict[str, Union[Dict[str, Any], Any]]
VisualInstance = Union[Mesh, Glyph, Marker, Points, Arrows]
Factory = Union[MeshFactory, PointsFactory, GlyphFactory, MarkerFactory, WindowFactory]


class VedoObjectFactory:
    """
    | VedoObjectFactory contains all the Factories of a scene.
    """

    def __init__(self):

        self.next_id: int = 0
        self.objects_dict: Dict[int, ObjectDescription] = {}
        self.updated_object_dict: Dict[int, ObjectDescription] = {}
        self.windows_id: List[int] = []
        self.factories: Dict[int, Factory] = {}
        self.non_updatable_objects: List[str] = ['Arrows']

    def add_object(self, object_type: str, data_dict: ObjectDescription) -> Tuple[ObjectDescription, int, Factory]:
        """
        | Create a factory with the given object type and data dictionary.

        :param str object_type: Type of the desired factory Object (Mesh, Points, Glyph, Marker, Window, Arrows)
        :param data_dict: Dictionary that contains the associated data
        :type data_dict: Dict[str, Union[Dict[str, Any], Any]]
        :return: The fully parsed and updated dictionary, its index, the associated factory
        """

        self.factories[self.next_id] = self.factory_getter(object_type)

        if self.factories[self.next_id] is None:
            raise ValueError("The given type does not exist. Please use on of the following :\n"
                             "Mesh, mesh\n"
                             "Points, PointCloud, Point, points, point\n"
                             "Glyph, glyph\n"
                             "Marker, marker, markers, Markers\n"
                             "Window, window\n"
                             "Arrows, arrows, Arrow, arrow\n")

        self.objects_dict[self.next_id] = self.factories[self.next_id].parse(data_dict)

        if object_type in ['Window', 'window']:
            self.windows_id.append(self.next_id)

        self.next_id += 1

        return self.objects_dict[self.next_id - 1], self.next_id - 1, self.factories[self.next_id - 1]

    def update_object_dict(self, object_id: int, new_data_dict: ObjectDescription) -> Tuple[ObjectDescription, Factory]:
        """
        | Update the object with the given ID using the data passed by new_data_dict.

        :param int object_id: ID of the object to update
        :param new_data_dict: Dictionary containing the data to update
        :type new_data_dict: Dict[str, Union[Dict[str, Any], Any]]
        :return: The updated dictionary, the updated factory
        """

        if object_id not in self.factories:
            self.factories[object_id] = self.factory_getter(self.objects_dict[object_id]["type"])
        self.objects_dict[object_id] = self.factories[object_id].parse(new_data_dict)
        self.updated_object_dict[object_id] = {}
        for field in new_data_dict:
            self.updated_object_dict[object_id][field] = new_data_dict[field]
        return self.objects_dict[object_id], self.factories[object_id]

    def update_object_instance(self, object_id: int, instance: VisualInstance,
                               viewer_data: Tuple[Plotter, int]) -> VisualInstance:
        """
        | Update the given instance using the factory corresponding to the passed object_id

        :param int object_id: ID of the factory of the object to use
        :param instance: VisualInstance object to update
        :type instance: Union[Mesh, Glyph, Marker, Points, Arrows]
        :param Tuple[Plotter, int] viewer_data: Tuple with [Viewer instance, index of sub-window] in which object is
                                                rendered
        :return: The updated VisualInstance
        """

        if object_id not in self.factories:
            self.factories[object_id] = self.factory_getter(self.objects_dict[object_id]["type"])
        viewer, at = viewer_data[0], viewer_data[1]

        if self.objects_dict[object_id]['type'] in self.non_updatable_objects:
            viewer.remove(instance)
            updated_instance = self.factories[object_id].update_instance(instance)
            viewer.add(updated_instance, at=at)
        else:
            updated_instance = self.factories[object_id].update_instance(instance)
        self.updated_object_dict[object_id] = {}
        # return updated_instance, viewer
        return updated_instance

    @staticmethod
    def factory_getter(object_type: str) -> Factory:
        """
        | Helper function that return a default Factory corresponding to the given object_type

        :param str object_type: Type of the object
        :return: Factory corresponding to the given object type
        """

        factory = None
        if object_type in ["Mesh", "mesh"]:
            factory = MeshFactory()
        elif object_type in ["Points", "PointCloud", "Point", "points", "point"]:
            factory = PointsFactory()
        elif object_type in ["Glyph", "glyph"]:
            factory = GlyphFactory()
        elif object_type in ['Marker', 'marker', 'markers', "Markers"]:
            factory = MarkerFactory()
        elif object_type in ['Window', 'window']:
            factory = WindowFactory()
        elif object_type in ['Arrow', 'Arrows', 'arrow', 'arrows']:
            factory = ArrowsFactory()
        return factory
