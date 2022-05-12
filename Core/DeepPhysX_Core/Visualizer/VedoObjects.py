from typing import Callable, Dict, Union, Tuple
from vedo import Mesh, Glyph, Marker, Points, Arrows, Plotter

from DeepPhysX_Core.Visualizer.VedoObjectFactories.VedoObjectFactory import VedoObjectFactory
from DeepPhysX_Core.Visualizer.VedoObjectFactories.VedoObjectFactory import ObjectDescription, VisualInstance


create_filter: Dict[str, Callable[[ObjectDescription], VisualInstance]] = {

    "Mesh": lambda sorted_data: Mesh(inputobj=[sorted_data["positions"], sorted_data["cells"]],
                                     c=sorted_data["c"],
                                     alpha=sorted_data["alpha"]).wireframe(sorted_data['wireframe']).computeNormals(sorted_data['computeNormals']),

    "Points": lambda sorted_data: Points(inputobj=sorted_data["positions"],
                                         c=sorted_data["c"],
                                         alpha=sorted_data["alpha"],
                                         r=sorted_data["r"]),

    "Marker": lambda sorted_data: Marker(symbol=sorted_data["symbol"],
                                         pos=sorted_data["position"][0],
                                         c=sorted_data["c"],
                                         alpha=sorted_data["alpha"],
                                         s=sorted_data["s"],
                                         filled=sorted_data["filled"]),

    "Glyph": lambda sorted_data: Glyph(mesh=sorted_data["positions"],
                                       glyphObj=create_filter['Marker'](sorted_data["glyphObj"]),
                                       orientationArray=sorted_data["orientationArray"],
                                       scaleByScalar=sorted_data["scaleByScalar"],
                                       scaleByVectorSize=sorted_data["scaleByVectorSize"],
                                       scaleByVectorComponents=sorted_data["scaleByVectorComponents"],
                                       colorByScalar=sorted_data["colorByScalar"],
                                       colorByVectorSize=sorted_data["colorByVectorSize"],
                                       tol=sorted_data["tol"],
                                       c=sorted_data["c"],
                                       alpha=sorted_data["alpha"]),

    "Arrows": lambda sorted_data: Arrows(startPoints=sorted_data['positions'],
                                         endPoints=sorted_data['positions'] + sorted_data['vectors'],
                                         c=sorted_data['c'],
                                         alpha=sorted_data['alpha'],
                                         res=sorted_data['res']),

    "Window": lambda sorted_data: None  # Window is not a vedo object hence we create a pass through
}


class VedoObjects:
    """
    | Container class that contain a scene description (factory, hierarchy, instances).
    | VedoObjects is container that matches visualizer and VisualInstances to provide an easy and intuitive mean to
      update Vedo scenes from a remote client.
    """

    def __init__(self):

        self.name = self.__class__.__name__
        self.objects_instance = {}
        self.objects_factory = VedoObjectFactory()

    def create_object(self, data_dict: ObjectDescription) -> None:
        """
        | Initialise a factory and a VisualInstance with the given parameters (Type, positions, etc.).

        :param Dict[str, Union[Dict[str, Any], Any]] data_dict: Dictionary that describes the parameters and type of
                                                                object
        """

        # Register the object in the ObjectsFactory
        sorted_data_dict, object_id, _ = self.objects_factory.add_object(object_type=data_dict['type'],
                                                                         data_dict=data_dict)
        # Create the VisualObject instance
        self.objects_instance[object_id] = create_filter[data_dict['type']](sorted_data_dict)

    def update_object(self, object_id: int, new_dict: ObjectDescription) -> None:
        """
        | Update the factory designed by the object_id with the given data.

        :param int object_id: ID of the factory/object to update
        :param Dict[str, Dict[str, Any]] new_dict: Dictionary containing the data to update
        """

        self.objects_factory.update_object_dict(object_id, new_dict)

    def update_instance(self, object_id: int, viewer_data: Tuple[Plotter, int]) -> None:
        """
        | Update the VisualInstance designed by the object_id with the corresponding factory.

        :param int object_id: ID of the factory/object to update
        :param Tuple[Plotter, int] viewer_data: Tuple with [Viewer instance, index of sub-window] in which object is
                                                rendered
        """

        updated_instance = self.objects_factory.update_object_instance(object_id=object_id,
                                                                       instance=self.objects_instance[object_id],
                                                                       viewer_data=viewer_data)
        self.objects_instance[object_id] = updated_instance
