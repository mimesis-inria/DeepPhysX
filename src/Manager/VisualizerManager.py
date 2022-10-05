from typing import Dict, Union, Any, Optional, Type
from os.path import join

from SSD.Core.Rendering.VedoVisualizer import VedoVisualizer

ObjectDescription = Dict[str, Union[Dict[str, Any], Any]]


class VisualizerManager:

    def __init__(self,
                 session: str,
                 visualizer: Type[VedoVisualizer],
                 data_manager: Optional[Any] = None):
        """
        Handle the 3D representation of the data from a visualizer.
        Allows easy access to basic functionalities of the visualizer

        :param data_manager: DataManager that handles the VisualizerManager
        :param visualizer: The class of the desired Visualizer
        """

        self.data_manager: Any = data_manager
        self.visualizer: VedoVisualizer = visualizer(database_dir=join(session, 'dataset'),
                                                     database_name='Visualization')

    def get_data_manager(self) -> Any:
        """
        | Return the manager that handles the VisualizerManager.

        :return: DataManager that handles the VisualizerManager
        """

        return self.data_manager

    def init_view(self) -> None:
        """
        Initialize VedoVisualizer class by parsing the scenes hierarchy and creating VisualInstances.
        """

        self.visualizer.init_visualizer()

    def render(self):
        """
        | Trigger a render step of the visualization window.
        """

        self.visualizer.render()
