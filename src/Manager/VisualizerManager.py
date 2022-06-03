from typing import Dict, Union, Any, Optional

ObjectDescription = Dict[str, Union[Dict[str, Any], Any]]


class VisualizerManager:
    """
    | Handle the 3D representation of the data from a visualizer.
    | Allows easy access to basic functionalities of the visualizer

    :param Optional[DataManager] data_manager: DataManager that handles the VisualizerManager
    :param Optional[VedoVisualizer] visualizer: The class of the desired Visualizer
    :param int screenshot_rate: Frequency on which a screenshot must be taken to store samples in the Dataset.
    """

    def __init__(self,
                 data_manager: Optional[Any] = None,
                 visualizer: Optional[Any] = None,
                 screenshot_rate: int = 0):

        self.data_manager: Any = data_manager
        self.visualizer: Any = visualizer()

        self.screenshot_rate: int = screenshot_rate
        self.screenshot_counter: int = 0

    def get_data_manager(self) -> Any:
        """
        | Return the manager that handles the VisualizerManager.

        :return: DataManager that handles the VisualizerManager
        """

        return self.data_manager

    def init_view(self, data_dict: Dict[int, Dict[int, ObjectDescription]]) -> None:
        """
        | Initialize VedoVisualizer class by parsing the scenes hierarchy and creating VisualInstances.
        | OBJECT DESCRIPTION DICTIONARY is usually obtained using the corresponding factory (VedoObjectFactory)
        | data_dict example:
        |       {SCENE_1_ID: {OBJECT_1.1_ID: {CONTENT OF OBJECT_1.1 DESCRIPTION DICTIONARY},
        |                     ...
        |                     OBJECT_1.N_ID: {CONTENT OF OBJECT_1.N DESCRIPTION DICTIONARY}
        |                     },
        |        ...
        |        SCENE_M_ID: {OBJECT_M.1_ID: {CONTENT OF OBJECT_K.1 DESCRIPTION DICTIONARY},
        |                     ...
        |                     OBJECT_M.K_ID: {CONTENT OF OBJECT_K.P DESCRIPTION DICTIONARY}
        |                    }
        |        }

        :param data_dict: Dictionary describing the scene hierarchy and object parameters
        :type data_dict: Dict[int, Dict[int, Dict[str, Union[Dict[str, Any], Any]]]]
        """

        self.visualizer.init_view(data_dict)

    def update_visualizer(self, data_dict: Dict[int, Dict[int, ObjectDescription]]) -> None:
        """
        | Call update_object_dict on all designed objects

        :param data_dict: Dictionary describing the scene hierarchy and object parameters
        :type data_dict: Dict[int, Dict[int, ObjectDescription]]
        """

        self.visualizer.update_visualizer(data_dict)
        self.screenshot_counter += 1
        if self.screenshot_counter == self.screenshot_rate:
            self.visualizer.save_screenshot(session_dir=self.data_manager.get_manager().session_dir)
            self.screenshot_counter = 0

    def render(self):
        """
        | Trigger a render step of the visualization window.
        """

        self.visualizer.render()

    def saveSample(self, session_dir):
        """
        | Save the samples as a filetype defined by the visualizer

        :param str session_dir: Directory in which to save the file
        """

        self.visualizer.save_sample(session_dir=session_dir)

