from typing import Any, Optional, Dict, Union, Tuple, List, Type
from numpy import ndarray

from DeepPhysX.simulation.abstract_controller import AbstractController
from SimRender.core import Viewer
from time import time


class DPXSimulation:

    def __init__(self, **kwargs):
        """
        BaseEnvironment computes simulated data for the networks and its training process.
        """

        self.__controller: AbstractController = kwargs.pop('environment_controller')
        self.name: str = f"{self.__class__.__name__} n°{self.environment_id}"

    @property
    def environment_id(self) -> int:
        return self.__controller.environment_ids[0]

    @property
    def environment_nb(self) -> int:
        return self.__controller.environment_ids[1]

    @property
    def viewer(self) -> Optional[Viewer]:
        return self.__controller.viewer

    @property
    def compute_training_data(self) -> bool:
        return self.__controller.compute_training_data

    ##########################################################################################
    ##########################################################################################
    #                                 Environment initialization                             #
    ##########################################################################################
    ##########################################################################################

    def create(self) -> None:
        """
        Create the Environment. Automatically called when Environment is launched.
        Must be implemented by user.
        """

        raise NotImplementedError

    def init(self) -> None:
        """
        Initialize the Environment. Automatically called when Environment is launched.
        Not mandatory.
        """

        pass

    def init_database(self) -> None:
        """
        Define the fields of the training dataset. Automatically called when Environment is launched.
        Must be implemented by user.
        """

        raise NotImplementedError

    def init_visualization(self) -> None:
        """
        Define the visualization objects to send to the Visualizer. Automatically called when Environment is launched.
        Not mandatory.
        """

        pass

    def save_parameters(self, **kwargs) -> None:
        """
        Save a set of parameters in the Database.
        """

        self.__controller.save_parameters(**kwargs)

    def load_parameters(self) -> Dict[str, Any]:
        """
        Load a set of parameters from the Database.
        """

        return self.__controller.load_parameters()

    ##########################################################################################
    ##########################################################################################
    #                                 Environment behavior                                   #
    ##########################################################################################
    ##########################################################################################

    async def step(self) -> None:
        """
        Compute the number of steps in the Environment specified by simulations_per_step in EnvironmentConfig.
        Must be implemented by user.
        """

        raise NotImplementedError

    def check_sample(self) -> bool:
        """
        Check if the current produced sample is usable for training.
        Not mandatory.

        :return: Current data can be used or not
        """

        return True

    def apply_prediction(self, prediction: Dict[str, ndarray]) -> None:
        """
        Apply networks prediction in environment.
        Not mandatory.

        :param prediction: Prediction data.
        """

        pass

    def close(self) -> None:
        """
        Close the Environment. Automatically called when Environment is shut down.
        Not mandatory.
        """

        pass

    ##########################################################################################
    ##########################################################################################
    #                                 Defining data samples                                  #
    ##########################################################################################
    ##########################################################################################

    def define_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Specify the data fields names and types.

        :param fields: Field or list of fields to store data.
        """

        self.__controller.define_database_fields(fields=fields)

    def set_data(self, **kwargs) -> None:
        """
        Set the training data to send to the TcpIpServer or the EnvironmentManager.
        """

        self.__controller.set_data(**kwargs)

    @property
    def data(self) -> Dict[str, ndarray]:
        return self.__controller.get_data()

    ##########################################################################################
    ##########################################################################################
    #                                   Available requests                                   #
    ##########################################################################################
    ##########################################################################################

    def get_prediction(self, **kwargs) -> Dict[str, ndarray]:
        """
        Request a prediction from networks.

        :return: networks prediction.
        """

        return self.__controller.get_prediction(**kwargs)

    def update_visualisation(self) -> None:
        """
        Triggers the Visualizer update.
        """

        if self.viewer is not None:
            # s = time()
            self.viewer.render()
            # print('rendering_time = ', time() - s)

    def __str__(self):

        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Name: {self.name} n°{self.environment_id}\n"
        description += f"    Comments:\n"
        description += f"    Input size:\n"
        description += f"    Output size:\n"
        return description
