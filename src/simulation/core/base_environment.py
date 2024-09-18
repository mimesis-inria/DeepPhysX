from typing import Any, Optional, Dict, Union, Tuple, List, Type
from numpy import ndarray

from SSD.Core.Rendering.user_api import UserAPI
from DeepPhysX.Core.Environment.AbstractController import AbstractController


class BaseEnvironment:

    def __init__(self, **kwargs):
        """
        BaseEnvironment computes simulated data for the network and its training process.
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
    def visual(self) -> Optional[UserAPI]:
        return self.__controller.visualization_factory

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
        Apply network prediction in environment.
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

    def define_training_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Specify the training data fields names and types.

        :param fields: Field or list of fields to tag as training data.
        """

        self.__controller.define_database_fields(table_name='Training', fields=fields)

    def define_additional_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Specify the additional data fields names and types.

        :param fields: Field or list of Fields to tag as additional data.
        """

        self.__controller.define_database_fields(table_name='Additional', fields=fields)

    def set_training_data(self, **kwargs) -> None:
        """
        Set the training data to send to the TcpIpServer or the EnvironmentManager.
        """

        self.__controller.set_training_data(**kwargs)

    def set_additional_data(self,
                            **kwargs) -> None:
        """
        Set the additional data to send to the TcpIpServer or the EnvironmentManager.
        """

        self.__controller.set_additional_data(**kwargs)

    @property
    def training_data(self) -> Dict[str, ndarray]:
        return self.__controller.get_training_data()

    @property
    def additional_data(self) -> Dict[str, ndarray]:
        return self.__controller.get_additional_data()

    ##########################################################################################
    ##########################################################################################
    #                                   Available requests                                   #
    ##########################################################################################
    ##########################################################################################

    def get_prediction(self, **kwargs) -> Dict[str, ndarray]:
        """
        Request a prediction from network.

        :return: network prediction.
        """

        return self.__controller.get_prediction(**kwargs)

    def update_visualisation(self) -> None:
        """
        Triggers the Visualizer update.
        """

        if self.visual is not None:
            self.visual.render()

    def __str__(self):

        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Name: {self.name} n°{self.environment_id}\n"
        description += f"    Comments:\n"
        description += f"    Input size:\n"
        description += f"    Output size:\n"
        return description