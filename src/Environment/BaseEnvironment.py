from typing import Any, Optional, Dict, Union, Tuple
from numpy import array, ndarray

from SSD.Core.Storage.Database import Database

from DeepPhysX.Core.Visualization.VedoFactory import VedoFactory
from DeepPhysX.Core.AsyncSocket.TcpIpClient import TcpIpClient


class BaseEnvironment(TcpIpClient):

    def __init__(self,
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 instance_id: int = 0,
                 number_of_instances: int = 1,
                 as_tcp_ip_client: bool = True,
                 environment_manager: Optional[Any] = None,
                 visu_db: Optional[Union[Database, Tuple[str, str]]] = None):
        """
        BaseEnvironment is an environment class to compute simulated data for the network and its optimization process.

        :param ip_address: IP address of the TcpIpObject.
        :param port: Port number of the TcpIpObject.
        :param instance_id: ID of the instance.
        :param number_of_instances: Number of simultaneously launched instances.
        :param as_tcp_ip_client: Environment is owned by a TcpIpClient if True, by an EnvironmentManager if False.
        :param environment_manager: EnvironmentManager that handles the Environment if 'as_tcp_ip_client' is False.
        :param visu_db: The path to the visualization Database or the visualization Database object to connect to.
        """

        TcpIpClient.__init__(self,
                             instance_id=instance_id,
                             number_of_instances=number_of_instances,
                             as_tcp_ip_client=as_tcp_ip_client,
                             ip_address=ip_address,
                             port=port)

        # Input and output to give to the network
        self.input: ndarray = array([])
        self.output: ndarray = array([])
        # Variables to store samples from Dataset
        self.sample_in: Optional[ndarray] = None
        self.sample_out: Optional[ndarray] = None
        # Loss data
        self.loss_data: Any = None
        # Manager if the Environment is not a TcpIpClient
        self.environment_manager: Any = environment_manager

        self.factory: Optional[VedoFactory] = None
        if visu_db is not None:
            if type(visu_db) == list:
                self.factory = VedoFactory(database_path=visu_db,
                                           idx_instance=instance_id,
                                           remote=True)
            else:
                self.factory = VedoFactory(database=visu_db,
                                           idx_instance=instance_id)

    ##########################################################################################
    ##########################################################################################
    #                                 Initializing Environment                               #
    ##########################################################################################
    ##########################################################################################

    def recv_parameters(self,
                        param_dict: Dict[Any, Any]) -> None:
        """
        Exploit received parameters before scene creation.
        Not mandatory.

        :param param_dict: Dictionary of parameters.
        """

        pass

    def create(self) -> None:
        """
        Create the Environment.
        Must be implemented by user.
        """

        raise NotImplementedError

    def init(self) -> None:
        """
        Initialize the Environment.
        Not mandatory.
        """

        pass

    def send_parameters(self) -> Dict[Any, Any]:
        """
        Create a dictionary of parameters to send to the manager.
        Not mandatory.

        :return: Dictionary of parameters
        """

        return {}

    def init_visualization(self) -> None:
        """
        Define the visualization objects to send to he Visualizer.
        Not mandatory.

        :return: Dictionary of visualization data
        """

        pass

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

    def apply_prediction(self,
                         prediction: ndarray) -> None:
        """
        Apply network prediction in environment.
        Not mandatory.

        :param prediction: Prediction data.
        """

        pass

    def close(self) -> None:
        """
        Close the Environment.
        Not mandatory.
        """

        pass

    ##########################################################################################
    ##########################################################################################
    #                                   Defining a sample                                    #
    ##########################################################################################
    ##########################################################################################

    def set_training_data(self,
                          input_array: ndarray,
                          output_array: ndarray) -> None:
        """
        Set the training data to send to the TcpIpServer or the EnvironmentManager.

        :param input_array: Network input
        :param output_array: Network expected output
        """

        # Training data is set if the Environment can compute data
        if self.compute_essential_data:
            self.input = input_array
            self.output = output_array

    def set_loss_data(self,
                      loss_data: Any) -> None:
        """
        Set the loss data to send to the TcpIpServer or the EnvironmentManager.

        :param loss_data: Optional data to compute loss.
        """

        # Training data is set if the Environment can compute data
        if self.compute_essential_data:
            self.loss_data = loss_data if type(loss_data) in [list, ndarray] else array([loss_data])

    def set_additional_dataset(self,
                               label: str,
                               data: ndarray) -> None:
        """
        Set additional data fields to store in the dataset.

        :param label: Name of the data field.
        :param data: Data to store.
        """

        # Training data is set if the Environment can compute data
        if self.compute_essential_data:
            self.additional_fields[label] = data if type(data) in [list, ndarray] else array([data])

    def reset_additional_datasets(self) -> None:
        """
        Reset the additional dataset dictionaries.
        """

        self.additional_fields = {}

    ##########################################################################################
    ##########################################################################################
    #                                   Available requests                                   #
    ##########################################################################################
    ##########################################################################################

    def get_prediction(self,
                       input_array: ndarray) -> ndarray:
        """
        Request a prediction from Network.

        :param input_array: Network input.
        :return: Network prediction.
        """

        # If Environment is a TcpIpClient, send request to the Server
        if self.as_tcp_ip_client:
            return TcpIpClient.request_get_prediction(self, input_array=input_array)

        # Otherwise, check the hierarchy of managers
        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        elif self.environment_manager.data_manager.manager is None:
            raise ValueError("Cannot request prediction if Manager does not exist")
        elif not hasattr(self.environment_manager.data_manager.manager, 'network_manager'):
            raise AttributeError("Cannot request prediction if NetworkManager does not exist. If using a data "
                                 "generation pipeline, please disable get_prediction requests.")
        elif self.environment_manager.data_manager.manager.network_manager is None:
            raise ValueError("Cannot request prediction if NetworkManager does not exist")
        # Get a prediction
        return self.environment_manager.data_manager.get_prediction(network_input=input_array[None, ])

    def update_visualisation(self) -> None:
        """
        Triggers the Visualizer update.
        """

        # If Environment is a TcpIpClient, request to the Server
        if self.as_tcp_ip_client:
            self.request_update_visualization()
        self.factory.render()

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseEnvironment object
        """

        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Name: {self.name} n°{self.instance_id}\n"
        description += f"    Comments:\n"
        description += f"    Input size:\n"
        description += f"    Output size:\n"
        return description
