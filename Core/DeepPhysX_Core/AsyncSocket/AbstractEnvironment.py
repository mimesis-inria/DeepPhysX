from typing import Optional, Dict, Any
from numpy import array, ndarray


class AbstractEnvironment:
    """
    | AbstractEnvironment sets the Environment API for TcpIpClient.
    | Do not use AbstractEnvironment to implement a personal Environment, use BaseEnvironment instead.

    :param int instance_id: ID of the instance
    :param int number_of_instances: Number of simultaneously launched instances
    :param bool as_tcp_ip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False
    """

    def __init__(self,
                 as_tcp_ip_client: bool = True,
                 instance_id: int = 0,
                 number_of_instances: int = 1):

        self.name: str = self.__class__.__name__ + f" n°{instance_id}"

        # TcpIpClient variables
        if instance_id > number_of_instances:
            raise ValueError(f"[{self.name}] Instance ID ({instance_id}) is bigger than max instances "
                             f"({number_of_instances}).")
        self.as_tcp_ip_client: bool = as_tcp_ip_client
        self.instance_id: int = instance_id
        self.number_of_instances: int = number_of_instances

        # Training data variables
        self.input: ndarray = array([])
        self.output: ndarray = array([])
        self.loss_data: Any = None
        self.compute_essential_data: bool = True

        # Dataset data variables
        self.sample_in: Optional[ndarray] = None
        self.sample_out: Optional[ndarray] = None
        self.additional_fields: Dict[str, Any] = {}

    def create(self) -> None:
        """
        | Create the Environment. Automatically called when Environment is launched.
        | Must be implemented by user.
        """

        raise NotImplementedError

    def init(self) -> None:
        """
        | Initialize the Environment. Automatically called when Environment is launched.
        | Not mandatory.
        """

        pass

    def close(self) -> None:
        """
        | Close the Environment. Automatically called when Environment is shut down.
        | Not mandatory.
        """

        pass

    async def step(self) -> None:
        """
        | Compute the number of steps in the Environment specified by simulations_per_step in EnvironmentConfig.
        | Must be implemented by user.
        """

        raise NotImplementedError

    def check_sample(self) -> bool:
        """
        | Check if the current produced sample is usable.
        | Not mandatory.

        :return: Current sample can be used or not
        """

        return True

    def recv_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        | Receive simulation parameters set in Config from the Server. Automatically called before create and init.

        :param Dict[str, Any] param_dict: Dictionary of parameters
        """

        pass

    def send_visualization(self) -> dict:
        """
        | Define the visualization objects to send to the Visualizer. Automatically called after create and init.
        | Not mandatory.

        :return: Initial visualization data dictionary
        """

        return {}

    def send_parameters(self) -> dict:
        """
        | Create a dictionary of parameters to send to the manager. Automatically called after create and init.
        | Not mandatory.

        :return: Dictionary of parameters
        """

        return {}

    def apply_prediction(self, prediction: ndarray) -> None:
        """
        | Apply network prediction in environment.
        | Not mandatory.

        :param ndarray prediction: Prediction data
        """

        pass
