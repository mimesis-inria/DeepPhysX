from typing import Optional, Dict, Any, Union, Tuple
from numpy import ndarray

from SSD.Core.Storage.Database import Database


class AbstractEnvironment:

    def __init__(self,
                 as_tcp_ip_client: bool = True,
                 instance_id: int = 0,
                 instance_nb: int = 1,
                 visualization_db: Optional[Union[Database, Tuple[str, str]]] = None):
        """
        AbstractEnvironment sets the Environment API for TcpIpClient.
        Do not use AbstractEnvironment to implement an Environment, use BaseEnvironment instead.

        :param as_tcp_ip_client: If True, the Environment is a TcpIpObject, else it is owned by an EnvironmentManager.
        :param instance_id: Index of this instance.
        :param instance_nb: Number of simultaneously launched instances.
        """

        self.name: str = self.__class__.__name__ + f" n°{instance_id}"

        # TcpIpClient variables
        if instance_id > instance_nb:
            raise ValueError(f"[{self.name}] Instance ID ({instance_id}) is bigger than max instances "
                             f"({instance_nb}).")
        self.as_tcp_ip_client: bool = as_tcp_ip_client
        self.instance_id: int = instance_id
        self.instance_nb: int = instance_nb

        # Manager of the Environment
        self.environment_manager: Any = None
        self.tcp_ip_client: Any = None

        # Training data variables
        self.compute_training_data: bool = True

        # Dataset data variables
        self.update_line: Optional[int] = None
        self.sample_training: Optional[Dict[str, Any]] = None
        self.sample_additional: Optional[Dict[str, Any]] = None

    ##########################################################################################
    ##########################################################################################
    #                                 Initializing Environment                               #
    ##########################################################################################
    ##########################################################################################

    def create(self) -> None:
        raise NotImplementedError

    def init(self) -> None:
        raise NotImplementedError

    def init_database(self) -> None:
        raise NotImplementedError

    def init_visualization(self) -> None:
        raise NotImplementedError

    ##########################################################################################
    ##########################################################################################
    #                                 Environment behavior                                   #
    ##########################################################################################
    ##########################################################################################

    async def step(self) -> None:
        raise NotImplementedError

    def check_sample(self) -> bool:
        raise NotImplementedError

    def apply_prediction(self, prediction: Dict[str, ndarray]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    ##########################################################################################
    ##########################################################################################
    #                                   Defining a sample                                    #
    ##########################################################################################
    ##########################################################################################

    def _send_training_data(self) -> None:
        raise NotImplementedError

    def _reset_training_data(self) -> None:
        raise NotImplementedError

    def _update_training_data(self, line_id: int) -> None:
        raise NotImplementedError

    def _get_training_data(self, line_id: int) -> None:
        raise NotImplementedError
