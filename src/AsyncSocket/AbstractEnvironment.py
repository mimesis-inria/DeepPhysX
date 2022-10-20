from typing import Optional, Dict, Any
from numpy import ndarray

from SSD.Core.Storage.Database import Database


class AbstractEnvironment:

    def __init__(self,
                 as_tcp_ip_client: bool = True,
                 instance_id: int = 0,
                 number_of_instances: int = 1):
        """
        AbstractEnvironment sets the Environment API for TcpIpClient.
        Do not use AbstractEnvironment to implement a personal Environment, use BaseEnvironment instead.

        :param instance_id: ID of the instance.
        :param number_of_instances: Number of simultaneously launched instances.
        :param as_tcp_ip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False.
        """

        self.name: str = self.__class__.__name__ + f" n°{instance_id}"

        # TcpIpClient variables
        if instance_id > number_of_instances:
            raise ValueError(f"[{self.name}] Instance ID ({instance_id}) is bigger than max instances "
                             f"({number_of_instances}).")
        self.as_tcp_ip_client: bool = as_tcp_ip_client
        self.instance_id: int = instance_id
        self.number_of_instances: int = number_of_instances

        # Training data variables
        self.__training_data: Dict[str, ndarray] = {}
        self.__additional_data: Dict[str, ndarray] = {}
        self.compute_training_data: bool = True

        # Dataset data variables
        self.database: Optional[Database] = None
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
        pass

    def init_database(self) -> None:
        raise NotImplementedError

    def init_visualization(self) -> None:
        pass

    ##########################################################################################
    ##########################################################################################
    #                                 Environment behavior                                   #
    ##########################################################################################
    ##########################################################################################

    async def step(self) -> None:
        raise NotImplementedError

    def check_sample(self) -> bool:
        return True

    def apply_prediction(self, prediction: Dict[str, ndarray]) -> None:
        pass

    def close(self) -> None:
        pass

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

    def _get_training_data(self, line: int) -> None:
        raise NotImplementedError
