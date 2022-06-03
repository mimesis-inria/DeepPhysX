from typing import Any, Dict, Optional, Type
from os import cpu_count
from os.path import join, dirname
from threading import Thread
from subprocess import run as subprocessRun
from sys import modules, executable

from DeepPhysX.Core.AsyncSocket.TcpIpServer import TcpIpServer
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX.Core.Visualizer.VedoVisualizer import VedoVisualizer


class BaseEnvironmentConfig:
    """
    | BaseEnvironmentConfig is a configuration class to parameterize and create a BaseEnvironment for the
      EnvironmentManager.

    :param Type[BaseEnvironment] environment_class: Class from which an instance will be created
    :param Optional[Type[VedoVisualizer]] visualizer: Class of the Visualizer to use
    :param int simulations_per_step: Number of iterations to compute in the Environment at each time step
    :param int max_wrong_samples_per_step: Maximum number of wrong samples to produce in a step
    :param bool always_create_data: If True, data will always be created from environment. If False, data will be
                                    created from the environment during the first epoch and then re-used from the
                                    Dataset.
    :param bool record_wrong_samples: If True, wrong samples are recorded through Visualizer
    :param int screenshot_sample_rate: A screenshot of the viewer will be done every x sample
    :param bool use_dataset_in_environment: If True, the dataset will always be used in the environment
    :param Optional[Dict[Any, Any]] param_dict: Dictionary containing specific environment parameters
    :param bool as_tcp_ip_client: Environment is owned by a TcpIpClient if True, by an EnvironmentManager if False
    :param int number_of_thread: Number of thread to run
    :param int max_client_connection: Maximum number of handled instances
    :param Optional[str] environment_file: Path of the file containing the Environment class
    :param str ip_address: IP address of the TcpIpObject
    :param int port: Port number of the TcpIpObject
    """

    def __init__(self,
                 environment_class: Type[BaseEnvironment],
                 visualizer: Optional[Type[VedoVisualizer]] = None,
                 simulations_per_step: int = 1,
                 max_wrong_samples_per_step: int = 10,
                 always_create_data: bool = False,
                 record_wrong_samples: bool = False,
                 screenshot_sample_rate: int = 0,
                 use_dataset_in_environment: bool = False,
                 param_dict: Optional[Dict[Any, Any]] = None,
                 as_tcp_ip_client: bool = True,
                 number_of_thread: int = 1,
                 max_client_connection: int = 1000,
                 environment_file: Optional[str] = None,
                 ip_address: str = 'localhost',
                 port: int = 10000):

        if param_dict is None:
            param_dict = {}
        self.name: str = self.__class__.__name__

        # Check simulations_per_step type and value
        if type(simulations_per_step) != int:
            raise TypeError(f"[{self.name}] Wrong simulations_per_step type: int required, get "
                            f"{type(simulations_per_step)}")
        if simulations_per_step < 1:
            raise ValueError(f"[{self.name}] Given simulations_per_step value is negative or null")
        # Check max_wrong_samples_per_step type and value
        if type(max_wrong_samples_per_step) != int:
            raise TypeError(f"[{self.name}] Wrong max_wrong_samples_per_step type: int required, get "
                            f"{type(max_wrong_samples_per_step)}")
        if max_wrong_samples_per_step < 1:
            raise ValueError(f"[{self.name}] Given max_wrong_simulations_per_step value is negative or null")
        # Check always_create_data type
        if type(always_create_data) != bool:
            raise TypeError(f"[{self.name}] Wrong always_create_data type: bool required, get "
                            f"{type(always_create_data)}")
        if type(number_of_thread) != int:
            raise TypeError(f"[{self.name}] The number_of_thread number must be a positive integer.")
        if number_of_thread < 0:
            raise ValueError(f"[{self.name}] The number_of_thread number must be a positive integer.")

        # TcpIpClients parameterization
        self.environment_class: Type[BaseEnvironment] = environment_class
        self.environment_file: str = environment_file if environment_file is not None else modules[
            self.environment_class.__module__].__file__
        self.param_dict: Optional[Dict[Any, Any]] = param_dict
        self.as_tcp_ip_client: bool = as_tcp_ip_client

        # EnvironmentManager parameterization
        self.received_parameters: Dict[Any, Any] = {}
        self.always_create_data: bool = always_create_data
        self.record_wrong_samples: bool = record_wrong_samples
        self.screenshot_sample_rate: int = screenshot_sample_rate
        self.use_dataset_in_environment: bool = use_dataset_in_environment
        self.simulations_per_step: int = simulations_per_step
        self.max_wrong_samples_per_step: int = max_wrong_samples_per_step
        self.visualizer: Optional[Type[VedoVisualizer]] = visualizer

        # TcpIpServer parameterization
        self.ip_address: str = ip_address
        self.port: int = port
        self.server_is_ready: bool = False
        self.number_of_thread: int = min(max(number_of_thread, 1), cpu_count())  # Assert nb is between 1 and cpu_count
        self.max_client_connections = max_client_connection

    def create_server(self, environment_manager: Optional[Any] = None, batch_size: int = 1) -> TcpIpServer:
        """
        | Create a TcpIpServer and launch TcpIpClients in subprocesses.

        :param Optional[Any] environment_manager: EnvironmentManager
        :param int batch_size: Number of sample in a batch
        :return: TcpIpServer object
        """

        # Create server
        server = TcpIpServer(max_client_count=self.max_client_connections, batch_size=batch_size,
                             nb_client=self.number_of_thread, manager=environment_manager,
                             ip_address=self.ip_address, port=self.port)
        server_thread = Thread(target=self.start_server, args=(server,))
        server_thread.start()

        # Create clients
        client_threads = []
        for i in range(self.number_of_thread):
            client_thread = Thread(target=self.start_client, args=(i,))
            client_threads.append(client_thread)
        for client in client_threads:
            client.start()

        # Return server to manager when ready
        while not self.server_is_ready:
            pass
        return server

    def start_server(self, server: TcpIpServer) -> None:
        """
        | Start TcpIpServer.

        :param server: TcpIpServer
        """

        # Allow clients connections
        server.connect()
        # Send and receive parameters with clients
        self.received_parameters = server.initialize(self.param_dict)
        # Server is ready
        self.server_is_ready: bool = True

    def start_client(self, idx: int = 1) -> None:
        """
        | Run a subprocess to start a TcpIpClient.

        :param int idx: Index of client
        """

        script = join(dirname(modules[BaseEnvironment.__module__].__file__), 'launcherBaseEnvironment.py')
        # Usage: python3 script.py <file_path> <environment_class> <ip_address> <port> <idx> <nb_threads>"
        subprocessRun([executable,
                       script,
                       self.environment_file,
                       self.environment_class.__name__,
                       self.ip_address,
                       str(self.port),
                       str(idx),
                       str(self.number_of_thread)])

    def create_environment(self, environment_manager: Any) -> BaseEnvironment:
        """
        | Create an Environment that will not be a TcpIpObject.

        :param Any environment_manager: EnvironmentManager that handles the Environment
        :return: Environment object
        """

        # Create instance
        try:
            environment = self.environment_class(environment_manager=environment_manager, as_tcp_ip_client=False)
        except:
            raise ValueError(f"[{self.name}] Given 'environment_class' cannot be created in {self.name}")
        if not isinstance(environment, BaseEnvironment):
            raise TypeError(f"[{self.name}] Wrong 'environment_class' type: BaseEnvironment required, get "
                            f"{self.environment_class}")
        # Create & Init Environment
        environment.recv_parameters(self.param_dict)
        environment.create()
        environment.init()
        return environment

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """

        # Todo: fields in Configs are the set in Managers or objects, remove __str__ method
        description = "\n"
        description += f"{self.name}\n"
        description += f"    Environment class: {self.environment_class.__name__}\n"
        description += f"    Simulations per step: {self.simulations_per_step}\n"
        description += f"    Max wrong samples per step: {self.max_wrong_samples_per_step}\n"
        description += f"    Always create data: {self.always_create_data}\n"
        return description
