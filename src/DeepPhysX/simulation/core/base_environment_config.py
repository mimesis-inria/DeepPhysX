from typing import Any, Optional, Type, Dict, Tuple
from os import cpu_count
from os.path import join, dirname
from threading import Thread
from subprocess import run
from sys import modules, executable

from DeepPhysX.simulation.utils.tcpip_server import TcpIpServer
from DeepPhysX.simulation.core.base_environment_controller import BaseEnvironmentController, BaseEnvironment


class BaseEnvironmentConfig:

    def __init__(self,
                 environment_class: Type[BaseEnvironment],
                 nb_parallel_env: int = 1,
                 simulations_per_step: int = 1,
                 max_wrong_samples_per_step: int = 10,
                 load_samples: bool = False,
                 only_first_epoch: bool = True,
                 always_produce: bool = False,
                 visualizer: Optional[str] = None,
                 record_wrong_samples: bool = False,
                 env_kwargs: Optional[Dict[str, Any]] = None):
        """
        BaseEnvironmentConfig is a configuration class to parameterize and create a BaseEnvironment for the
        EnvironmentManager.

        :param environment_class: Class from which an instance will be created.
        :param nb_parallel_env: Number of simulations to run in parallel.
        :param simulations_per_step: Number of iterations to compute in the Environment at each time step.
        :param max_wrong_samples_per_step: Maximum number of wrong samples to produce in a step.
        :param load_samples: If True, the dataset will always be used in the environment.
        :param only_first_epoch: If True, data will always be created from environment. If False, data will be created
                                 from the environment during the first epoch and then re-used from the Dataset.
        :param always_produce: If True, data will always be produced in Environment(s).
        :param visualizer: Backend of the Visualizer to use.
        :param record_wrong_samples: If True, wrong samples are recorded through Visualizer.
        :param env_kwargs: Additional arguments to pass to the Environment.
        """

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
        # Check only_first_epoch type
        if type(only_first_epoch) != bool:
            raise TypeError(f"[{self.name}] Wrong always_create_data type: bool required, get "
                            f"{type(only_first_epoch)}")
        if nb_parallel_env < 0:
            raise ValueError(f"[{self.name}] The nb_parallel_env number must be a positive integer.")

        # TcpIpClients variables
        self.environment_class: Type[BaseEnvironment] = environment_class
        self.environment_file: str = modules[self.environment_class.__module__].__file__

        # TcpIpServer variables
        self.nb_parallel_env: int = min(max(nb_parallel_env, 1), cpu_count())  # Assert nb is between 1 and cpu_count
        self.server_is_ready: bool = False

        # EnvironmentManager variables
        self.simulations_per_step: int = simulations_per_step
        self.max_wrong_samples_per_step: int = max_wrong_samples_per_step
        self.load_samples: bool = load_samples
        self.only_first_epoch: bool = only_first_epoch
        self.always_produce: bool = always_produce
        self.environment_kwargs: Dict[str, Any] = {} if env_kwargs is None else env_kwargs

        # Visualizer variables
        self.visualizer: Optional[str] = visualizer
        self.record_wrong_samples: bool = record_wrong_samples

    def create_server(self,
                      environment_manager: Optional[Any] = None,
                      batch_size: int = 1,
                      visualization_db: Optional[Tuple[str, str]] = None) -> TcpIpServer:
        """
        Create a TcpIpServer and launch TcpIpClients in subprocesses.

        :param environment_manager: EnvironmentManager.
        :param batch_size: Number of sample in a batch.
        :param visualization_db: Path to the visualization Database to connect to.
        :return: TcpIpServer object.
        """

        # Create server
        server = TcpIpServer(nb_client=self.nb_parallel_env,
                             batch_size=batch_size,
                             manager=environment_manager)
        server_thread = Thread(target=self.start_server, args=(server, visualization_db))
        server_thread.start()

        # Create clients
        client_threads = []
        for i in range(self.nb_parallel_env):
            client_thread = Thread(target=self.start_client, args=(server, i + 1))
            client_threads.append(client_thread)
        for client in client_threads:
            client.start()

        # Return server to manager when it is ready
        while not self.server_is_ready:
            pass
        return server

    def start_server(self,
                     server: TcpIpServer,
                     visualization_db: Optional[Tuple[str, str]] = None) -> None:
        """
        Start TcpIpServer.

        :param server: TcpIpServer.
        :param visualization_db: Path to the visualization Database to connect to.
        """

        server.connect()
        server.initialize(visualization_db=visualization_db,
                          env_kwargs=self.environment_kwargs)
        self.server_is_ready = True

    def start_client(self,
                     server: TcpIpServer,
                     idx: int = 1) -> None:
        """
        Run a subprocess to start a TcpIpClient.

        :param server: TcpIpServer.
        :param idx: Index of client.
        """

        script = join(dirname(modules[BaseEnvironment.__module__].__file__), 'launcher_base_environment.py')
        run([executable, script, self.environment_file, self.environment_class.__name__,
             server.ip_address, str(server.port), str(idx), str(self.nb_parallel_env)])

    def create_environment(self) -> BaseEnvironmentController:
        """
        Create an Environment that will not be a TcpIpObject.

        :return: Environment object.
        """

        return BaseEnvironmentController(environment_class=self.environment_class,
                                         environment_kwargs=self.environment_kwargs)

    def __str__(self):

        description = "\n"
        description += f"{self.name}\n"
        description += f"    Environment class: {self.environment_class.__name__}\n"
        description += f"    Simulations per step: {self.simulations_per_step}\n"
        description += f"    Max wrong samples per step: {self.max_wrong_samples_per_step}\n"
        description += f"    Always create data: {self.only_first_epoch}\n"
        return description