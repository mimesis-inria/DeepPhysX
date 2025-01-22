from typing import Optional, List, Tuple, Type, Dict, Any
from asyncio import run as async_run
from os import cpu_count
from os.path import join, dirname
from sys import modules, executable
from threading import Thread
from subprocess import run

from DeepPhysX.simulation.simulation_controller import SimulationController
from DeepPhysX.simulation.multiprocess.tcpip_server import TcpIpServer
from DeepPhysX.networks.network_manager import NetworkManager
from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.simulation.simulation_controller import DPXSimulation



class SimulationManager:

    def __init__(self,
                 simulation_class: Type[DPXSimulation],
                 simulation_kwargs: Optional[Dict[str, Any]] = None,
                 nb_parallel_env: int = 1,
                 simulations_per_step: int = 1,
                 max_wrong_samples_per_step: int = 10,
                 load_samples: bool = False,
                 only_first_epoch: bool = True,
                 always_produce: bool = False,
                 use_viewer: bool = False):
        """
        """

        self.name: str = self.__class__.__name__

        # Config
        self.simulation_class = simulation_class
        self.simulation_file = modules[self.simulation_class.__module__].__file__
        self.server_is_ready = False
        self.simulation_kwargs = {} if simulation_kwargs is None else simulation_kwargs


        # Data production variables
        self.batch_size: int = 1
        self.only_first_epoch: bool = only_first_epoch
        self.load_samples: bool = load_samples
        self.always_produce: bool = always_produce
        self.simulations_per_step: int = simulations_per_step
        self.max_wrong_samples_per_step: int = max_wrong_samples_per_step
        self.dataset_batch: Optional[List[List[int]]] = None
        self.use_viewer: bool = use_viewer
        self.nb_parallel_env = min(max(nb_parallel_env, 1), cpu_count())
        self.allow_prediction_requests: bool = True


        self.server: Optional[TcpIpServer] = None
        self.environment_controller: Optional[SimulationController] = None
        self.__network_manager: Optional[NetworkManager] = None
        self.__database_manager: Optional[DatabaseManager] = None
        self.get_data = self.__get_data_from_environment
        self.dispatch_batch = self.__dispatch_batch_to_environment

    ################
    # Init methods #
    ################

    def init_data_pipeline(self, batch_size: int) -> None:
        """
        """

        self.allow_prediction_requests = False
        self.init_training_pipeline(batch_size=batch_size)

    def init_training_pipeline(self, batch_size: int) -> None:
        """
        """

        self.batch_size = batch_size

        # Create Server
        if self.nb_parallel_env > 1:
            self.create_server(batch_size=batch_size)
            self.get_data = self.__get_data_from_server
            self.dispatch_batch = self.__dispatch_batch_to_server
        # Create Environment
        else:
            self.create_simulation()

    def init_prediction_pipeline(self) -> None:
        """
        """

        self.nb_parallel_env = 1
        self.create_simulation()

    def create_server(self, batch_size: int):

        # Create server
        self.server = TcpIpServer(nb_client=self.nb_parallel_env,
                                  batch_size=batch_size,
                                  manager=self,
                                  use_viewer=self.use_viewer)
        server_thread = Thread(target=self.start_server)
        server_thread.start()

        # Create clients
        client_threads = []
        for i in range(self.nb_parallel_env):
            client_thread = Thread(target=self.start_client, args=(i + 1,))
            client_threads.append(client_thread)
        for client in client_threads:
            client.start()

        # Return server to manager when it is ready
        while not self.server_is_ready:
            pass

    def start_server(self) -> None:
        """
        Start TcpIpServer.
        """

        self.server.connect()
        self.server.initialize(env_kwargs=self.simulation_kwargs)
        self.server_is_ready = True

    def start_client(self,
                     idx: int) -> None:
        """
        Run a subprocess to start a TcpIpClient.

        :param idx: Index of client.
        """

        script = join(dirname(modules[DPXSimulation.__module__].__file__), 'multiprocess', 'launcher.py')
        run([executable, script, self.simulation_file, self.simulation_class.__name__,
             self.server.ip_address, str(self.server.port), str(idx), str(self.nb_parallel_env)])

    def create_simulation(self):

        self.environment_controller = SimulationController(environment_class=self.simulation_class,
                                                           environment_kwargs=self.simulation_kwargs)
        self.environment_controller.environment_manager = self
        self.environment_controller.create_environment()
        if self.use_viewer:
            self.environment_controller.launch_visualization()


    ##########################################################################################
    ##########################################################################################
    #                              DatabaseHandler management                                #
    ##########################################################################################
    ##########################################################################################

    def connect_to_database(self,
                            database_path: Tuple[str, str],
                            normalize_data: bool):

        if self.environment_controller is not None:
            self.environment_controller.connect_to_database(database_path=database_path, normalize_data=normalize_data)
        elif self.server is not None:
            self.server.connect_to_database(database_path=database_path, normalize_data=normalize_data)

    def connect_to_network_manager(self, network_manager):
        self.__network_manager = network_manager

    def connect_to_database_manager(self, database_manager):
        self.__database_manager = database_manager

    ##########################################################################################
    ##########################################################################################
    #                                Data creation management                                #
    ##########################################################################################
    ##########################################################################################

    def __get_data_from_server(self,
                               animate: bool = True) -> List[List[int]]:
        """
        Compute a batch of data from Environments requested through TcpIpServer.

        :param animate: If True, triggers an environment step.
        """

        return self.server.get_batch(animate)

    def __get_data_from_environment(self,
                                    animate: bool = True,
                                    save_data: bool = True,
                                    request_prediction: bool = False) -> List[List[int]]:
        """
        Compute a batch of data directly from Environment.

        :param animate: If True, triggers an environment step.
        :param save_data: If True, data must be stored in the Database.
        :param request_prediction: If True, a prediction request will be triggered.
        """

        from time import time

        # Produce batch while batch size is not complete
        nb_sample = 0
        dataset_lines = []
        t1, t2, t3 = 0, 0, 0
        while nb_sample < self.batch_size:

            t = time()

            # 1. Send a sample from the Database if one is given
            update_line = None
            if self.dataset_batch is not None:
                update_line = self.dataset_batch.pop(0)
                self.environment_controller.trigger_get_data(line_id=update_line)

            t1 += time() - t
            t = time()

            # 2. Run the defined number of steps
            if animate:
                for current_step in range(self.simulations_per_step):
                    # Sub-steps do not produce data
                    self.environment_controller.compute_training_data = current_step == self.simulations_per_step - 1
                    async_run(self.environment_controller.environment.step())

            t2 += time() - t
            t = time()

            # 3. Add the produced sample index to the batch if the sample is validated
            if self.environment_controller.environment.check_sample():
                nb_sample += 1
                # 3.1. The prediction Pipeline triggers a prediction request
                if request_prediction:
                    self.environment_controller.trigger_prediction()
                # 3.2. Add the data to the Database
                if save_data:
                    # Update the line if the sample was given by the database
                    if update_line is None:
                        new_line = self.environment_controller.trigger_send_data()
                        dataset_lines.append(new_line)
                    # Create a new line otherwise
                    else:
                        self.environment_controller.trigger_update_data(line_id=update_line)
                        dataset_lines.append(update_line)
                # 3.3. Rest the data variables
                self.environment_controller.reset_data()

            t3 += time() - t
        # print(f't1={round(t1, 4)}, t2={round(t2, 4)}, t3={round(t3, 4)}')


        return dataset_lines

    def __dispatch_batch_to_server(self,
                                   data_lines: List[int],
                                   animate: bool = True) -> None:
        """
        Send samples from the Database to the Environments and get back the produced data.

        :param data_lines: Batch of indices of samples.
        :param animate: If True, triggers an environment step.
        """

        # Define the batch to dispatch
        self.server.set_dataset_batch(data_lines)
        # Get data
        self.__get_data_from_server(animate=animate)

    def __dispatch_batch_to_environment(self,
                                        data_lines: Optional[List[int]] = None,
                                        animate: bool = True,
                                        save_data: bool = True,
                                        request_prediction: bool = False) -> None:
        """
        Send samples from the Database to the Environment and get back the produced data.

        :param data_lines: Batch of indices of samples.
        :param animate: If True, triggers an environment step.
        :param save_data: If True, data must be stored in the Database.
        :param request_prediction: If True, a prediction request will be triggered.
        """

        # Define the batch to dispatch
        self.dataset_batch = data_lines.copy() if data_lines is not None else None
        # Get data
        self.__get_data_from_environment(animate=animate,
                                         save_data=save_data,
                                         request_prediction=request_prediction)

    def get_prediction(self, instance_id: int):

        self.__network_manager.get_prediction(instance_id=instance_id)
        # self.__network_manager.compute_online_prediction(instance_id=instance_id)

    ##########################################################################################
    ##########################################################################################
    #                                   Manager behavior                                     #
    ##########################################################################################
    ##########################################################################################

    def close(self) -> None:
        """
        Launch the closing procedure of the EnvironmentManager.
        """

        # Server case
        if self.server is not None:
            self.server.close()

        # Environment case
        if self.environment_controller is not None:
            self.environment_controller.close()

    def __str__(self) -> str:

        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Always create data: {self.only_first_epoch}\n"
        description += f"    Number of threads: {self.nb_parallel_env}\n"
        return description
