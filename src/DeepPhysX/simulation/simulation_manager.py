from typing import Optional, List, Tuple, Type, Dict, Any
from os import cpu_count
from os.path import join, dirname
from sys import modules, executable
from threading import Thread
from subprocess import run

from DeepPhysX.simulation.multiprocess.tcpip_server import TcpIpServer
from DeepPhysX.networks.network_manager import NetworkManager
from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.simulation.dpx_simulation import DPXSimulation, SimulationController



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
        SimulationManager handles the numerical simulation(s) to produce synthetic data and communicate with the neural
        network.

        :param simulation_class: The numerical simulation class.
        :param simulation_kwargs: Dict of kwargs to create an instance of the numerical simulation.
        :param nb_parallel_env: Number of numerical simulations to run in parallel.
        :param simulations_per_step: Number of simulation steps to compute before producing a data sample.
        :param max_wrong_samples_per_step:
        :param load_samples:
        :param only_first_epoch:
        :param always_produce:
        :param use_viewer: If True, the viewer will be displayed.
        """

        # Simulation variables
        self.__simulation_class = simulation_class
        self.__simulation_file = modules[self.__simulation_class.__module__].__file__
        self.__simulation_kwargs = {} if simulation_kwargs is None else simulation_kwargs

        # Single Simulation controller variables
        self.simulation_controller: Optional[SimulationController] = None
        self.get_data = self.__get_data_from_simulation
        self.dispatch_batch = self.__dispatch_batch_to_simulation

        # Multi Simulations controller variables
        self.__server: Optional[TcpIpServer] = None
        self.__server_is_ready = False

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

        # Manager variables
        self.__network_manager: Optional[NetworkManager] = None
        self.__database_manager: Optional[DatabaseManager] = None

    ################
    # Init methods #
    ################

    def init_data_pipeline(self, batch_size: int) -> None:
        """
        Init the SimulationManager for the data generation pipeline.

        :param batch_size: Number of sample to produce per batch.
        """

        self.allow_prediction_requests = False
        self.init_training_pipeline(batch_size=batch_size)

    def init_training_pipeline(self, batch_size: int) -> None:
        """
        Init the SimulationManager for the training pipeline.

        :param batch_size: Number of sample to produce per batch.
        """

        self.batch_size = batch_size

        # Create Server
        if self.nb_parallel_env > 1:
            self.__create_server(batch_size=batch_size)
            self.get_data = self.__get_data_from_server
            self.dispatch_batch = self.__dispatch_batch_to_server

        # Create Environment
        else:
            self.__create_simulation()

    def init_prediction_pipeline(self) -> None:
        """
        Init the SimulationManager for the prediction pipeline.
        """

        self.nb_parallel_env = 1
        self.__create_simulation()

    @staticmethod
    def __check_init(foo):
        """
        Wrapper to check that an 'init_*_pipeline' method was called before to use the SimulationManager.
        """

        def wrapper(self, *args, **kwargs):
            if self.__server is None and self.simulation_controller is None:
                raise ValueError(f"[SimulationManager] The manager is not completely initialized; please use one of the "
                                 f"'init_*_pipeline' methods.")
            return foo(self, *args, **kwargs)

        return wrapper

    #############################
    # Controller create methods #
    #############################

    def __create_simulation(self):

        self.simulation_controller = SimulationController(simulation_class=self.__simulation_class,
                                                          simulation_kwargs=self.__simulation_kwargs,
                                                          manager=self)
        self.simulation_controller.create_simulation(use_viewer=self.use_viewer)

    def __create_server(self, batch_size: int):

        # Create server
        self.__server = TcpIpServer(nb_client=self.nb_parallel_env, batch_size=batch_size,
                                    manager=self, use_viewer=self.use_viewer)
        server_thread = Thread(target=self.__start_server)
        server_thread.start()

        # Create clients
        client_threads = []
        for i in range(self.nb_parallel_env):
            client_thread = Thread(target=self.__start_client, args=(i + 1,))
            client_threads.append(client_thread)
        for client in client_threads:
            client.start()

        # Return server to manager when it is ready
        while not self.__server_is_ready:
            pass

    def __start_server(self) -> None:
        """
        Start TcpIpServer.
        """

        self.__server.connect()
        self.__server.initialize(env_kwargs=self.__simulation_kwargs)
        self.__server_is_ready = True

    def __start_client(self, idx: int) -> None:
        """
        Run a subprocess to start a TcpIpClient.

        :param idx: Index of client.
        """

        script = join(dirname(modules[DPXSimulation.__module__].__file__), 'multiprocess', 'launcher.py')
        run([executable, script, self.__simulation_file, self.__simulation_class.__name__,
             self.__server.ip_address, str(self.__server.port), str(idx), str(self.nb_parallel_env)])

    ##############################
    # Database access management #
    ##############################

    def connect_to_database(self,
                            database_path: Tuple[str, str],
                            normalize_data: bool):

        if self.simulation_controller is not None:
            self.simulation_controller.connect_to_database(database_path=database_path, normalize_data=normalize_data)
        elif self.__server is not None:
            self.__server.connect_to_database(database_path=database_path, normalize_data=normalize_data)

    def connect_to_network_manager(self, network_manager):
        self.__network_manager = network_manager

    def connect_to_database_manager(self, database_manager):
        self.__database_manager = database_manager

    #########################
    # Simulation management #
    #########################

    @__check_init
    def __get_data_from_server(self, animate: bool = True) -> List[int]:
        """
        Compute a batch of data from Environments requested through TcpIpServer.

        :param animate: If True, triggers a simulation step.
        """

        return self.__server.get_batch(animate)

    @__check_init
    def __get_data_from_simulation(self,
                                   animate: bool = True,
                                   save_data: bool = True,
                                   request_prediction: bool = False) -> List[int]:
        """
        Compute a batch of data directly from Environment.

        :param animate: If True, triggers a simulation step.
        :param save_data: If True, data must be stored in the Database.
        :param request_prediction: If True, a prediction request will be triggered.
        """

        # Produce batch while batch size is not complete
        nb_sample = 0
        dataset_lines = []
        while nb_sample < self.batch_size:

            # 1. Send a sample from the Database if one is given
            update_line = None
            if self.dataset_batch is not None:
                update_line = self.dataset_batch.pop(0)
                self.simulation_controller.trigger_get_data(line_id=update_line)

            # 2. Run the defined number of steps
            if animate:
                for current_step in range(self.simulations_per_step):
                    # Sub-steps do not produce data
                    self.simulation_controller.compute_training_data = current_step == self.simulations_per_step - 1
                    self.simulation_controller.simulation.step()

            # 3. Add the produced sample index to the batch if the sample is validated
            if self.simulation_controller.simulation.check_sample():
                nb_sample += 1
                # 3.1. The prediction Pipeline triggers a prediction request
                if request_prediction:
                    self.simulation_controller.trigger_prediction()
                # 3.2. Add the data to the Database
                if save_data:
                    # Update the line if the sample was given by the database
                    if update_line is None:
                        new_line = self.simulation_controller.trigger_send_data()
                        dataset_lines.append(new_line)
                    # Create a new line otherwise
                    else:
                        self.simulation_controller.trigger_update_data(line_id=update_line)
                        dataset_lines.append(update_line)
                # 3.3. Rest the data variables
                self.simulation_controller.reset_data()

        return dataset_lines

    @__check_init
    def __dispatch_batch_to_server(self,
                                   data_lines: List[int],
                                   animate: bool = True) -> None:
        """
        Send samples from the Database to the Environments and get back the produced data.

        :param data_lines: Batch of indices of samples.
        :param animate: If True, triggers a simulation step.
        """

        # Define the batch to dispatch
        self.__server.set_dataset_batch(data_lines)
        # Get data
        self.__get_data_from_server(animate=animate)

    @__check_init
    def __dispatch_batch_to_simulation(self,
                                       data_lines: Optional[List[int]] = None,
                                       animate: bool = True,
                                       save_data: bool = True,
                                       request_prediction: bool = False) -> None:
        """
        Send samples from the Database to the Environment and get back the produced data.

        :param data_lines: Batch of indices of samples.
        :param animate: If True, triggers a simulation step.
        :param save_data: If True, data must be stored in the Database.
        :param request_prediction: If True, a prediction request will be triggered.
        """

        # Define the batch to dispatch
        self.dataset_batch = data_lines.copy() if data_lines is not None else None
        # Get data
        self.__get_data_from_simulation(animate=animate,
                                        save_data=save_data,
                                        request_prediction=request_prediction)

    @__check_init
    def get_prediction(self, instance_id: int):

        self.__network_manager.get_prediction(instance_id=instance_id)

    @__check_init
    def is_viewer_open(self) -> bool:

        if self.simulation_controller.simulation.viewer is None:
            return False
        return self.simulation_controller.simulation.viewer.is_open

    ###################
    # Manager methods #
    ###################

    def close(self) -> None:
        """
        Launch the closing procedure of the EnvironmentManager.
        """

        # Server case
        if self.__server is not None:
            self.__server.close()

        # Environment case
        if self.simulation_controller is not None:
            self.simulation_controller.close()

    def __str__(self) -> str:

        desc = "\n"
        desc += f"# SIMULATION MANAGER\n"
        desc += f"    Always create data: {self.only_first_epoch}\n"
        desc += f"    Number of threads: {self.nb_parallel_env}\n"
        return desc
