from typing import Optional, List, Tuple
from asyncio import run as async_run

from DeepPhysX.simulation.simulation_config import TcpIpServer, SimulationConfig, SimulationController
from DeepPhysX.networks.network_manager import NetworkManager
from DeepPhysX.database.database_manager import DatabaseManager


class SimulationManager:

    def __init__(self,
                 config: SimulationConfig,
                 pipeline: str = '',
                 session: str = 'sessions/default',
                 produce_data: bool = True,
                 batch_size: int = 1):
        """
        EnvironmentManager handle the communication with Environment(s).

        :param config: Configuration object with the parameters of the Environment.
        :param pipeline: Type of the pipeline.
        :param session: Path to the session repository.
        :param produce_data: If True, this session will store data in the Database.
        :param batch_size: Number of samples in a single batch.
        """

        self.name: str = self.__class__.__name__

        # Data production variables
        self.batch_size: int = batch_size
        self.only_first_epoch: bool = config.only_first_epoch
        self.load_samples: bool = config.load_samples
        self.always_produce: bool = config.always_produce
        self.simulations_per_step: int = config.simulations_per_step
        self.max_wrong_samples_per_step: int = config.max_wrong_samples_per_step
        self.allow_prediction_requests: bool = pipeline != 'data_generation'
        self.dataset_batch: Optional[List[List[int]]] = None

        # Create a single Environment or a TcpIpServer
        force_local = pipeline == 'prediction'
        self.nb_parallel_env: int = 1 if force_local else config.nb_parallel_env
        self.server: Optional[TcpIpServer] = None
        self.environment_controller: Optional[SimulationController] = None

        # Create Server
        if self.nb_parallel_env > 1 and not force_local:
            self.server = config.create_server(environment_manager=self,
                                               batch_size=batch_size)

        # Create Environment
        else:
            self.environment_controller = config.create_environment()
            self.environment_controller.environment_manager = self
            self.environment_controller.create_environment()
            if config.use_viewer:
                self.environment_controller.launch_visualization()

        # Define whether methods are used for environment or server
        self.get_data = self.__get_data_from_server if self.server else self.__get_data_from_environment
        self.dispatch_batch = self.__dispatch_batch_to_server if self.server else self.__dispatch_batch_to_environment

        self.__network_manager: Optional[NetworkManager] = None
        self.__database_manager: Optional[DatabaseManager] = None

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
