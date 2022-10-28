from typing import Any, Optional, List
from asyncio import run as async_run
from os.path import join

from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig, TcpIpServer, BaseEnvironment
from DeepPhysX.Core.Database.DatabaseHandler import DatabaseHandler
from DeepPhysX.Core.Visualization.VedoVisualizer import VedoVisualizer


class EnvironmentManager:

    def __init__(self,
                 environment_config: BaseEnvironmentConfig,
                 data_manager: Optional[Any] = None,
                 pipeline: str = '',
                 session: str = 'sessions/default',
                 batch_size: int = 1):
        """
        EnvironmentManager handle the communication with Environment(s).

        :param environment_config: Configuration object with the parameters of the Environment.
        :param data_manager: DataManager that handles the EnvironmentManager.
        :param pipeline: Type of the pipeline.
        :param session: Path to the session repository.
        :param batch_size: Number of samples in a single batch.
        """

        self.name: str = self.__class__.__name__

        # Session variables
        self.data_manager: Any = data_manager

        # Data production variables
        self.batch_size: int = batch_size
        self.only_first_epoch: bool = environment_config.only_first_epoch
        self.load_samples: bool = environment_config.load_samples
        self.always_produce: bool = environment_config.always_produce
        self.simulations_per_step: int = environment_config.simulations_per_step
        self.max_wrong_samples_per_step: int = environment_config.max_wrong_samples_per_step
        self.allow_prediction_requests: bool = pipeline != 'data_generation'
        self.dataset_batch: Optional[List[List[int]]] = None

        # Create a Visualizer to provide the visualization Database
        self.visualizer: Optional[VedoVisualizer] = None
        visualization_db = None
        if environment_config.visualizer is not None:
            self.visualizer = environment_config.visualizer(database_dir=join(session, 'dataset'),
                                                            database_name='Visualization',
                                                            remote=environment_config.as_tcp_ip_client)
            visualization_db = self.visualizer.get_database()

        # Create a single Environment or a TcpIpServer
        force_local = pipeline == 'prediction'
        self.number_of_thread: int = 1 if force_local else environment_config.number_of_thread
        self.server: Optional[TcpIpServer] = None
        self.environment: Optional[BaseEnvironment] = None
        # Create Server
        if environment_config.as_tcp_ip_client and not force_local:
            self.server = environment_config.create_server(environment_manager=self,
                                                           batch_size=batch_size,
                                                           visualization_db=None if visualization_db is None else
                                                           visualization_db.get_path())
        # Create Environment
        else:
            self.environment = environment_config.create_environment(visualization_db=visualization_db)
            self.environment.environment_manager = self
            self.data_manager.connect_handler(self.environment.get_database_handler())
            self.environment.create()
            self.environment.init()
            self.environment.init_database()
            self.environment.init_visualization()

        # Define whether methods are used for environment or server
        self.get_database_handler = self.__get_server_db_handler if self.server else self.__get_environment_db_handler
        self.get_data = self.__get_data_from_server if self.server else self.__get_data_from_environment
        self.dispatch_batch = self.__dispatch_batch_to_server if self.server else self.__dispatch_batch_to_environment

        # Init the Visualizer once Environments are initialized
        if self.visualizer is not None:
            if len(self.visualizer.get_database().get_tables()) == 1:
                self.visualizer.get_database().load()
            self.visualizer.init_visualizer()

    ##########################################################################################
    ##########################################################################################
    #                              DatabaseHandler management                                #
    ##########################################################################################
    ##########################################################################################

    def __get_server_db_handler(self) -> DatabaseHandler:
        """
        Get the DatabaseHandler of the TcpIpServer.
        """

        return self.server.get_database_handler()

    def __get_environment_db_handler(self) -> DatabaseHandler:
        """
        Get the DatabaseHandler of the Environment.
        """

        return self.environment.get_database_handler()

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

        # Produce batch while batch size is not complete
        nb_sample = 0
        dataset_lines = []
        while nb_sample < self.batch_size:

            # 1. Send a sample from the Database if one is given
            update_line = None
            if self.dataset_batch is not None:
                update_line = self.dataset_batch.pop(0)
                self.environment._get_training_data(update_line)

            # 2. Run the defined number of steps
            if animate:
                for current_step in range(self.simulations_per_step):
                    # Sub-steps do not produce data
                    self.environment.compute_training_data = current_step == self.simulations_per_step - 1
                    async_run(self.environment.step())

            # 3. Add the produced sample index to the batch if the sample is validated
            if self.environment.check_sample():
                nb_sample += 1
                # 3.1. The prediction Pipeline triggers a prediction request
                if request_prediction:
                    self.environment._get_prediction()
                # 3.2. Add the data to the Database
                if save_data:
                    # Update the line if the sample was given by the database
                    if update_line is None:
                        new_line = self.environment._send_training_data()
                        dataset_lines.append(new_line)
                    # Create a new line otherwise
                    else:
                        self.environment._update_training_data(update_line)
                        dataset_lines.append(update_line)
                # 3.3. Rest the data variables
                self.environment._reset_training_data()

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
                                        data_lines: List[int],
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
        self.dataset_batch = data_lines.copy()
        # Get data
        self.__get_data_from_environment(animate=animate,
                                         save_data=save_data,
                                         request_prediction=request_prediction)

    ##########################################################################################
    ##########################################################################################
    #                                   Requests management                                  #
    ##########################################################################################
    ##########################################################################################

    def update_visualizer(self,
                          instance: int) -> None:
        """
        Update the Visualizer.

        :param instance: Index of the Environment render to update.
        """

        if self.visualizer is not None:
            self.visualizer.render_instance(instance)

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
        if self.server:
            self.server.close()

        # Environment case
        if self.environment:
            self.environment.close()

    def __str__(self) -> str:

        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Always create data: {self.only_first_epoch}\n"
        description += f"    Number of threads: {self.number_of_thread}\n"
        return description
