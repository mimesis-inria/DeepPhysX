from typing import Any, Optional, List
from asyncio import run as async_run
from os.path import join

from SSD.Core.Storage.Database import Database

from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig, TcpIpServer, BaseEnvironment
from DeepPhysX.Core.Visualization.VedoVisualizer import VedoVisualizer


class EnvironmentManager:

    def __init__(self,
                 environment_config: BaseEnvironmentConfig,
                 data_manager: Optional[Any] = None,
                 session: str = 'sessions/default',
                 batch_size: int = 1):
        """
        Deals with the online generation of data for both training and running of the neural networks.

        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param session: Path to the session directory.
        :param data_manager: DataManager that handles the EnvironmentManager.
        :param batch_size: Number of samples in a batch of data.
        """

        self.name: str = self.__class__.__name__

        # Managers architecture
        self.data_manager: Any = data_manager

        # Data producing parameters
        self.batch_size: int = batch_size
        self.only_first_epoch: bool = environment_config.only_first_epoch
        self.load_samples: bool = environment_config.load_samples
        self.simulations_per_step: int = environment_config.simulations_per_step
        self.max_wrong_samples_per_step: int = environment_config.max_wrong_samples_per_step
        self.dataset_batch: Optional[List[int]] = None

        # Create the Visualizer
        self.visualizer: Optional[VedoVisualizer] = None
        visualization_db = None
        if environment_config.visualizer is not None:
            self.visualizer = environment_config.visualizer(database_dir=join(session, 'dataset'),
                                                            database_name='Visualization',
                                                            remote=environment_config.as_tcp_ip_client)
            visualization_db = self.visualizer.get_database()

        # Create a single Environment or a TcpIpServer
        self.number_of_thread: int = environment_config.number_of_thread
        self.server: Optional[TcpIpServer] = None
        self.environment: Optional[BaseEnvironment] = None
        if environment_config.as_tcp_ip_client:
            self.server = environment_config.create_server(environment_manager=self,
                                                           batch_size=batch_size,
                                                           visualization_db=None if visualization_db is None else visualization_db.get_path())
            self.data_manager.get_database().load()
        else:
            self.environment = environment_config.create_environment(visualization_db=visualization_db)
            self.environment.environment_manager = self
            self.data_manager.connect_handler(self.environment.get_database_handler())

            # Create & Init Environment
            self.environment.create()
            self.environment.init()
            self.environment.init_database()
            self.environment.init_visualization()


        # Define get_data and dispatch methods
        self.get_database_handler = self.get_server_database_handler if self.server else self.get_environment_database_handler
        self.change_database = self.change_database_in_server if self.server else self.change_database_in_environment
        self.get_data = self.get_data_from_server if self.server else self.get_data_from_environment
        self.dispatch_batch = self.dispatch_batch_to_server if self.server else self.dispatch_batch_to_environment

        # Init visualizer
        if self.visualizer is not None:
            if len(self.visualizer.get_database().get_tables()) == 1:
                self.visualizer.get_database().load()
            self.visualizer.init_visualizer()

    def get_server_database_handler(self):
        return self.server.get_database_handler()

    def get_environment_database_handler(self):
        return self.environment.get_database_handler()

    def change_database_in_server(self, database: Database):
        self.server.change_database(database.get_path())

    def change_database_in_environment(self, database):
        self.environment.database = database

    def get_data_from_server(self,
                             animate: bool = True) -> List[int]:
        """
        Compute a batch of data from Environments requested through TcpIpServer.

        :param  animate: If True, triggers an environment step.
        """

        # Get data from server
        return self.server.get_batch(animate)

    def get_data_from_environment(self,
                                  animate: bool = True,
                                  request_prediction: bool = False) -> List[int]:
        """
        Compute a batch of data directly from Environment.

        :param animate: If True, triggers an environment step.
        """

        # 1. Produce batch while batch size is not complete
        nb_sample = 0
        dataset_lines = []
        while nb_sample < self.batch_size:

            # 1.1 Send a sample if a batch from dataset is given
            update_line = None
            if self.dataset_batch is not None:
                update_line = self.dataset_batch.pop(0)
                self.environment._get_training_data(update_line)

            # 1.2 Run the defined number of step
            if animate:
                for current_step in range(self.simulations_per_step):
                    # Sub-steps do not produce data
                    self.environment.compute_training_data = current_step == self.simulations_per_step - 1
                    async_run(self.environment.step())

            # 1.3 Add the produced sample to the batch if the sample is validated
            if self.environment.check_sample():
                nb_sample += 1
                if update_line is None:
                    new_line = self.environment._send_training_data()
                    dataset_lines.append(new_line)
                else:
                    self.environment._update_training_data(update_line)
                    dataset_lines.append(update_line)
                if request_prediction:
                    self.environment._get_prediction()
                self.environment._reset_training_data()

        return dataset_lines

    def dispatch_batch_to_server(self,
                                 data_lines: List[int],
                                 animate: bool = True) -> None:
        """
        Send samples from dataset to the Environments. Get back the training data.

        :param data_lines: Batch of samples.
        :param animate: If True, triggers an environment step.
        :return: Batch of training data.
        """

        # Define the batch to dispatch
        self.server.set_dataset_batch(data_lines)
        # Get data
        self.get_data_from_server(animate=animate)

    def dispatch_batch_to_environment(self,
                                      data_lines: List[int],
                                      animate: bool = True,
                                      request_prediction: bool = False) -> None:
        """
        Send samples from dataset to the Environments. Get back the training data.

        :param batch: Batch of samples.
        :param animate: If True, triggers an environment step.
        :return: Batch of training data.
        """

        # Define the batch to dispatch
        self.dataset_batch = data_lines.copy()
        # Get data
        self.get_data_from_environment(animate=animate,
                                       request_prediction=request_prediction)

    def request_prediction(self):

        self.environment._get_prediction()

    def update_visualizer(self,
                          instance: int) -> None:
        """
        Update the Visualizer.

        :param instance: Index of the Environment render to update.
        """

        if self.visualizer is not None:
            self.visualizer.render_instance(instance)

    def close(self) -> None:
        """
        Close the environment
        """

        # Server case
        if self.server:
            self.server.close()
        # No server case
        if self.environment:
            self.environment.close()

    def __str__(self) -> str:
        """
        :return: A string containing valuable information about the EnvironmentManager
        """

        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Always create data: {self.only_first_epoch}\n"
        # description += f"    Record wrong samples: {self.record_wrong_samples}\n"
        description += f"    Number of threads: {self.number_of_thread}\n"
        # description += f"    Managed objects: Environment: {self.environment.env_name}\n"
        # description += str(self.environment)
        return description
