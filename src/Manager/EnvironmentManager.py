from typing import Any, Dict, Optional, Union
from numpy import ndarray
from asyncio import run as async_run
from copy import copy
from os.path import join

from SSD.Core.Storage.Database import Database

from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig, TcpIpServer, BaseEnvironment
from DeepPhysX.Core.Visualization.VedoVisualizer import VedoVisualizer


class EnvironmentManager:

    def __init__(self,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 data_manager: Any = None,
                 session: str = 'sessions/default',
                 data_db: Optional[Database] = None,
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
        self.always_create_data: bool = environment_config.always_create_data
        self.use_dataset_in_environment: bool = environment_config.use_dataset_in_environment
        self.simulations_per_step: int = environment_config.simulations_per_step
        self.max_wrong_samples_per_step: int = environment_config.max_wrong_samples_per_step
        self.dataset_batch: Optional[Dict[str, Dict[int, Any]]] = None

        # Create the Visualizer
        self.visualizer: Optional[VedoVisualizer] = None
        visu_db = None
        if environment_config.visualizer is not None:
            self.visualizer = environment_config.visualizer(database_dir=join(session, 'dataset'),
                                                            database_name='Visualization',
                                                            remote=environment_config.as_tcp_ip_client)
            visu_db = self.visualizer.get_database()

        # Create a single Environment or a TcpIpServer
        self.number_of_thread: int = environment_config.number_of_thread
        self.server: Optional[TcpIpServer] = None
        self.environment: Optional[BaseEnvironment] = None
        if environment_config.as_tcp_ip_client:
            self.server = environment_config.create_server(environment_manager=self,
                                                           batch_size=batch_size,
                                                           data_db=data_db if data_db is None else data_db.get_path(),
                                                           visu_db=visu_db if visu_db is None else visu_db.get_path())
            self.data_manager.get_database().load()
        else:
            self.environment = environment_config.create_environment(environment_manager=self,
                                                                     data_db=data_db,
                                                                     visu_db=visu_db)

        # Define get_data and dispatch methods
        self.get_data = self.get_data_from_server if self.server else self.get_data_from_environment
        self.dispatch_batch = self.dispatch_batch_to_server if self.server else self.dispatch_batch_to_environment

        # Init visualizer
        if self.visualizer is not None:
            if len(self.visualizer.get_database().get_tables()) == 1:
                self.visualizer.get_database().load()
            self.visualizer.init_visualizer()

    def get_data_manager(self) -> Any:
        """
        Get the DataManager of this EnvironmentManager.

        :return: The DataManager of this EnvironmentManager.
        """

        return self.data_manager

    def get_data_from_server(self,
                             animate: bool = True) -> None:
        """
        Compute a batch of data from Environments requested through TcpIpServer.

        :param  animate: If True, triggers an environment step.
        """

        # Get data from server
        self.server.get_batch(animate)

    def get_data_from_environment(self,
                                  animate: bool = True) -> None:
        """
        Compute a batch of data directly from Environment.

        :param animate: If True, triggers an environment step.
        """

        # 1. Produce batch while batch size is not complete
        nb_sample = 0
        while nb_sample < self.batch_size:

            # # 1.1 Send a sample if a batch from dataset is given
            # if self.dataset_batch is not None:
            #     # Extract a sample from dataset batch: input
            #     self.environment.sample_in = self.dataset_batch['input'][0]
            #     self.dataset_batch['input'] = self.dataset_batch['input'][1:]
            #     # Extract a sample from dataset batch: output
            #     self.environment.sample_out = self.dataset_batch['output'][0]
            #     self.dataset_batch['output'] = self.dataset_batch['output'][1:]
            #     # Extract a sample from dataset batch: additional fields
            #     additional_fields = {}
            #     if 'additional_fields' in self.dataset_batch:
            #         for field in self.dataset_batch['additional_fields']:
            #             additional_fields[field] = self.dataset_batch['additional_fields'][field][0]
            #             self.dataset_batch['additional_fields'][field] = self.dataset_batch['additional_fields'][field][1:]
            #     self.environment.additional_fields = additional_fields

            # 1.2 Run the defined number of step
            if animate:
                for current_step in range(self.simulations_per_step):
                    # Sub-steps do not produce data
                    self.environment.compute_training_data = current_step == self.simulations_per_step - 1
                    async_run(self.environment.step())

            # 1.3 Add the produced sample to the batch if the sample is validated
            if self.environment.check_sample():
                nb_sample += 1
                self.environment._send_training_data()
                self.environment._reset_training_data()

    def dispatch_batch_to_server(self,
                                 batch: Dict[str, Union[ndarray, dict]],
                                 animate: bool = True) -> Dict[str, Union[ndarray, dict]]:
        """
        Send samples from dataset to the Environments. Get back the training data.

        :param batch: Batch of samples.
        :param animate: If True, triggers an environment step.
        :return: Batch of training data.
        """

        # Define the batch to dispatch
        self.server.set_dataset_batch(batch)
        # Empty the server queue
        while not self.server.data_fifo.empty():
            self.server.data_fifo.get()
        # Get data
        return self.get_data(animate=animate)

    def dispatch_batch_to_environment(self,
                                      batch: Dict[str, Union[ndarray, dict]],
                                      animate: bool = True) -> Dict[str, Union[ndarray, dict]]:
        """
        Send samples from dataset to the Environments. Get back the training data.

        :param batch: Batch of samples.
        :param animate: If True, triggers an environment step.
        :return: Batch of training data.
        """

        # Define the batch to dispatch
        self.dataset_batch = copy(batch)
        # Get data
        return self.get_data(animate=animate)

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
        description += f"    Always create data: {self.always_create_data}\n"
        # description += f"    Record wrong samples: {self.record_wrong_samples}\n"
        description += f"    Number of threads: {self.number_of_thread}\n"
        # description += f"    Managed objects: Environment: {self.environment.env_name}\n"
        # Todo: manage the print log of each Environment since they can have different parameters
        # description += str(self.environment)
        return description
