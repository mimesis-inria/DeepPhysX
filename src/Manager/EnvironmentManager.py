from typing import Any, Dict, Optional, Union
from numpy import array, ndarray
from asyncio import run as async_run
from copy import copy
from os.path import join

from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig, TcpIpServer, BaseEnvironment
from DeepPhysX.Core.Visualization.VedoVisualizer import VedoVisualizer


class EnvironmentManager:

    def __init__(self,
                 environment_config: BaseEnvironmentConfig,
                 session: str,
                 data_manager: Any = None,
                 batch_size: int = 1,
                 training: bool = True):
        """
        Deals with the online generation of data for both training and running of the neural networks.

        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param session: Path to the session directory.
        :param data_manager: DataManager that handles the EnvironmentManager.
        :param batch_size: Number of samples in a batch of data.
        :param training: True if this session is a network training.
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
        self.train: bool = training
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
                                                           visu_db=visu_db.get_path())
        else:
            self.environment = environment_config.create_environment(environment_manager=self,
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
                             get_inputs: bool = True,
                             get_outputs: bool = True,
                             animate: bool = True) -> Dict[str, Union[ndarray, dict]]:
        """
        Compute a batch of data from Environments requested through TcpIpServer.

        :param get_inputs: If True, compute and return input.
        :param get_outputs: If True, compute and return output.
        :param  animate: If True, triggers an environment step.
        :return: Dictionary containing all labeled data sent by the clients in their own dictionary + in and out key
                 corresponding to the batch.
        """

        # Get data from server
        batch = self.server.get_batch(get_inputs, get_outputs, animate)
        # Filter input and output
        training_data = {'input': array(batch['input']) if get_inputs else array([]),
                         'output': array(batch['output']) if get_outputs else array([])}
        # Convert each additional field
        for field in batch['additional_fields']:
            batch['additional_fields'][field] = array(batch['additional_fields'][field])
        training_data['additional_fields'] = batch['additional_fields']
        # Convert loss data
        if 'loss' in batch and len(batch['loss']) != 0:
            training_data['loss'] = array(batch['loss'])
        # Return batch
        return training_data

    def get_data_from_environment(self,
                                  get_inputs: bool = True,
                                  get_outputs: bool = True,
                                  animate: bool = True) -> Dict[str, Union[ndarray, dict]]:
        """
        Compute a batch of data directly from Environment.

        :param get_inputs: If True, compute and return input.
        :param get_outputs: If True, compute and return output.
        :param animate: If True, triggers an environment step.
        :return: Dictionary containing all labeled data sent by the clients in their own dictionary + in and out key
                 corresponding to the batch.
        """

        # Init training data container, define production conditions
        input_condition = lambda x: len(x) < self.batch_size if get_inputs else lambda _: False
        output_condition = lambda x: len(x) < self.batch_size if get_outputs else lambda _: False
        training_data = {'input': [], 'output': []}

        # 1. Produce batch while batch size is not complete
        while input_condition(training_data['input']) and output_condition(training_data['output']):

            # 1.1 Send a sample if a batch from dataset is given
            if self.dataset_batch is not None:
                # Extract a sample from dataset batch: input
                self.environment.sample_in = self.dataset_batch['input'][0]
                self.dataset_batch['input'] = self.dataset_batch['input'][1:]
                # Extract a sample from dataset batch: output
                self.environment.sample_out = self.dataset_batch['output'][0]
                self.dataset_batch['output'] = self.dataset_batch['output'][1:]
                # Extract a sample from dataset batch: additional fields
                additional_fields = {}
                if 'additional_fields' in self.dataset_batch:
                    for field in self.dataset_batch['additional_fields']:
                        additional_fields[field] = self.dataset_batch['additional_fields'][field][0]
                        self.dataset_batch['additional_fields'][field] = self.dataset_batch['additional_fields'][field][1:]
                self.environment.additional_fields = additional_fields

            # 1.2 Run the defined number of step
            if animate:
                for current_step in range(self.simulations_per_step):
                    # Sub-steps do not produce data
                    self.environment.compute_essential_data = current_step == self.simulations_per_step - 1
                    async_run(self.environment.step())

            # 1.3 Add the produced sample to the batch if the sample is validated
            if self.environment.check_sample():
                # Network's input
                if get_inputs:
                    training_data['input'].append(self.environment.input)
                    self.environment.input = array([])
                # Network's output
                if get_outputs:
                    training_data['output'].append(self.environment.output)
                    self.environment.output = array([])
                # Check if there is loss data
                if self.environment.loss_data:
                    if 'loss' not in training_data:
                        training_data['loss'] = []
                    training_data['loss'].append(self.environment.loss_data)
                    self.environment.loss_data = None
                # Check if there is additional dataset fields
                if self.environment.additional_fields != {}:
                    if 'additional_fields' not in training_data:
                        training_data['additional_fields'] = {}
                    for field in self.environment.additional_fields:
                        if field not in training_data['additional_fields']:
                            training_data['additional_fields'][field] = []
                        training_data['additional_fields'][field].append(self.environment.additional_fields[field])
                    self.environment.additional_fields = {}

        # 2. Convert data in ndarray
        for key in training_data:
            # If key does not contain a dict, convert value directly
            if key != 'additional_fields':
                training_data[key] = array(training_data[key])
            # If key contains a dict, convert item by item
            else:
                for field in training_data[key]:
                    training_data[key][field] = array(training_data[key][field])

        return training_data

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
