from typing import Any, Optional, Dict, List
import os.path
from numpy import ndarray
from json import load as json_load

from DeepPhysX.Core.Manager.DatasetManager import DatasetManager
from DeepPhysX.Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig


class DataManager:

    def __init__(self,
                 dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig,
                 session: str,
                 manager: Optional[Any] = None,
                 new_session: bool = True,
                 training: bool = True,
                 offline: bool = False,
                 store_data: bool = True,
                 batch_size: int = 1):

        """
        DataManager deals with the generation of input / output tensors. His job is to call get_data on either the
        DatasetManager or the EnvironmentManager according to the context.

        :param dataset_config: Specialisation containing the parameters of the dataset manager
        :param environment_config: Specialisation containing the parameters of the environment manager
        :param session: Path to the session directory.
        :param manager: Manager that handle The DataManager
        :param new_session: Define the creation of new directories to store data
        :param training: True if this session is a network training
        :param offline: True if the session is done offline
        :param store_data: Format {\'in\': bool, \'out\': bool} save the tensor when bool is True
        :param batch_size: Number of samples in a batch
        """

        self.name: str = self.__class__.__name__

        # Manager references
        self.manager: Optional[Any] = manager
        self.dataset_manager: Optional[DatasetManager] = None
        self.environment_manager: Optional[EnvironmentManager] = None

        # DataManager parameters
        self.is_training: bool = training
        self.allow_dataset_fetch: bool = True
        self.data: Optional[Dict[str, ndarray]] = None

        # Data normalization coefficients (default: mean = 0, standard deviation = 1)
        self.normalization: Dict[str, List[float]] = {'input': [0., 1.], 'output': [0., 1.]}
        # If normalization flag is set to True, try to load existing coefficients
        if dataset_config is not None and dataset_config.normalize:
            json_file_path = None
            # Existing Dataset in the current session
            if os.path.exists(os.path.join(session, 'dataset')):
                json_file_path = os.path.join(session, 'dataset', 'dataset.json')
            # Dataset provided by config
            elif dataset_config.dataset_dir is not None:
                dataset_dir = dataset_config.dataset_dir
                if dataset_dir[-1] != "/":
                    dataset_dir += "/"
                if dataset_dir[-8:] != "dataset/":
                    dataset_dir += "dataset/"
                if os.path.exists(dataset_dir):
                    json_file_path = os.path.join(dataset_dir, 'dataset.json')
            # If Dataset exists then a json file is associated
            if json_file_path is not None:
                with open(json_file_path) as json_file:
                    json_dict = json_load(json_file)
                    # Get the normalization coefficients
                    for field in self.normalization.keys():
                        if field in json_dict['normalization']:
                            self.normalization[field] = json_dict['normalization'][field]

        # Training
        if self.is_training:
            # Always create a dataset_manager for training
            create_dataset = True
            # Create an environment if prediction must be applied else ask DatasetManager
            create_environment = False
            if environment_config is not None:
                create_environment = None if not environment_config.use_dataset_in_environment else True
        # Prediction
        else:
            # Create an environment for prediction if an environment config is provided
            create_environment = environment_config is not None
            # Create a dataset if data will be stored from environment during prediction
            create_dataset = store_data
            # Create a dataset also if data should be loaded from any partition
            create_dataset = create_dataset or (dataset_config is not None and dataset_config.dataset_dir is not None)

        # Create dataset if required
        if create_dataset:
            self.dataset_manager = DatasetManager(data_manager=self, dataset_config=dataset_config,
                                                  session=session, new_session=new_session, training=self.is_training,
                                                  offline=offline, store_data=store_data)
        # Create environment if required
        if create_environment is None:  # If None then the dataset_manager exists
            create_environment = self.dataset_manager.new_dataset()
        if create_environment:
            self.environment_manager = EnvironmentManager(data_manager=self, environment_config=environment_config,
                                                          session=session, batch_size=batch_size,
                                                          training=self.is_training)

    def get_manager(self) -> Any:
        """
        Return the Manager of this DataManager.

        :return: The Manager of this DataManager.
        """

        return self.manager

    def get_data(self,
                 epoch: int = 0,
                 batch_size: int = 1,
                 animate: bool = True) -> Dict[str, ndarray]:
        """
        Fetch data from the EnvironmentManager or the DatasetManager according to the context.

        :param epoch: Current epoch number.
        :param batch_size: Size of the desired batch.
        :param animate: Allow EnvironmentManager to generate a new sample.
        :return: Dict containing the newly computed data.
        """

        # Training
        if self.is_training:
            data = None
            # Get data from environment if used and if the data should be created at this epoch
            if data is None and self.environment_manager is not None and \
                    (epoch == 0 or self.environment_manager.always_create_data) and self.dataset_manager.new_dataset():
                self.allow_dataset_fetch = False
                data = self.environment_manager.get_data(animate=animate, get_inputs=True, get_outputs=True)
                self.dataset_manager.add_data(data)
            # Force data from the dataset
            else:
                data = self.dataset_manager.get_data(batch_size=batch_size, get_inputs=True, get_outputs=True)
                if self.environment_manager is not None and self.environment_manager.use_dataset_in_environment:
                    new_data = self.environment_manager.dispatch_batch(batch=data)
                    if len(new_data['input']) != 0:
                        data['input'] = new_data['input']
                    if len(new_data['output']) != 0:
                        data['output'] = new_data['output']
                    if 'loss' in new_data:
                        data['loss'] = new_data['loss']
                elif self.environment_manager is not None:
                    # EnvironmentManager is no longer used
                    self.environment_manager.close()
                    self.environment_manager = None

        # Prediction
        else:
            if self.dataset_manager is not None and not self.dataset_manager.new_dataset():
                # Get data from dataset
                data = self.dataset_manager.get_data(batch_size=1, get_inputs=True, get_outputs=True)
                if self.environment_manager is not None:
                    new_data = self.environment_manager.dispatch_batch(batch=data, animate=animate)
                else:
                    new_data = data
                if len(new_data['input']) != 0:
                    data['input'] = new_data['input']
                if len(new_data['output']) != 0:
                    data['output'] = new_data['output']
                if 'loss' in new_data:
                    data['loss'] = new_data['loss']
            else:
                # Get data from environment
                data = self.environment_manager.get_data(animate=animate, get_inputs=True, get_outputs=True)
                # Record data
                if self.dataset_manager is not None:
                    self.dataset_manager.add_data(data)

        self.data = data
        return data

    def get_prediction(self,
                       network_input: ndarray) -> ndarray:
        """
        Get a Network prediction from an input array. Normalization is applied on input and prediction.

        :param network_input: Input array of the Network.
        :return: Network prediction.
        """

        # Apply normalization
        network_input = self.normalize_data(network_input, 'input')
        # Get a prediction
        prediction = self.manager.network_manager.compute_online_prediction(network_input=network_input)
        # Unapply normalization on prediction
        return self.normalize_data(prediction, 'output', reverse=True)

    def apply_prediction(self,
                         prediction: ndarray) -> None:
        """
        Apply the Network prediction in the Environment.

        :param prediction: Prediction of the Network to apply.
        """

        if self.environment_manager is not None:
            # Unapply normalization on prediction
            prediction = self.normalize_data(prediction, 'output', reverse=True)
            # Apply prediction
            self.environment_manager.environment.apply_prediction(prediction)

    def normalize_data(self,
                       data: ndarray,
                       field: str,
                       reverse: bool = False) -> ndarray:
        """
        Apply or unapply normalization following current standard score.

        :param data: Data to normalize.
        :param field: Specify if data is an 'input' or an 'output'.
        :param reverse: If False, apply normalization; if False, unapply normalization.
        :return: Data with applied or misapplied normalization.
        """

        if not reverse:
            # Apply normalization
            return (data - self.normalization[field][0]) / self.normalization[field][1]
        # Unapply normalization
        return (data * self.normalization[field][1]) + self.normalization[field][0]

    def close(self) -> None:
        """
        Launch the closing procedure of Managers.
        """

        if self.environment_manager is not None:
            self.environment_manager.close()
        if self.dataset_manager is not None:
            self.dataset_manager.close()

    def __str__(self) -> str:
        """
        :return: A string containing valuable information about the DataManager
        """

        data_manager_str = ""
        if self.environment_manager:
            data_manager_str += str(self.environment_manager)
        if self.dataset_manager:
            data_manager_str += str(self.dataset_manager)
        return data_manager_str
