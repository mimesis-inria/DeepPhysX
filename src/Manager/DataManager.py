from typing import Any, Optional, Dict, List, Union
from numpy import ndarray

from DeepPhysX.Core.Manager.DatabaseManager import DatasetManager, Database
from DeepPhysX.Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Database.BaseDatasetConfig import BaseDatasetConfig


class DataManager:

    def __init__(self,
                 dataset_config: Optional[BaseDatasetConfig] = None,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 manager: Optional[Any] = None,
                 session: str = 'sessions/default',
                 new_session: bool = True,
                 pipeline: str = '',
                 produce_data: bool = True,
                 batch_size: int = 1):

        """
        DataManager deals with the generation, storage and loading of training data.
        A batch is given with a call to 'get_data' on either the DatasetManager or the EnvironmentManager according to
        the context.

        :param dataset_config: Specialisation containing the parameters of the dataset manager.
        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param manager: Manager that handle the DataManager
        :param session: Path to the session directory.
        :param bool new_session: Flag that indicates whether if the session is new
        :param is_training: Flag that indicates whether if this session is training a Network.
        :param produce_data: Flag that indicates whether if this session is producing data.
        :param int batch_size: Number of samples in a batch
        """

        self.name: str = self.__class__.__name__

        # Managers variables
        self.manager: Optional[Any] = manager
        self.dataset_manager: Optional[DatasetManager] = None
        self.environment_manager: Optional[EnvironmentManager] = None

        # Data normalization coefficients (default: mean = 0, standard deviation = 1)
        self.normalization: Dict[str, List[float]] = {'input': [0., 1.], 'output': [0., 1.]}
        # # If normalization flag is set to True, try to load existing coefficients
        # if dataset_config is not None and dataset_config.normalize:
        #     json_file_path = None
        #     # Existing Dataset in the current session
        #     if os.path.exists(os.path.join(session, 'dataset')):
        #         json_file_path = os.path.join(session, 'dataset', 'dataset.json')
        #     # Dataset provided by config
        #     elif dataset_config.existing_dir is not None:
        #         dataset_dir = dataset_config.existing_dir
        #         if dataset_dir[-1] != "/":
        #             dataset_dir += "/"
        #         if dataset_dir[-8:] != "dataset/":
        #             dataset_dir += "dataset/"
        #         if os.path.exists(dataset_dir):
        #             json_file_path = os.path.join(dataset_dir, 'dataset.json')
        #     # If Dataset exists then a json file is associated
        #     if json_file_path is not None:
        #         with open(json_file_path) as json_file:
        #             json_dict = json_load(json_file)
        #             # Get the normalization coefficients
        #             for field in self.normalization.keys():
        #                 if field in json_dict['normalization']:
        #                     self.normalization[field] = json_dict['normalization'][field]

        # Create a DatasetManager if required
        create_dataset = pipeline in ['data_generation', 'training'] or produce_data
        database = None
        if create_dataset:
            self.dataset_manager = DatasetManager(dataset_config=dataset_config,
                                                  session=session,
                                                  data_manager=self,
                                                  new_session=new_session,
                                                  pipeline=pipeline,
                                                  produce_data=produce_data)
            database = self.dataset_manager.database

        # Create an EnvironmentManager if required
        create_environment = environment_config is not None
        if create_environment:
            self.environment_manager = EnvironmentManager(environment_config=environment_config,
                                                          data_manager=self,
                                                          session=session,
                                                          data_db=database,
                                                          batch_size=batch_size)

        # DataManager variables
        self.pipeline = pipeline
        self.produce_data = produce_data
        self.batch_size = batch_size
        self.data_lines: Union[List[int], int] = []

    def get_manager(self) -> Any:
        """
        Return the Manager of this DataManager.

        :return: The Manager of this DataManager.
        """

        return self.manager

    def get_database(self) -> Database:
        return self.dataset_manager.database

    def change_database(self) -> None:
        # self.manager.change_database(self.dataset_manager.database)
        self.environment_manager.change_database(self.dataset_manager.database)

    def get_data(self,
                 epoch: int = 0,
                 animate: bool = True) -> None:
        """
        Fetch data from the EnvironmentManager or the DatasetManager according to the context.

        :param epoch: Current epoch number.
        :param animate: Allow EnvironmentManager to generate a new sample.
        :return: Dict containing the newly computed data.
        """

        # Data generation case
        if self.pipeline == 'data_generation':
            self.data_lines = self.environment_manager.get_data(animate=animate)
            self.dataset_manager.add_data()

        # Training case
        elif self.pipeline == 'training':

            # Get data from Environment(s) if used and if the data should be created at this epoch
            if self.environment_manager is not None and (epoch == 0 or self.environment_manager.always_create_data):
                self.data_lines = self.environment_manager.get_data(animate=animate)
                self.dataset_manager.add_data()

            # Get data from Dataset
            else:
                self.data_lines = self.dataset_manager.get_data(batch_size=self.batch_size)
                print(self.data_lines)
                # Dispatch a batch to clients
                if self.environment_manager is not None and self.environment_manager.use_dataset_in_environment:
                    self.data_lines = self.environment_manager.dispatch_batch(batch=self.data_lines)

                # Environment is no longer used
                elif self.environment_manager is not None:
                    self.environment_manager.close()
                    self.environment_manager = None

        # Prediction pipeline
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
