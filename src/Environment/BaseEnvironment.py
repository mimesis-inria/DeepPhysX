from typing import Any, Optional, Dict, Union, Tuple, List, Type
from numpy import ndarray
from os.path import isfile, join

from SSD.Core.Storage.Database import Database

from DeepPhysX.Core.Visualization.VedoFactory import VedoFactory
from DeepPhysX.Core.AsyncSocket.AbstractEnvironment import AbstractEnvironment
from DeepPhysX.Core.Database.DatabaseHandler import DatabaseHandler


class BaseEnvironment(AbstractEnvironment):

    def __init__(self,
                 as_tcp_ip_client: bool = True,
                 instance_id: int = 1,
                 instance_nb: int = 1,
                 visualization_db: Optional[Union[Database, Tuple[str, str]]] = None,
                 **kwargs):
        """
        BaseEnvironment computes simulated data for the Network and its training process.

        :param as_tcp_ip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False.
        :param instance_id: ID of the instance.
        :param instance_nb: Number of simultaneously launched instances.
        :param visualization_db: The path to the visualization Database or the visualization Database object to connect.
        """

        AbstractEnvironment.__init__(self,
                                     as_tcp_ip_client=as_tcp_ip_client,
                                     instance_id=instance_id,
                                     instance_nb=instance_nb)

        # Training data variables
        self.__data_training: Dict[str, ndarray] = {}
        self.__data_additional: Dict[str, ndarray] = {}
        self.compute_training_data: bool = True

        # Dataset data variables
        self.update_line: Optional[int] = None
        self.sample_training: Optional[Dict[str, Any]] = None
        self.sample_additional: Optional[Dict[str, Any]] = None
        self.__first_add: List[bool] = [True, True]

        # Connect the Environment to the data Database
        self.__database_handler = DatabaseHandler(on_init_handler=self.__database_handler_init)

        # Connect the Factory to the visualization Database
        self.factory: Optional[VedoFactory] = None
        if visualization_db is not None:
            if type(visualization_db) == list:
                self.factory = VedoFactory(database_path=visualization_db,
                                           idx_instance=instance_id,
                                           remote=True)
            else:
                self.factory = VedoFactory(database=visualization_db,
                                           idx_instance=instance_id)

    ##########################################################################################
    ##########################################################################################
    #                                 Environment initialization                             #
    ##########################################################################################
    ##########################################################################################

    def create(self) -> None:
        """
        Create the Environment. Automatically called when Environment is launched.
        Must be implemented by user.
        """

        raise NotImplementedError

    def init(self) -> None:
        """
        Initialize the Environment. Automatically called when Environment is launched.
        Not mandatory.
        """

        pass

    def init_database(self) -> None:
        """
        Define the fields of the training dataset. Automatically called when Environment is launched.
        Must be implemented by user.
        """

        raise NotImplementedError

    def init_visualization(self) -> None:
        """
        Define the visualization objects to send to the Visualizer. Automatically called when Environment is launched.
        Not mandatory.
        """

        pass

    def save_parameters(self, **kwargs) -> None:
        """
        Save a set of parameters in the Database.
        """

        # Create a dedicated Database
        database_dir = self.__database_handler.get_database_dir()
        if isfile(join(database_dir, 'environment_parameters.db')):
            database = Database(database_dir=database_dir,
                                database_name='environment_parameters').load()
        else:
            database = Database(database_dir=database_dir,
                                database_name='environment_parameters').new()

        # Create fields and add data
        fields = [(field, type(value)) for field, value in kwargs.items()]
        database.create_table(table_name=f'Environment_{self.instance_id}', fields=fields)
        database.add_data(table_name=f'Environment_{self.instance_id}', data=kwargs)
        database.close()

    def load_parameters(self) -> Dict[str, Any]:
        """
        Load a set of parameters from the Database.
        """

        # Load the dedicated Database and the parameters
        database_dir = self.__database_handler.get_database_dir()
        if isfile(join(database_dir, 'environment_parameters.db')):
            database = Database(database_dir=database_dir,
                                database_name='environment_parameters').load()
            parameters = database.get_line(table_name=f'Environment_{self.instance_id}')
            del parameters['id']
            return parameters
        return {}

    ##########################################################################################
    ##########################################################################################
    #                                 Environment behavior                                   #
    ##########################################################################################
    ##########################################################################################

    async def step(self) -> None:
        """
        Compute the number of steps in the Environment specified by simulations_per_step in EnvironmentConfig.
        Must be implemented by user.
        """

        raise NotImplementedError

    def check_sample(self) -> bool:
        """
        Check if the current produced sample is usable for training.
        Not mandatory.

        :return: Current data can be used or not
        """

        return True

    def apply_prediction(self,
                         prediction: Dict[str, ndarray]) -> None:
        """
        Apply network prediction in environment.
        Not mandatory.

        :param prediction: Prediction data.
        """

        pass

    def close(self) -> None:
        """
        Close the Environment. Automatically called when Environment is shut down.
        Not mandatory.
        """

        pass

    ##########################################################################################
    ##########################################################################################
    #                                 Defining data samples                                  #
    ##########################################################################################
    ##########################################################################################

    def define_training_fields(self,
                               fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Specify the training data fields names and types.

        :param fields: Field or list of fields to tag as training data.
        """

        self.__database_handler.create_fields(table_name='Training',
                                              fields=fields)

    def define_additional_fields(self,
                                 fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Specify the additional data fields names and types.

        :param fields: Field or list of Fields to tag as additional data.
        """

        self.__database_handler.create_fields(table_name='Additional',
                                              fields=fields)

    def set_training_data(self, **kwargs) -> None:
        """
        Set the training data to send to the TcpIpServer or the EnvironmentManager.
        """

        # Check kwargs
        if self.__first_add[0]:
            self.__database_handler.load()
            self.__first_add[0] = False
            required_fields = list(set(self.__database_handler.get_fields(table_name='Training')) - {'id', 'env_id'})
            if len(required_fields) > 0:
                for field in kwargs.keys():
                    if field not in required_fields:
                        raise ValueError(f"[{self.name}] The field '{field}' is not in the training Database."
                                         f"Required fields are {required_fields}.")
                for field in required_fields:
                    if field not in kwargs.keys():
                        raise ValueError(f"[{self.name}] The field '{field}' was not defined in training data."
                                         f"Required fields are {required_fields}.")

        # Training data is set if the Environment can compute data
        if self.compute_training_data:
            self.__data_training = kwargs
            self.__data_training['env_id'] = self.instance_id

    def set_additional_data(self,
                            **kwargs) -> None:
        """
        Set the additional data to send to the TcpIpServer or the EnvironmentManager.
        """

        # Additional data is also set if the Environment can compute data
        if self.compute_training_data:
            self.__data_additional = kwargs
            self.__data_additional['env_id'] = self.instance_id

    ##########################################################################################
    ##########################################################################################
    #                                   Available requests                                   #
    ##########################################################################################
    ##########################################################################################

    def get_prediction(self, **kwargs) -> Dict[str, ndarray]:
        """
        Request a prediction from Network.

        :return: Network prediction.
        """

        # Check kwargs
        if self.__first_add[1]:

            if (self.environment_manager is not None and not self.environment_manager.allow_prediction_requests) or \
                    (self.tcp_ip_client is not None and not self.tcp_ip_client.allow_prediction_requests):
                raise ValueError(f"[{self.name}] Prediction request is not available in data generation Pipeline.")

            if len(kwargs) == 0 and len(self.__data_training) == 0:
                raise ValueError(f"[{self.name}] The prediction request requires the network fields.")
            self.__database_handler.load()
            self.__first_add[1] = False
            required_fields = list(set(self.__database_handler.get_fields(table_name='Exchange')) - {'id'})
            for field in kwargs.keys():
                if field not in required_fields:
                    raise ValueError(f"[{self.name}] The field '{field}' is not in the training Database."
                                     f"Required fields are {required_fields}.")

        # Avoid empty sample
        if len(kwargs) == 0:
            required_fields = set(self.__database_handler.get_fields(table_name='Exchange')) - {'id'}
            necessary_fields = list(required_fields.intersection(self.__data_training.keys()))
            kwargs = {field: self.__data_training[field] for field in necessary_fields}

        # If Environment is a TcpIpClient, send request to the Server
        if self.tcp_ip_client is not None:
            return self.tcp_ip_client.get_prediction(**kwargs)

        # Otherwise, check the hierarchy of managers
        elif self.environment_manager is not None:
            if self.environment_manager.data_manager is None:
                raise ValueError("Cannot request prediction if DataManager does not exist")
            # Get a prediction
            self.__database_handler.update(table_name='Exchange',
                                           data=kwargs,
                                           line_id=self.instance_id)
            self.environment_manager.data_manager.get_prediction(self.instance_id)
            data_pred = self.__database_handler.get_line(table_name='Exchange',
                                                         line_id=self.instance_id)
            del data_pred['id']
            return data_pred

        else:
            raise ValueError(f"[{self.name}] This Environment has not Manager.")

    def update_visualisation(self) -> None:
        """
        Triggers the Visualizer update.
        """

        # If Environment is a TcpIpClient, request to the Server
        if self.as_tcp_ip_client:
            self.tcp_ip_client.request_update_visualization()
        self.factory.render()

    def _get_prediction(self):
        """
        Request a prediction from Network and apply it to the Environment.
        Should not be used by users.
        """

        training_data = self.__data_training.copy()
        required_fields = self.__database_handler.get_fields(table_name='Exchange')
        for field in self.__data_training.keys():
            if field not in required_fields:
                del training_data[field]
        self.apply_prediction(self.get_prediction(**training_data))

    ##########################################################################################
    ##########################################################################################
    #                                 Database communication                                 #
    ##########################################################################################
    ##########################################################################################

    def get_database_handler(self) -> DatabaseHandler:
        """
        Get the DatabaseHandler of the Environment.
        """

        return self.__database_handler

    def __database_handler_init(self):
        """
        Init event of the DatabaseHandler.
        """

        # Load Database and create basic fields
        self.__database_handler.load()

    def _send_training_data(self) -> List[int]:
        """
        Add the training data and the additional data in their respective Databases.
        Should not be used by users.

        :return: Index of the samples in the Database.
        """

        line_id = self.__database_handler.add_data(table_name='Training',
                                                   data=self.__data_training)
        self.__database_handler.add_data(table_name='Additional',
                                         data=self.__data_additional)
        return line_id

    def _update_training_data(self,
                              line_id: List[int]) -> None:
        """
        Update the training data and the additional data in their respective Databases.
        Should not be used by users.

        :param line_id: Index of the samples to update.
        """

        if self.__data_training != {}:
            self.__database_handler.update(table_name='Training',
                                           data=self.__data_training,
                                           line_id=line_id)
        if self.__data_additional != {}:
            self.__database_handler.update(table_name='Additional',
                                           data=self.__data_additional,
                                           line_id=line_id)

    def _get_training_data(self,
                           line_id: List[int]) -> None:
        """
        Get the training data and the additional data from their respective Databases.
        Should not be used by users.

        :param line_id: Index of the sample to get.
        """

        self.update_line = line_id
        self.sample_training = self.__database_handler.get_line(table_name='Training',
                                                                line_id=line_id)
        self.sample_additional = self.__database_handler.get_line(table_name='Additional',
                                                                  line_id=line_id)
        self.sample_additional = None if len(self.sample_additional) == 1 else self.sample_additional

    def _reset_training_data(self) -> None:
        """
        Reset the training data and the additional data variables.
        Should not be used by users.
        """

        self.__data_training = {}
        self.__data_additional = {}
        self.sample_training = None
        self.sample_additional = None
        self.update_line = None

    def __str__(self):

        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Name: {self.name} n°{self.instance_id}\n"
        description += f"    Comments:\n"
        description += f"    Input size:\n"
        description += f"    Output size:\n"
        return description
