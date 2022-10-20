from typing import Any, Optional, Dict, Union, Tuple, List, Type
from numpy import ndarray
from os.path import isfile, join

from SSD.Core.Storage.Database import Database

from DeepPhysX.Core.Visualization.VedoFactory import VedoFactory
from DeepPhysX.Core.AsyncSocket.TcpIpClient import TcpIpClient


class BaseEnvironment(TcpIpClient):

    def __init__(self,
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 instance_id: int = 0,
                 number_of_instances: int = 1,
                 as_tcp_ip_client: bool = True,
                 environment_manager: Optional[Any] = None,
                 training_db: Optional[Union[Database, Tuple[str, str]]] = None,
                 visualization_db: Optional[Union[Database, Tuple[str, str]]] = None):
        """
        BaseEnvironment is an environment class to compute simulated data for the network and its optimization process.

        :param ip_address: IP address of the TcpIpObject.
        :param port: Port number of the TcpIpObject.
        :param instance_id: ID of the instance.
        :param number_of_instances: Number of simultaneously launched instances.
        :param as_tcp_ip_client: Environment is owned by a TcpIpClient if True, by an EnvironmentManager if False.
        :param environment_manager: EnvironmentManager that handles the Environment if 'as_tcp_ip_client' is False.
        :param visualization_db: The path to the visualization Database or the visualization Database object to connect to.
        """

        TcpIpClient.__init__(self,
                             instance_id=instance_id,
                             number_of_instances=number_of_instances,
                             as_tcp_ip_client=as_tcp_ip_client,
                             ip_address=ip_address,
                             port=port)

        # Training data variables
        self.__training_data: Dict[str, ndarray] = {}
        self.__additional_data: Dict[str, ndarray] = {}
        self.compute_training_data: bool = True

        # Dataset data variables
        self.database: Optional[Database] = None
        self.update_line: Optional[int] = None
        self.sample_training: Optional[Dict[str, Any]] = None
        self.sample_additional: Optional[Dict[str, Any]] = None
        self.__first_add: List[bool] = [True, True]

        # Manager if the Environment is not a TcpIpClient
        self.environment_manager: Any = environment_manager

        # Connect the Environment to the data Database
        self.database: Optional[Database] = None
        if training_db is not None:
            if type(training_db) == list:
                self.database = Database(database_dir=training_db[0],
                                         database_name=training_db[1]).load()
            else:
                self.database = training_db
            if self.instance_id == 1:
                self.database.create_fields(table_name='Training',
                                            fields=('env_id', int))
                self.database.create_fields(table_name='Additional',
                                            fields=('env_id', int))

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
    #                                 Initializing Environment                               #
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

    def apply_prediction(self, prediction: Dict[str, ndarray]) -> None:
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
    #                                   Defining a sample                                    #
    ##########################################################################################
    ##########################################################################################

    def save_parameters(self, **kwargs) -> None:
        """
        Save a set of parameters in the Database.
        """

        database_dir = self.database.get_path()[0]
        if isfile(join(database_dir, 'environment_parameters.db')):
            database = Database(database_dir=database_dir,
                                database_name='environment_parameters').load()
        else:
            database = Database(database_dir=database_dir,
                                database_name='environment_parameters').new()
        fields = [(field, type(value)) for field, value in kwargs.items()]
        database.create_table(table_name=f'Environment_{self.instance_id}', fields=fields)
        database.add_data(table_name=f'Environment_{self.instance_id}', data=kwargs)
        database.close()

    def load_parameters(self) -> Dict[str, Any]:
        """
        Load a set of parameters from the Database.
        """

        database_dir = self.database.get_path()[0]
        if isfile(join(database_dir, 'environment_parameters.db')):
            database = Database(database_dir=database_dir,
                                database_name='environment_parameters').load()
            parameters = database.get_line(table_name=f'Environment_{self.instance_id}')
            del parameters['id']
            return parameters
        return {}

    def define_training_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        self.database.load()
        if len(self.database.get_fields(table_name='Training')) == 2:
            self.database.create_fields(table_name='Training', fields=fields)

    def define_additional_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        self.database.load()
        if len(self.database.get_fields(table_name='Additional')) == 2:
            self.database.create_fields(table_name='Additional', fields=fields)

    def set_training_data(self, **kwargs) -> None:
        """
        Set the training data to send to the TcpIpServer or the EnvironmentManager.
        """

        # Check kwargs
        if self.__first_add[0]:
            if self.instance_id != 0:
                self.database.load()
            self.__first_add[0] = False
            required_fields = list(set(self.database.get_fields(table_name='Training')) - {'id', 'env_id'})
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
            self.__training_data = kwargs
            self.__training_data['env_id'] = self.instance_id

    def set_additional_data(self,
                            **kwargs) -> None:
        # Additional data is also set if the Environment can compute data
        if self.compute_training_data:
            self.__additional_data = kwargs
            self.__additional_data['env_id'] = self.instance_id

    def _send_training_data(self) -> int:
        line_id = self.database.add_data(table_name='Training',
                                         data=self.__training_data)
        self.database.add_data(table_name='Additional',
                               data=self.__additional_data)
        return line_id

    def _reset_training_data(self) -> None:
        self.__training_data = {}
        self.__additional_data = {}
        self.sample_training = None
        self.sample_additional = None
        self.update_line = None

    def _update_training_data(self,
                              line_id: int) -> None:
        if self.__training_data != {}:
            self.database.update(table_name='Training',
                                 data=self.__training_data,
                                 line_id=line_id)
        if self.__additional_data != {}:
            self.database.update(table_name='Additional',
                                 data=self.__additional_data,
                                 line_id=line_id)

    def _get_training_data(self,
                           line: int) -> None:
        self.update_line = line
        self.sample_training = self.database.get_line(table_name='Training',
                                                      line_id=line)
        self.sample_additional = self.database.get_line(table_name='Additional',
                                                        line_id=line)
        self.sample_additional = None if len(self.sample_additional) == 1 else self.sample_additional

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
            if self.instance_id != 0:
                self.database.load()
            self.__first_add[1] = False
            required_fields = list(set(self.database.get_fields(table_name='Prediction')) - {'id'})
            for field in kwargs.keys():
                if field not in required_fields:
                    raise ValueError(f"[{self.name}] The field '{field}' is not in the training Database."
                                     f"Required fields are {required_fields}.")

        # If Environment is a TcpIpClient, send request to the Server
        if self.as_tcp_ip_client:
            return TcpIpClient.get_prediction(self, **kwargs)

        # Otherwise, check the hierarchy of managers
        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        # Get a prediction
        self.database.update(table_name='Prediction',
                             data=kwargs,
                             line_id=self.instance_id)
        self.environment_manager.data_manager.get_prediction(self.instance_id)
        data_pred = self.database.get_line(table_name='Prediction',
                                           line_id=self.instance_id)
        del data_pred['id']
        return data_pred

    def update_visualisation(self) -> None:
        """
        Triggers the Visualizer update.
        """

        # If Environment is a TcpIpClient, request to the Server
        if self.as_tcp_ip_client:
            self.request_update_visualization()
        self.factory.render()

    def __str__(self) -> str:

        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Name: {self.name} n°{self.instance_id}\n"
        description += f"    Comments:\n"
        description += f"    Input size:\n"
        description += f"    Output size:\n"
        return description
