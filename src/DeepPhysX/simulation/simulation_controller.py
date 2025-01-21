from typing import Type, Tuple, Dict, Any, Union, List, Optional
from os.path import isfile, join
from numpy import ndarray

from DeepPhysX.simulation.dpx_simulation import DPXSimulation, AbstractController, Viewer
from DeepPhysX.database.database_handler import DatabaseHandler, Database


class SimulationController(AbstractController):

    def __init__(self,
                 environment_class: Type[DPXSimulation],
                 environment_kwargs: Dict[str, Any],
                 environment_ids: Tuple[int, int] = (1, 1),
                 use_database: bool = True):

        # Manager of the Environment
        self.environment_manager: Any = None
        self.tcp_ip_client: Any = None

        # Create the Environment instance
        self.__environment: Optional[DPXSimulation] = None
        self.__environment_class: Type[DPXSimulation] = environment_class
        self.__environment_kwargs: Dict[str, Any] = environment_kwargs
        self.__environment_id: int = environment_ids[0]
        self.__environment_nb: int = environment_ids[1]

        # Create the Database Handler
        self.__use_database: bool = use_database
        self.__database_handler: DatabaseHandler = DatabaseHandler()

        # Produced data variables
        self.__data: Dict[str, ndarray] = {}
        self.__sample: Dict[str, ndarray] = {}

        # Training data variables
        self.compute_training_data: bool = True

        # Database related variables
        self.__required_fields: List[str] = []
        self.__prediction_fields: List[str] = []
        self.__first_set: bool = True
        self.__first_get: bool = True
        self.update_line: Optional[List[int]] = None

        # Visualization data
        self.viewer: Optional[Viewer] = None

    @property
    def environment(self) -> DPXSimulation:
        return self.__environment

    @property
    def environment_ids(self) -> Tuple[int, int]:
        return self.__environment_id, self.__environment_nb

    @property
    def database_handler(self) -> DatabaseHandler:
        return self.__database_handler

    def create_environment(self) -> None:

        self.__environment: DPXSimulation = self.__environment_class(**self.__environment_kwargs,
                                                                     environment_controller=self)

        self.__environment.create()
        self.__environment.init()

    def connect_to_database(self,
                            database_path: Tuple[str, str],
                            normalize_data: bool):

        self.database_handler.init(database_path=database_path, normalize_data=normalize_data)
        self.__environment.init_database()

    def save_parameters(self, **kwargs) -> None:
        """
        Save a set of parameters in the Database.
        """

        # Create a dedicated Database
        database_dir = self.__database_handler.get_database_dir()
        database_name = 'environment_parameters'
        if isfile(join(database_dir, f'{database_name}.db')):
            database = Database(database_dir=database_dir, database_name=database_name).load()
        else:
            database = Database(database_dir=database_dir, database_name=database_name).new()

        # Create Fields and add data
        fields = [(field, type(value)) for field, value in kwargs.items()]
        database.create_table(table_name=f'Environment_{self.__environment_id}', fields=fields)
        database.add_data(table_name=f'Environment_{self.__environment_id}', data=kwargs)
        database.close()

    def load_parameters(self) -> Dict[str, Any]:
        """
        Load a set of parameters from the Database.
        """

        # Load the dedicated Database and the parameters
        database_dir = self.__database_handler.get_database_dir()
        database_name = 'environment_parameters'
        if isfile(join(database_dir, f'{database_name}.db')):
            database = Database(database_dir=database_dir, database_name=database_name).load()
            return database.get_line(table_name=f'Environment_{self.__environment_id}')
        return {}

    def define_database_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Specify the data fields names and types.

        :param fields: Field or list of fields to tag as training data.
        """

        if self.__environment_id == 1:
            self.__database_handler.create_fields(fields=fields)

    def launch_visualization(self, viewer_key: Optional[int] = None):

        self.viewer = Viewer(sync=False)
        self.__environment.init_visualization()
        self.viewer.launch(batch_key=viewer_key)

    def set_data(self, **kwargs) -> None:
        """
        Set the data to send to the TcpIpServer or the EnvironmentManager.
        """

        # 1. Check kwargs
        default_fields = {'id', 'env_id'}
        if len(self.__required_fields) == 0:
            self.__database_handler.load()
            self.__required_fields = list(set(self.__database_handler.get_fields()) - default_fields)

        # 1.1. Check that default fields are not set by user
        user_fields = set(kwargs.keys())
        if len((default_fields_set_by_user := user_fields - (user_fields - default_fields))) > 0:
            if self.__first_set:
                self.__first_set = False
                print(f"[{self.__environment.name}] WARNING: The fields {default_fields_set_by_user} are already "
                      f"default fields in the Database, please choose another name (default fields: {default_fields}).")
            for field in default_fields_set_by_user:
                kwargs.pop(field)
        user_fields = set(kwargs.keys())

        # 1.2. Check that the fields defined by the user are in the Database
        if len((non_existing_fields := user_fields - set(self.__required_fields))) > 0:
            raise ValueError(f"[{self.__environment.name}] The fields {non_existing_fields} are not in the training "
                             f"Database (required fields: {set(self.__required_fields)}).")

        # 1.3. Check that the required fields are all defined
        if len((non_defined_fields := set(self.__required_fields) - user_fields)) > 0:
            raise ValueError(f"[{self.__environment.name}] The fields {non_defined_fields} are required by the "
                             f"training Database but not set by the Environment "
                             f"(required fields: {set(self.__required_fields)}).")

        # 2. Set the training data
        if self.compute_training_data:
            self.__data = kwargs
            self.__data['env_id'] = self.__environment_id

    def get_training_data(self) -> Dict[str, ndarray]:
        return self.__sample_training

    def get_additional_data(self) -> Dict[str, ndarray]:
        return self.__sample_additional

    def get_prediction(self, **kwargs) -> Dict[str, ndarray]:
        """
        Request a prediction from networks.
        """

        # 1. Check Manager
        first_get = self.__first_get
        if first_get:
            self.__first_get = False
            if (self.environment_manager is not None and not self.environment_manager.allow_prediction_requests) or \
                    (self.tcp_ip_client is not None and not self.tcp_ip_client.allow_prediction_requests):
                raise ValueError(f"[{self.__environment.name}] Prediction request is not available in the Data "
                                 f"Generation pipeline.")

        # 2. Check training data
        default_fields = {'id', 'env_id'}
        if len(self.__prediction_fields) == 0:
            self.__database_handler.load()
            self.__prediction_fields = list(set(self.__database_handler.get_fields(exchange=True)) -
                                            default_fields)

        # 2.1. Check that default fields are not set by user
        user_fields = set(kwargs.keys())
        if len((default_fields_set_by_user := user_fields - (user_fields - default_fields))) > 0:
            if first_get:
                self.__first_get = False
                print(f"[{self.__environment.name}] WARNING: The fields {default_fields_set_by_user} are already "
                      f"default fields in the Database, please choose another name (default fields: {default_fields}).")
            for field in default_fields_set_by_user:
                kwargs.pop(field)
        user_fields = set(kwargs.keys())

        # 2.2. Check that the fields defined by the user are in the Database
        if len((non_existing_fields := user_fields - set(self.__prediction_fields))) > 0:
            raise ValueError(f"[{self.__environment.name}] The fields {non_existing_fields} are not in the training "
                             f"Database (required fields: {set(self.__prediction_fields)}).")

        # 2.3. Check that the required fields are all defined
        # if len((non_defined_fields := set(self.__prediction_fields) - user_fields)) > 0:
        #     raise ValueError(f"[{self.__environment.name}] The fields {non_defined_fields} are required by the "
        #                      f"training Database but not set by the Environment "
        #                      f"(required fields: {set(self.__prediction_fields)}).")

        # 3. Get the prediction from the networks
        # 3.1. Define the training data in the Database
        self.__database_handler.update(exchange=True, data=kwargs, line_id=self.__environment_id)
        # 3.2. Send a prediction request
        if self.tcp_ip_client is not None:
            self.tcp_ip_client.get_prediction()
        elif self.environment_manager is not None:
            self.environment_manager.get_prediction(self.__environment_id)
        else:
            raise ValueError(f"[{self.__environment.name}] The Environment has no Manager to request a prediction.")
        # 3.3. Receive the prediction data
        data_prediction = self.__database_handler.get_data(exchange=True, line_id=self.__environment_id)
        data_prediction.pop('id')
        return data_prediction

    def trigger_prediction(self) -> None:
        """
        Request a prediction from networks and apply it to the Environment.
        """

        # 1. Check training data
        default_fields = {'id', 'env_id'}
        if len(self.__prediction_fields) == 0:
            self.__database_handler.load()
            self.__prediction_fields = list(set(self.__database_handler.get_fields(exchange=True)) -
                                            default_fields)
        data_training = {}
        for field, value in self.__data.items():
            if field not in default_fields and field in self.__prediction_fields:
                data_training[field] = value

        # 2. Apply the prediction of the networks
        data_prediction = self.get_prediction(**data_training)
        self.__environment.apply_prediction(data_prediction)

    def trigger_send_data(self) -> List[int]:
        """
        Add the training data and the additional data in their respective Databases.
        """

        return self.__database_handler.add_data(data=self.__data)

    def trigger_update_data(self, line_id: List[int]) -> None:
        """
        Update the training data and the additional data in their respective Databases.
        """

        if len(self.__data) > 0:
            self.__database_handler.update(data=self.__data, line_id=line_id)

    def trigger_get_data(self, line_id: List[int]) -> None:
        """
        Get the training data and the additional data from their respective Databases.
        """

        self.update_line = line_id

        # Get Database training sample
        self.__sample_training = self.__database_handler.get_line(table_name='Training', line_id=line_id)
        self.set_training_data(**self.__sample_training)

        # Get Database additional sample
        self.__sample_additional = self.__database_handler.get_line(table_name='Additional', line_id=line_id)
        self.set_additional_data(**self.__sample_additional)
        self.__sample_additional = None if len(self.__sample_training) == 1 else self.__sample_training

    def reset_data(self) -> None:
        """
        Reset the training data and the additional data variables.
        """

        self.__data = {}
        self.__sample = None
        self.update_line = None

    def close(self):

        if self.viewer is not None:
            self.viewer.shutdown()
        self.__environment.close()
