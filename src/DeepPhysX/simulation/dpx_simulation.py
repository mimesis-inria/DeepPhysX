from typing import Type, Tuple, Dict, Any, Union, List, Optional
from os.path import isfile, join
from numpy import ndarray

from SimRender.core import Viewer
from DeepPhysX.database.database_handler import DatabaseHandler, Database


class DPXSimulation:

    def __init__(self, **kwargs):
        """
        BaseSimulation computes simulated data for the networks and its training process.
        """

        self.__controller: SimulationController = kwargs.pop('simulation_controller')

    @property
    def simulation_id(self) -> int:
        return self.__controller.simulation_ids[0]

    @property
    def simulation_nb(self) -> int:
        return self.__controller.simulation_ids[1]

    @property
    def viewer(self) -> Optional[Viewer]:
        return self.__controller.viewer

    @property
    def compute_training_data(self) -> bool:
        return self.__controller.compute_training_data

    ###################
    # Simulation init #
    ###################

    def create(self) -> None:
        """
        Create the Simulation. Automatically called when Simulation is launched.
        Must be implemented by user.
        """

        raise NotImplementedError

    def init(self) -> None:
        """
        Initialize the Simulation. Automatically called when Simulation is launched.
        Not mandatory.
        """

        pass

    def init_database(self) -> None:
        """
        Define the fields of the training dataset. Automatically called when Simulation is launched.
        Must be implemented by user.
        """

        raise NotImplementedError

    def init_visualization(self) -> None:
        """
        Define the visualization objects to send to the Visualizer. Automatically called when Simulation is launched.
        Not mandatory.
        """

        pass

    def save_parameters(self, **kwargs) -> None:
        """
        Save a set of parameters in the Database.
        """

        self.__controller.save_parameters(**kwargs)

    def load_parameters(self) -> Dict[str, Any]:
        """
        Load a set of parameters from the Database.
        """

        return self.__controller.load_parameters()

    ########################
    # Simulation behavior #
    ########################

    def step(self) -> None:
        """
        Compute the number of steps in the Simulation specified by simulations_per_step in SimulationConfig.
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
        Apply networks prediction in Simulation.
        Not mandatory.

        :param prediction: Prediction data.
        """

        pass

    def close(self) -> None:
        """
        Close the Simulation. Automatically called when Simulation is shut down.
        Not mandatory.
        """

        pass

    ################
    # Data samples #
    ################

    def add_data_field(self, field_name: str, field_type: Type):
        """
        Add a new data field in the Database.

        :param field_name: Name of the data field.
        :param field_type: Type of the data field.
        """

        self.__controller.define_database_fields(fields=(field_name, field_type))

    def set_data(self, **kwargs) -> None:
        """
        Set the training data to send to the TcpIpServer or the SimulationManager.
        """

        self.__controller.set_data(**kwargs)

    @property
    def data(self) -> Dict[str, ndarray]:
        return self.__controller.get_data()

    ############
    # Requests #
    ############

    def get_prediction(self, **kwargs) -> Dict[str, ndarray]:
        """
        Request a prediction from networks.

        :return: networks prediction.
        """

        return self.__controller.get_prediction(**kwargs)

    def update_visualisation(self) -> None:
        """
        Triggers the Visualizer update.
        """

        if self.viewer is not None:
            self.viewer.render()

    def __str__(self):

        description = "\n"
        description += f"  Simulation\n"
        # description += f"    Name: {self.name} n°{self.simulation_id}\n"
        # description += f"    Comments:\n"
        # description += f"    Input size:\n"
        # description += f"    Output size:\n"
        return description


class SimulationController:

    def __init__(self,
                 simulation_class: Type[DPXSimulation],
                 simulation_kwargs: Dict[str, Any],
                 manager: Any,
                 simulation_id: int = 1,
                 simulation_nb: int = 1):
        """
        """

        # Simulation variables
        self.__simulation: Optional[DPXSimulation] = None
        self.__simulation_class: Type[DPXSimulation] = simulation_class
        self.__simulation_kwargs: Dict[str, Any] = simulation_kwargs
        self.__simulation_id: int = simulation_id
        self.__simulation_nb: int = simulation_nb

        # Manager variables
        self.__manager = manager

        # Data access variables
        self.__database_handler: DatabaseHandler = DatabaseHandler()
        self.__first_set: bool = True
        self.__first_get: bool = True
        self.__data: Dict[str, ndarray] = {}
        self.__required_fields: List[str] = []
        self.__prediction_fields: List[str] = []
        self.compute_training_data: bool = True
        self.update_line: Optional[List[int]] = None

        # Viewer variables
        self.viewer: Optional[Viewer] = None

    @property
    def simulation(self) -> DPXSimulation:
        return self.__simulation

    @property
    def simulation_ids(self) -> Tuple[int, int]:
        return self.__simulation_id, self.__simulation_nb

    @property
    def database_handler(self) -> DatabaseHandler:
        return self.__database_handler

    def create_simulation(self) -> None:

        self.__simulation: DPXSimulation = self.__simulation_class(**self.__simulation_kwargs,
                                                                   simulation_controller=self)

        self.__simulation.create()
        self.__simulation.init()

    def connect_to_database(self,
                            database_path: Tuple[str, str],
                            normalize_data: bool):

        self.database_handler.init(database_path=database_path, normalize_data=normalize_data)
        self.__simulation.init_database()

    def save_parameters(self, **kwargs) -> None:
        """
        Save a set of parameters in the Database.
        """

        # Create a dedicated Database
        database_dir = self.__database_handler.get_database_dir()
        database_name = 'simulation_parameters'
        if isfile(join(database_dir, f'{database_name}.db')):
            database = Database(database_dir=database_dir, database_name=database_name).load()
        else:
            database = Database(database_dir=database_dir, database_name=database_name).new()

        # Create Fields and add data
        fields = [(field, type(value)) for field, value in kwargs.items()]
        database.create_table(table_name=f'Simulation_{self.__simulation_id}', fields=fields)
        database.add_data(table_name=f'Simulation_{self.__simulation_id}', data=kwargs)
        database.close()

    def load_parameters(self) -> Dict[str, Any]:
        """
        Load a set of parameters from the Database.
        """

        # Load the dedicated Database and the parameters
        database_dir = self.__database_handler.get_database_dir()
        database_name = 'simulation_parameters'
        if isfile(join(database_dir, f'{database_name}.db')):
            database = Database(database_dir=database_dir, database_name=database_name).load()
            return database.get_line(table_name=f'Simulation_{self.__simulation_id}')
        return {}

    def define_database_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Specify the data fields names and types.

        :param fields: Field or list of fields to tag as training data.
        """

        fields = [fields] if not isinstance(fields, list) else fields
        self.__database_handler.create_fields(fields=fields)
        self.__required_fields += [f[0] for f in fields]

    def launch_visualization(self, viewer_key: Optional[int] = None):

        self.viewer = Viewer(sync=False)
        self.__simulation.init_visualization()
        self.viewer.launch(batch_key=viewer_key)

    def set_data(self, **kwargs) -> None:
        """
        Set the data to send to the TcpIpServer or the SimulationManager.
        """

        # First add: check data fields
        if self.__first_set:

            self.__first_set = False
            default_fields = {'id', 'env_id'}
            user_fields = set(kwargs.keys())

            # Check that default fields are not set by user
            if len((default_fields_set_by_user := user_fields - (user_fields - default_fields))) > 0:
                raise ValueError(f"[Simulation] The fields {default_fields_set_by_user} are default fields "
                                 f"in the Database, please choose another name (default fields: {default_fields}).")

            # Check that the fields defined by the user are in the Database
            if len((non_existing_fields := user_fields - set(self.__required_fields))) > 0:
                raise ValueError(f"[Simulation] The fields {non_existing_fields} are not in the Database "
                                 f"(required fields: {set(self.__required_fields)}).")

            # Check that the required fields are all defined
            if len((non_defined_fields := set(self.__required_fields) - user_fields)) > 0:
                raise ValueError(f"[Simulation] The fields {non_defined_fields} are required in the Database "
                                 f"(required fields: {set(self.__required_fields)}).")

        # Set the training data
        if self.compute_training_data:
            self.__data = kwargs
            self.__data['env_id'] = self.__simulation_id

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
            if not self.__manager.allow_prediction_requests:
                raise ValueError(f"[{self.__simulation.name}] Prediction request is not available in the Data "
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
                print(f"[{self.__simulation.name}] WARNING: The fields {default_fields_set_by_user} are already "
                      f"default fields in the Database, please choose another name (default fields: {default_fields}).")
            for field in default_fields_set_by_user:
                kwargs.pop(field)
        user_fields = set(kwargs.keys())

        # 2.2. Check that the fields defined by the user are in the Database
        if len((non_existing_fields := user_fields - set(self.__prediction_fields))) > 0:
            raise ValueError(f"[{self.__simulation.name}] The fields {non_existing_fields} are not in the training "
                             f"Database (required fields: {set(self.__prediction_fields)}).")

        # 2.3. Check that the required fields are all defined
        # if len((non_defined_fields := set(self.__prediction_fields) - user_fields)) > 0:
        #     raise ValueError(f"[{self.__simulation.name}] The fields {non_defined_fields} are required by the "
        #                      f"training Database but not set by the Simulation "
        #                      f"(required fields: {set(self.__prediction_fields)}).")

        # 3. Get the prediction from the networks
        # 3.1. Define the training data in the Database
        self.__database_handler.update(exchange=True, data=kwargs, line_id=self.__simulation_id)
        # 3.2. Send a prediction request
        self.__manager.get_prediction(self.__simulation_id)
        # 3.3. Receive the prediction data
        data_prediction = self.__database_handler.get_data(exchange=True, line_id=self.__simulation_id)
        data_prediction.pop('id')
        return data_prediction

    def trigger_prediction(self) -> None:
        """
        Request a prediction from networks and apply it to the Simulation.
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
        self.__simulation.apply_prediction(data_prediction)

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
        self.__sample_training = self.__database_handler.get_data(table_name='Training', line_id=line_id)
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
        self.__simulation.close()
