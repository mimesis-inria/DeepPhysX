from typing import Type, Tuple, Dict, Any, Union, List, Optional
from numpy import ndarray

from SimRender.core import Viewer
from DeepPhysX.database.database_controller import DatabaseController

try:
    import Sofa
    import Sofa.Simulation
    from SimRender.sofa import Viewer as SofaViewer
except ImportError:
    pass


class Simulation:

    def __init__(self, **kwargs):
        """
        Simulation computes simulated data for the networks and its training process.
        """

        self.__controller: Optional[SimulationController] = kwargs.pop('simulation_controller', None)
        self.viewer: Optional[Viewer] = None

    @property
    def simulation_id(self) -> int:

        return self.__controller.simulation_ids[0]

    @property
    def simulation_nb(self) -> int:

        return self.__controller.simulation_ids[1]

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

    #######################
    # Simulation behavior #
    #######################

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

    def add_data_field(self, field_name: str, field_type: Type) -> None:
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
        """

        return self.__controller.get_prediction(**kwargs)


class SofaSimulation(Sofa.Core.Controller, Simulation):

    def __init__(self, **kwargs):
        """
        SofaSimulation computes simulated data for the networks and its training process using the Sofa framework.
        """

        Sofa.Core.Controller.__init__(self, **kwargs)
        # Warning: Define root node before init Environment
        self.root = Sofa.Core.Node('root')
        self.root.addObject(self)
        Simulation.__init__(self, **kwargs)
        self.viewer: Optional[SofaViewer] = None

    def step(self) -> None:
        """
        Trigger a Sofa time step.
        """

        Sofa.Simulation.animate(self.root, self.root.dt.value)


class SimulationController:

    def __init__(self,
                 simulation_class: Type[Simulation],
                 simulation_kwargs: Dict[str, Any],
                 manager: Any,
                 simulation_id: int = 1,
                 simulation_nb: int = 1):
        """
        SimulationController allows components to interact with the simulation.
        
        :param simulation_class: The numerical simulation class.
        :param simulation_kwargs: Dict of kwargs to create an instance of the numerical simulation.
        :param manager: The SimulationManager that handles this controller.
        :param simulation_id: Index of the controlled simulation.
        :param simulation_nb: Nuber of parallel simulations.
        """

        # Simulation variables
        self.__simulation: Optional[Simulation] = None
        self.__simulation_class: Type[Simulation] = simulation_class
        self.__simulation_kwargs: Dict[str, Any] = simulation_kwargs
        self.__simulation_id: int = simulation_id
        self.__simulation_nb: int = simulation_nb

        # Manager variables
        self.__manager = manager

        # Data access variables
        self.__database: DatabaseController = DatabaseController()
        self.__first_set: bool = True
        self.__first_get: bool = True
        self.__data: Dict[str, ndarray] = {}
        self.__required_fields: List[str] = []
        self.__prediction_fields: List[str] = []
        self.compute_training_data: bool = True

    @property
    def simulation(self) -> Simulation:

        return self.__simulation

    @property
    def simulation_ids(self) -> Tuple[int, int]:

        return self.__simulation_id, self.__simulation_nb

    def create_simulation(self, use_viewer: bool, viewer_key: Optional[int] = None) -> None:
        """
        Create the simulation instance, initialize database fields and visualization.
        """

        # Create the simulation instance
        self.__simulation: Simulation = self.__simulation_class(**self.__simulation_kwargs, simulation_controller=self)
        self.__simulation.create()

        # In case of Sofa simulation, init the root node
        if isinstance(self.__simulation, SofaSimulation):
            Sofa.Simulation.initRoot(self.__simulation.root)

        # In case of visualization, launch the viewer
        if use_viewer:

            # Create the viewer instance
            if isinstance(self.__simulation, SofaSimulation):
                viewer = SofaViewer(root_node=self.__simulation.root, sync=False)
            else:
                viewer = Viewer(sync=False)

            # Init the visualization in the simulation
            self.__simulation.viewer = viewer
            self.__simulation.init_visualization()

            # Launch the viewer (use batch_key in case of multi-simulations)
            viewer.launch(batch_key=viewer_key)

    def connect_to_database(self,
                            database_path: Tuple[str, str],
                            normalize_data: bool) -> None:
        """
        Connect to the database controller.

        :param database_path: Path to the database repository.
        :param normalize_data: If True, data should be normalized.
        """

        # Initialize the database instance
        self.__database.init(database_path=database_path, normalize_data=normalize_data)

        # Create user data fields
        self.__simulation.init_database()

    def define_database_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Specify the data fields names and types.

        :param fields: Field or list of fields to tag as training data.
        """

        fields = [fields] if not isinstance(fields, list) else fields
        self.__database.create_fields(fields=fields)
        self.__required_fields += [f[0] for f in fields]

    def set_data(self, **kwargs) -> None:
        """
        Set the data to send to the database.
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

    def get_data(self) -> Dict[str, Any]:
        """
        Return the current data samples.
        """

        return self.__data

    def get_prediction(self, **kwargs) -> Dict[str, ndarray]:
        """
        Request a prediction from the network.
        """

        # 1. Check Manager
        first_get = self.__first_get
        if first_get:
            self.__first_get = False
            if not self.__manager.allow_prediction_requests:
                raise ValueError("[Simulation] Prediction request is not available in the Data "
                                 "Generation pipeline.")

        # 2. Check training data
        default_fields = {'id', 'env_id'}
        if len(self.__prediction_fields) == 0:
            self.__database.load()
            self.__prediction_fields = list(set(self.__database.get_fields(exchange=True)) -
                                            default_fields)

        # 2.1. Check that default fields are not set by user
        user_fields = set(kwargs.keys())
        if len((default_fields_set_by_user := user_fields - (user_fields - default_fields))) > 0:
            if first_get:
                self.__first_get = False
                print(f"[Simulation] WARNING: The fields {default_fields_set_by_user} are already "
                      f"default fields in the Database, please choose another name (default fields: {default_fields}).")
            for field in default_fields_set_by_user:
                kwargs.pop(field)
        user_fields = set(kwargs.keys())

        # 2.2. Check that the fields defined by the user are in the Database
        if len((non_existing_fields := user_fields - set(self.__prediction_fields))) > 0:
            raise ValueError(f"[Simulation] The fields {non_existing_fields} are not in the training "
                             f"Database (required fields: {set(self.__prediction_fields)}).")

        # 3. Get the prediction from the networks
        # 3.1. Define the training data in the Database
        self.__database.update(exchange=True, data=kwargs, line_id=self.__simulation_id)
        # 3.2. Send a prediction request
        self.__manager.get_prediction(self.__simulation_id)
        # 3.3. Receive the prediction data
        data_prediction = self.__database.get_data(exchange=True, line_id=self.__simulation_id)
        data_prediction.pop('id')
        return {key: value[0] for key, value in data_prediction.items()}

    def trigger_prediction(self) -> None:
        """
        Request a prediction from networks and apply it to the Simulation.
        """

        # 1. Check training data
        default_fields = {'id', 'env_id'}
        if len(self.__prediction_fields) == 0:
            self.__database.load()
            self.__prediction_fields = list(set(self.__database.get_fields(exchange=True)) -
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

        return self.__database.add_data(data=self.__data)

    def trigger_update_data(self, line_id: List[int]) -> None:
        """
        Update the training data and the additional data in their respective Databases.
        """

        if len(self.__data) > 0:
            self.__database.update(data=self.__data, line_id=line_id)

    def trigger_get_data(self, line_id: List[int]) -> None:
        """
        Get the training data and the additional data from their respective Databases.
        """

        self.__data = self.__database.get_data(line_id=line_id)
        self.__data['env_id'] = self.__simulation_id

    def reset_data(self) -> None:
        """
        Reset the training data and the additional data variables.
        """

        self.__data = {}

    def close(self) -> None:
        """
        Close the simulation and the viewer.
        """

        if self.__simulation.viewer is not None:
            self.__simulation.viewer.shutdown()
        self.__simulation.close()
