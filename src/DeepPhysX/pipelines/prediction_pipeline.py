from typing import Optional
from os.path import join, exists

from DeepPhysX.simulation.simulation_manager import SimulationManager, SimulationConfig
from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.networks.network_manager import NetworkManager
from DeepPhysX.utils.path import get_session_dir


class PredictionPipeline:

    def __init__(self,
                 network_manager: NetworkManager,
                 simulation_config: SimulationConfig,
                 database_manager: Optional[DatabaseManager] = None,
                 session_dir: str = 'session',
                 session_name: str = 'training',
                 step_nb: int = -1,
                 record: bool = False):
        """
        """

        # Define the session repository
        self.session_dir = get_session_dir(session_dir, False)
        if not exists(path := join(self.session_dir, session_name)):
            raise ValueError(f"[{self.__class__.__name__}] The following directory does not exist: {path}")

        # Create a DatabaseManager
        self.database_manager = database_manager
        # self.database_manager = DatabaseManager(config=database_config,
        #                                         session=join(self.session_dir, session_name))
        self.database_manager.init_prediction_pipeline(session=join(self.session_dir, session_name),
                                                       produce_data=record)

        # Create a SimulationManager
        self.simulation_manager = SimulationManager(config=simulation_config,
                                                     pipeline='prediction',
                                                     session=path,
                                                     produce_data=record,
                                                     batch_size=1)
        self.simulation_manager.connect_to_database(database_path=self.database_manager.get_database_path(),
                                                    normalize_data=self.database_manager.normalize)

        # Create a NetworkManager
        self.network_manager = network_manager
        self.network_manager.init_prediction(session=path)
        self.network_manager.connect_to_database(database_path=self.database_manager.get_database_path(),
                                                 normalize_data=self.database_manager.normalize)
        self.network_manager.link_clients(1)

        self.simulation_manager.connect_to_network_manager(network_manager=self.network_manager)
        self.simulation_manager.connect_to_database_manager(database_manager=self.database_manager)

        # Prediction variables
        self.step_nb = step_nb
        self.step_id = 0
        self.produce_data = record

    def execute(self) -> None:
        """
        Launch the prediction Pipeline.
        Each event is already implemented for a basic pipeline but can also be rewritten via inheritance to describe a
        more complex Pipeline.
        """

        while self.step_id < self.step_nb if self.step_nb > 0 else True:
            self.step_id += 1

            # Get data from Dataset
            if self.simulation_manager.load_samples:
                self.data_lines = self.database_manager.get_data(batch_size=1)
                self.simulation_manager.dispatch_batch(data_lines=self.data_lines,
                                                       animate=True,
                                                       request_prediction=True,
                                                       save_data=self.produce_data)
            # Get data from Environment
            else:
                self.data_lines = self.simulation_manager.get_data(animate=True,
                                                                   request_prediction=True,
                                                                   save_data=self.produce_data)
                if self.produce_data:
                    self.database_manager.add_data(self.data_lines)

            if self.simulation_manager.environment_controller.viewer is not None and self.step_nb < 0:
                if not self.simulation_manager.environment_controller.viewer.is_open:
                    break

        self.simulation_manager.close()
        self.database_manager.close()
        self.network_manager.close()

    def __str__(self):

        description = "\n"
        description += f"# {self.__class__.__name__}\n"
        description += f"    Session repository: {self.session_dir}\n"
        description += f"   Number of step: {self.step_nb}\n"
        return description
