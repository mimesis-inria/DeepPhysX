from typing import Optional
from os.path import join, exists

from DeepPhysX.simulation.core.simulation_manager import SimulationManager, SimulationConfig
from DeepPhysX.database.database_manager import DatabaseManager, DatabaseConfig
from DeepPhysX.networks.core.network_manager import NetworkManager, NetworkConfig
from DeepPhysX.utils.path import get_session_dir


class PredictionPipeline:

    def __init__(self,
                 network_config: NetworkConfig,
                 simulation_config: SimulationConfig,
                 database_config: Optional[DatabaseConfig] = None,
                 session_dir: str = 'session',
                 session_name: str = 'training',
                 step_nb: int = -1,
                 record: bool = False):
        """
        BasePrediction is a pipeline defining the running process of an artificial neural networks.
        It provides a highly tunable learning process that can be used with any machine learning library.

        :param network_config: Configuration object with the parameters of the networks.
        :param simulation_config: Configuration object with the parameters of the Environment.
        :param database_config: Configuration object with the parameters of the Database.
        :param session_dir: Relative path to the directory which contains sessions repositories.
        :param session_name: Name of the new the session repository.
        :param step_nb: Number of simulation step to play.
        :param record: If True, prediction data will be saved in a dedicated Database.
        """

        # Define the session repository
        self.session_dir = get_session_dir(session_dir, False)
        if not exists(path := join(self.session_dir, session_name)):
            raise ValueError(f"[{self.__class__.__name__}] The following directory does not exist: {path}")

        # Create a DatabaseManager
        self.database_manager = DatabaseManager(config=database_config,
                                                session=join(self.session_dir, session_name))
        self.database_manager.init_prediction_pipeline(produce_data=record)

        # Create a SimulationManager
        self.simulation_manager = SimulationManager(config=simulation_config,
                                                     pipeline='prediction',
                                                     session=path,
                                                     produce_data=record,
                                                     batch_size=1)
        self.simulation_manager.connect_to_database(database_path=self.database_manager.get_database_path(),
                                                    normalize_data=self.database_manager.config.normalize)

        # Create a NetworkManager
        self.network_manager = NetworkManager(network_config=network_config,
                                              pipeline='prediction',
                                              session=path,
                                              new_session=False)
        self.network_manager.connect_to_database(database_path=self.database_manager.get_database_path(),
                                                 normalize_data=self.database_manager.config.normalize)
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

        self.simulation_manager.close()
        self.database_manager.close()
        self.network_manager.close()

    def __str__(self):

        description = "\n"
        description += f"# {self.__class__.__name__}\n"
        description += f"    Session repository: {self.session_dir}\n"
        description += f"   Number of step: {self.step_nb}\n"
        return description
