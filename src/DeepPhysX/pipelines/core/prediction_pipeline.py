from typing import Optional
from os.path import join, exists

from DeepPhysX.pipelines.core.base_pipeline import BasePipeline
from DeepPhysX.simulation.core.simulation_manager import SimulationManager, SimulationConfig
from DeepPhysX.database.database_manager import DatabaseManager, DatabaseConfig
from DeepPhysX.networks.core.network_manager import NetworkManager, NetworkConfig


class Prediction(BasePipeline):

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

        BasePipeline.__init__(self,
                              network_config=network_config,
                              database_config=database_config,
                              environment_config=simulation_config,
                              session_dir=session_dir,
                              session_name=session_name,
                              new_session=False,
                              pipeline='prediction')

        # Define the session repository
        if not exists(path := join(self.session_dir, self.session_name)):
            raise ValueError(f"[{self.name}] The following directory does not exist: {path}")



        # Create a DataManager
        # self.data_manager = DataManager(pipeline=self,
        #                                 database_config=database_config,
        #                                 environment_config=environment_config,
        #                                 session=path,
        #                                 new_session=False,
        #                                 produce_data=record,
        #                                 batch_size=1)
        # self.data_manager.connect_handler(self.network_manager.get_database_handler())
        self.database_manager = DatabaseManager(database_config=database_config,
                                                pipeline=self.type,
                                                session=join(self.session_dir, self.session_name),
                                                new_session=False,
                                                produce_data=record)
        self.simulation_manager = SimulationManager(config=simulation_config,
                                                     pipeline=self.type,
                                                     session=path,
                                                     produce_data=record,
                                                     batch_size=1)
        self.simulation_manager.connect_to_database(**self.database_manager.get_database_paths())

        # Create a NetworkManager
        self.network_manager = NetworkManager(network_config=network_config,
                                              pipeline=self.type,
                                              session=path,
                                              new_session=False)
        self.network_manager.connect_to_database(**self.database_manager.get_database_paths())
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

        self.prediction_begin()
        while self.prediction_condition():
            self.sample_begin()
            self.predict()
            self.sample_end()
        self.prediction_end()

    def prediction_begin(self) -> None:
        """
        Called once at the beginning of the prediction Pipeline.
        """

        pass

    def prediction_condition(self) -> bool:
        """
        Condition that characterize the end of the prediction Pipeline.
        """

        running = self.step_id < self.step_nb if self.step_nb > 0 else True
        self.step_id += 1
        return running

    def sample_begin(self) -> None:
        """
        Called one at the beginning of each sample.
        """

        pass

    def predict(self) -> None:
        """
        Pull the data from the manager and return the prediction.
        """

        # self.data_manager.get_data(epoch=0, animate=True)
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

    def sample_end(self) -> None:
        """
        Called one at the end of each sample.
        """

        pass

    def prediction_end(self) -> None:
        """
        Called once at the end of the prediction Pipeline.
        """

        self.simulation_manager.close()
        self.database_manager.close()
        self.network_manager.close()

    def __str__(self):

        description = BasePipeline.__str__(self)
        description += f"   Number of step: {self.step_nb}\n"
        return description
