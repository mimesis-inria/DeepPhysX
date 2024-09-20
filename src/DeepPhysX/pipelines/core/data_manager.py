from typing import Any, Optional, Dict, List

from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.simulation.core.environment_manager import EnvironmentManager
from DeepPhysX.simulation.core.base_environment_config import BaseEnvironmentConfig
from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.database.database_handler import DatabaseHandler


class DataManager:

    def __init__(self,
                 pipeline: Any,
                 database_config: Optional[DatabaseConfig] = None,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 new_session: bool = True,
                 session: str = 'sessions/default',
                 produce_data: bool = True,
                 batch_size: int = 1):

        """
        DataManager deals with the generation, storage and loading of training data.

        :param pipeline: Pipeline that handle the DataManager.
        :param database_config: Configuration object with the parameters of the Database.
        :param environment_config: Configuration object with the parameters of the Environment.
        :param new_session: If True, the session is done in a new repository.
        :param session: Path to the session repository.
        :param produce_data: If True, this session will store data in the Database.
        :param batch_size: Number of samples in a single batch.
        """

        self.name: str = self.__class__.__name__

        # Session variables
        self.pipeline: Optional[Any] = pipeline
        self.database_manager: Optional[DatabaseManager] = None
        self.environment_manager: Optional[EnvironmentManager] = None

        # Create a DatabaseManager
        self.database_manager = DatabaseManager(database_config=database_config,
                                                pipeline=pipeline.type,
                                                session=session,
                                                new_session=new_session,
                                                produce_data=produce_data)

        # Create an EnvironmentManager if required
        if environment_config is not None:
            self.environment_manager = EnvironmentManager(environment_config=environment_config,
                                                          data_manager=self,
                                                          pipeline=pipeline.type,
                                                          session=session,
                                                          produce_data=produce_data,
                                                          batch_size=batch_size)

        # DataManager variables
        self.produce_data = produce_data
        self.batch_size = batch_size
        self.data_lines: List[List[int]] = []

    @property
    def nb_environment(self) -> Optional[int]:
        """
        Get the number of Environments managed by the EnvironmentManager.
        """

        if self.environment_manager is None:
            return None
        return 1 if self.environment_manager.server is None else self.environment_manager.number_of_thread

    @property
    def normalization(self) -> Dict[str, List[float]]:
        """
        Get the normalization coefficients computed by the DatabaseManager.
        """

        return self.database_manager.normalization

    def connect_handler(self,
                        handler: DatabaseHandler) -> None:
        """
        Add a new DatabaseHandler to the list of handlers of the DatabaseManager.

        :param handler: New handler to register.
        """

        self.database_manager.connect_handler(handler)

    def get_data(self,
                 epoch: int = 0,
                 animate: bool = True,
                 load_samples: bool = True) -> None:
        """
        Fetch data from the EnvironmentManager or the DatabaseManager according to the context.

        :param epoch: Current epoch number.
        :param animate: Allow EnvironmentManager to trigger a step itself in order to generate a new sample.
        :param load_samples: If True, trigger a sample loading from the Database.
        """

        # Data generation case
        if self.pipeline.type == 'data_generation':
            self.environment_manager.get_data(animate=animate)
            self.database_manager.add_data()

        # Training case
        elif self.pipeline.type == 'training':

            # Get data from Environment(s) if used and if the data should be created at this epoch
            if self.environment_manager is not None and self.produce_data and \
                    (epoch == 0 or self.environment_manager.always_produce):
                self.data_lines = self.environment_manager.get_data(animate=animate)
                self.database_manager.add_data(self.data_lines)

            # Get data from Dataset
            else:
                self.data_lines = self.database_manager.get_data(batch_size=self.batch_size)
                # Dispatch a batch to clients
                if self.environment_manager is not None:
                    if self.environment_manager.load_samples and \
                            (epoch == 0 or not self.environment_manager.only_first_epoch):
                        self.environment_manager.dispatch_batch(data_lines=self.data_lines,
                                                                animate=animate)
                    # Environment is no longer used
                    else:
                        self.environment_manager.close()
                        self.environment_manager = None

        # Prediction pipeline
        else:

            # Get data from Dataset
            if self.environment_manager.load_samples:
                self.data_lines = self.database_manager.get_data(batch_size=1) if load_samples else None
                self.environment_manager.dispatch_batch(data_lines=self.data_lines,
                                                        animate=animate,
                                                        request_prediction=True,
                                                        save_data=self.produce_data)
            # Get data from Environment
            else:
                self.data_lines = self.environment_manager.get_data(animate=animate,
                                                                    request_prediction=True,
                                                                    save_data=self.produce_data)
                if self.produce_data:
                    self.database_manager.add_data(self.data_lines)

    def load_sample(self) -> List[int]:
        """
        Load a sample from the Database.

        :return: Index of the loaded line.
        """

        self.data_lines = self.database_manager.get_data(batch_size=1)
        return self.data_lines[0]

    def get_prediction(self,
                       instance_id: int) -> None:
        """
        Get a networks prediction for the specified Environment instance.
        """

        # Get a prediction
        if self.pipeline is None:
            raise ValueError("Cannot request prediction if Manager (and then NetworkManager) does not exist.")
        self.pipeline.network_manager.compute_online_prediction(instance_id=instance_id,
                                                                normalization=self.normalization)

    def close(self) -> None:
        """
        Launch the closing procedure of the DataManager.
        """

        if self.environment_manager is not None:
            self.environment_manager.close()
        if self.database_manager is not None:
            self.database_manager.close()

    def __str__(self):

        data_manager_str = ""
        if self.environment_manager:
            data_manager_str += str(self.environment_manager)
        if self.database_manager:
            data_manager_str += str(self.database_manager)
        return data_manager_str
