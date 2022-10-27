from typing import Any, Optional, Dict, List

from DeepPhysX.Core.Manager.DatabaseManager import DatabaseManager, Database
from DeepPhysX.Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig


class DataManager:

    def __init__(self,
                 pipeline: Any,
                 database_config: Optional[BaseDatabaseConfig] = None,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 session: str = 'sessions/default',
                 new_session: bool = True,
                 produce_data: bool = True,
                 batch_size: int = 1):

        """
        DataManager deals with the generation, storage and loading of training data.
        A batch is given with a call to 'get_data' on either the DatabaseManager or the EnvironmentManager according to
        the context.

        :param pipeline: Pipeline that handle the DataManager.
        :param database_config: Specialisation containing the parameters of the dataset manager.
        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param session: Path to the session directory.
        :param new_session: Flag that indicates whether if the session is new.
        :param produce_data: Flag that indicates whether if this session is producing data.
        :param int batch_size: Number of samples in a batch.
        """

        self.name: str = self.__class__.__name__

        # Managers variables
        self.pipeline: Optional[Any] = pipeline
        self.database_manager: Optional[DatabaseManager] = None
        self.environment_manager: Optional[EnvironmentManager] = None
        self.connected_managers: List[Any] = []

        # Create a DatabaseManager
        self.database_manager = DatabaseManager(database_config=database_config,
                                                data_manager=self,
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
                                                          batch_size=batch_size)

        # DataManager variables
        self.produce_data = produce_data
        self.batch_size = batch_size
        self.data_lines: List[List[int]] = []

    def connect_handler(self, handler):
        self.database_manager.connect_handler(handler)

    def get_database(self) -> Database:
        return self.database_manager.database

    def connect_manager(self, manager: Any):
        self.connected_managers.append(manager)

    def change_database(self) -> None:
        for manager in self.connected_managers:
            manager.change_database(self.database_manager.database)

    @property
    def nb_environment(self):
        if self.environment_manager is None:
            return None
        return 1 if self.environment_manager.server is None else self.environment_manager.number_of_thread

    def get_data(self,
                 epoch: int = 0,
                 animate: bool = True) -> None:
        """
        Fetch data from the EnvironmentManager or the DatabaseManager according to the context.

        :param epoch: Current epoch number.
        :param animate: Allow EnvironmentManager to generate a new sample.
        :return: Dict containing the newly computed data.
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
                self.data_lines = self.database_manager.get_data(batch_size=1)
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

    def get_prediction(self,
                       instance_id: int) -> None:
        """
        Get a Network prediction from an input array. Normalization is applied on input and prediction.

        :return: Network prediction.
        """

        # Get a prediction
        if self.pipeline is None:
            raise ValueError("Cannot request prediction if Manager (and then NetworkManager) does not exist.")
        self.pipeline.network_manager.compute_online_prediction(instance_id=instance_id,
                                                                normalization=self.normalization)

    @property
    def normalization(self) -> Dict[str, List[float]]:
        return self.database_manager.normalization

    def close(self) -> None:
        """
        Launch the closing procedure of Managers.
        """

        if self.environment_manager is not None:
            self.environment_manager.close()
        if self.database_manager is not None:
            self.database_manager.close()

    def __str__(self) -> str:
        """
        :return: A string containing valuable information about the DataManager
        """

        data_manager_str = ""
        if self.environment_manager:
            data_manager_str += str(self.environment_manager)
        if self.database_manager:
            data_manager_str += str(self.database_manager)
        return data_manager_str
