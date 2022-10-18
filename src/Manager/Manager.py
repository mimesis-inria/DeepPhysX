from typing import Dict, Tuple, Optional
from os.path import join, exists
from numpy import ndarray

from DeepPhysX.Core.Manager.DataManager import DataManager
from DeepPhysX.Core.Manager.NetworkManager import NetworkManager
from DeepPhysX.Core.Manager.StatsManager import StatsManager
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Utils.path import get_first_caller, create_dir


class Manager:

    def __init__(self,
                 network_config: Optional[BaseNetworkConfig] = None,
                 dataset_config: Optional[BaseDatasetConfig] = None,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 session_dir: str = 'sessions',
                 session_name: str = 'default',
                 new_session: bool = True,
                 pipeline: str = '',
                 produce_data: bool = True,
                 batch_size: int = 1,
                 debug_session: bool = False):
        """
        Collection of all the specialized managers. Allows for some basic functions call.
        More specific behaviour have to be directly call from the corresponding manager.

        :param network_config: Specialisation containing the parameters of the network manager.
        :param dataset_config: Specialisation containing the parameters of the dataset manager.
        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param session_name: Name of the newly created directory if session is not defined.
        :param session_dir: Name of the directory in which to write all the necessary data.
        :param bool new_session: Flag that indicates whether if the session is new
        :param is_training: Flag that indicates whether if this session is training a Network.
        :param produce_data: Flag that indicates whether if this session is producing data.
        :param int batch_size: Number of samples in a batch.
        """

        self.name = self.__class__.__name__

        # Define the session repository
        root = get_first_caller()
        session_dir = join(root, session_dir)

        if pipeline in ['data_generation', 'training']:
            # Avoid unwanted overwritten data
            if new_session:
                self.session = create_dir(session_dir=session_dir,
                                          session_name=session_name)
            else:
                self.session = join(session_dir, session_name)
        else:
            self.session = join(session_dir, session_name)
            if not exists(self.session):
                raise ValueError(f"[{self.name}] The running session directory {self.session} does not exist.")

        # Create a DataManager
        self.data_manager = DataManager(dataset_config=dataset_config,
                                        environment_config=environment_config,
                                        manager=self,
                                        session=self.session,
                                        new_session=new_session,
                                        pipeline=pipeline,
                                        produce_data=produce_data,
                                        batch_size=batch_size)
        database = self.data_manager.get_database()
        self.batch_size = batch_size

        # Create a NetworkMmanager
        self.network_manager = NetworkManager(network_config=network_config,
                                              manager=self,
                                              session=self.session,
                                              new_session=new_session,
                                              pipeline=pipeline,
                                              data_db=database)

        # Create a StatsManager for training only
        self.debug_session = debug_session
        self.stats_manager = StatsManager(manager=self,
                                          session=self.session) if pipeline == 'training' else None

    def change_database(self, database):
        self.network_manager.change_database(database)

    def get_data(self,
                 epoch: int = 0,
                 animate: bool = True) -> None:
        """
        Fetch data from the DataManager.

        :param int epoch: Epoch ID
        :param bool animate: If True allows running environment step
        """

        self.data_manager.get_data(epoch=epoch,
                                   animate=animate)

    def optimize_network(self) -> Tuple[ndarray, Dict[str, float]]:
        """
        Compute a prediction and run a back propagation with the current batch.

        :return: The network prediction and the associated loss value
        """

        # Forward pass and optimization step
        data_pred, data_loss = self.network_manager.compute_prediction_and_loss(
            data_lines=self.data_manager.data_lines,
            normalization=self.data_manager.normalization,
            optimize=True)
        return data_pred, data_loss

    def save_network(self) -> None:
        """
        Save network parameters.
        """

        self.network_manager.save_network()

    def close(self) -> None:
        """
        Call all managers close procedure.
        """

        if self.network_manager is not None:
            self.network_manager.close()
        if self.stats_manager is not None:
            self.stats_manager.close()
        if self.data_manager is not None:
            self.data_manager.close()

    def __str__(self) -> str:

        manager_description = ""
        if self.network_manager is not None:
            manager_description += str(self.network_manager)
        if self.data_manager is not None:
            manager_description += str(self.data_manager)
        if self.stats_manager is not None:
            manager_description += str(self.stats_manager)
        return manager_description
