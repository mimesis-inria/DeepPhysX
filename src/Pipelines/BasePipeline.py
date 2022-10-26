from typing import Optional, Any, List, Union

from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Manager.Manager import Manager
from DeepPhysX.Core.Manager.DataManager import DataManager
from DeepPhysX.Core.Manager.StatsManager import StatsManager
from DeepPhysX.Core.Manager.NetworkManager import NetworkManager
from DeepPhysX.Core.Manager.DatabaseManager import DatabaseManager
from DeepPhysX.Core.Manager.EnvironmentManager import EnvironmentManager


class BasePipeline:

    def __init__(self,
                 network_config: Optional[BaseNetworkConfig] = None,
                 database_config: Optional[BaseDatabaseConfig] = None,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 session_dir: str = 'sessions',
                 session_name: str = 'default',
                 new_session: bool = True,
                 pipeline: str = ''):
        """
        Pipelines implement the main loop that defines data flow through components (Environment, Dataset, Network...).

        :param network_config: Specialisation containing the parameters of the network manager.
        :param database_config: Specialisation containing the parameters of the dataset manager.
        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param session_dir: Name of the directory in which to write all the necessary data.
        :param session_name: Name of the newly created directory if session is not defined.
        :param new_session: If True, the session will be run in a new repository.
        :param pipeline: Name of the Pipeline.
        """

        self.name: str = self.__class__.__name__

        # Check the configurations
        if network_config is not None and not isinstance(network_config, BaseNetworkConfig):
            raise TypeError(f"[{self.name}] The Network configuration must be a BaseNetworkConfig object.")
        if database_config is not None and not isinstance(database_config, BaseDatabaseConfig):
            raise TypeError(f"[{self.name}] The Dataset configuration must be a BaseDatabaseConfig object.")
        if environment_config is not None and not isinstance(environment_config, BaseEnvironmentConfig):
            raise TypeError(f"[{self.name}] The Environment configuration must be a BaseEnvironmentConfig object.")

        # Check the session path variables
        if type(session_dir) != str:
            raise TypeError(f"[{self.name}] The given 'session_dir'={session_dir} must be a str.")
        elif len(session_dir) == 0:
            session_dir = 'sessions'
        if type(session_name) != str:
            raise TypeError(f"[{self.name}] The given 'session=name'={session_name} must be a str.")
        elif len(session_name) == 0:
            session_name = pipeline

        # Configuration variables
        self.database_config: BaseDatabaseConfig = database_config
        self.network_config: BaseNetworkConfig = network_config
        self.environment_config: BaseEnvironmentConfig = environment_config

        # Session variables
        self.session_dir = session_dir
        self.session_name = session_name
        self.new_session = new_session
        self.type = pipeline

        # Main manager
        self.manager: Optional[Manager] = None

    def execute(self):
        """
        Launch the Pipeline.
        """

        raise NotImplemented

    def __get_any_manager(self,
                          manager_names: Union[str, List[str]]) -> Optional[Any]:
        """
        Return the desired Manager associated with the pipeline if it exists.

        :param manager_names: Name of the desired Manager or order of access to this desired Manager.
        :return: The desired Manager associated with the Pipeline.
        """

        # If manager variable is not defined, cannot access other manager
        if self.manager is None:
            return None

        # Direct access to manager
        if type(manager_names) == str:
            return getattr(self.manager, manager_names) if hasattr(self.manager, manager_names) else None

        # Intermediates to access manager
        accessed_manager = self.manager
        for next_manager in manager_names:
            if hasattr(accessed_manager, next_manager):
                accessed_manager = getattr(accessed_manager, next_manager)
            else:
                return None
        return accessed_manager

    def get_network_manager(self) -> Optional[NetworkManager]:
        """
        Return the NetworkManager associated with the pipeline if it exists.

        :return: The NetworkManager associated with the pipeline.
        """

        return self.__get_any_manager(manager_names='network_manager')

    def get_data_manager(self) -> Optional[DataManager]:
        """
        Return the DataManager associated with the pipeline if it exists.

        :return: The DataManager associated with the pipeline.
        """

        return self.__get_any_manager(manager_names='data_manager')

    def get_stats_manager(self) -> Optional[StatsManager]:
        """
        Return the StatsManager associated with the pipeline if it exists.

        :return: The StatsManager associated with the pipeline.
        """

        return self.__get_any_manager(manager_names='stats_manager')

    def get_dataset_manager(self) -> Optional[DatabaseManager]:
        """
        Return the DatabaseManager associated with the pipeline if it exists.

        :return: The DatabaseManager associated with the pipeline.
        """

        return self.__get_any_manager(manager_names=['data_manager', 'database_manager'])

    def get_environment_manager(self) -> Optional[EnvironmentManager]:
        """
        Return the EnvironmentManager associated with the pipeline if it exists.

        :return: The EnvironmentManager associated with the pipeline.
        """

        return self.__get_any_manager(manager_names=['data_manager', 'environment_manager'])
