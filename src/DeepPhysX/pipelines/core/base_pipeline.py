from typing import Optional, Any, List, Union

from DeepPhysX.networks.core.dpx_network_config import DPXNetworkConfig
from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.simulation.core.base_environment_config import BaseEnvironmentConfig
from DeepPhysX.pipelines.core.data_manager import DataManager
from DeepPhysX.networks.core.stats_manager import StatsManager
from DeepPhysX.networks.core.network_manager import NetworkManager
from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.simulation.core.environment_manager import EnvironmentManager
from DeepPhysX.utils.path import get_session_dir


class BasePipeline:

    def __init__(self,
                 network_config: Optional[DPXNetworkConfig] = None,
                 database_config: Optional[DatabaseConfig] = None,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 new_session: bool = True,
                 session_dir: str = 'sessions',
                 session_name: str = 'default',
                 pipeline: str = ''):
        """
        pipelines implement the main loop that defines data flow through components (Environment, Dataset, networks...).

        :param network_config: Configuration object with the parameters of the networks.
        :param database_config: Configuration object with the parameters of the Database.
        :param environment_config: Configuration object with the parameters of the Environment.
        :param new_session: If True, a new repository will be created for this session.
        :param session_dir: Path to the directory which contains your DeepPhysX session repositories.
        :param session_name: Name of the current session repository.
        :param pipeline: Name of the Pipeline.
        """

        self.name: str = self.__class__.__name__

        # Check the configurations
        if network_config is not None and not isinstance(network_config, BaseNetworkConfig):
            raise TypeError(f"[{self.name}] The networks configuration must be a BaseNetworkConfig object.")
        if database_config is not None and not isinstance(database_config, DatabaseConfig):
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
        self.database_config: DatabaseConfig = database_config
        self.network_config: BaseNetworkConfig = network_config
        self.environment_config: BaseEnvironmentConfig = environment_config

        # Session variables
        self.session_dir = get_session_dir(session_dir, new_session)
        self.session_name = session_name
        self.new_session = new_session
        self.type = pipeline

    def execute(self):
        """
        Launch the Pipeline.
        """

        raise NotImplemented

    def __get_any_manager(self,
                          manager_names: Union[str, List[str]]) -> Optional[Any]:
        """
        Return the desired Manager associated with the Pipeline if it exists.

        :param manager_names: Name of the desired Manager or order of access to this desired Manager.
        :return: The desired Manager associated with the Pipeline.
        """

        # Direct access to manager
        if type(manager_names) == str:
            return getattr(self, manager_names) if hasattr(self, manager_names) else None

        # Intermediates to access manager
        accessed_manager = self
        for next_manager in manager_names:
            if hasattr(accessed_manager, next_manager):
                accessed_manager = getattr(accessed_manager, next_manager)
            else:
                return None
        return accessed_manager

    def get_network_manager(self) -> Optional[NetworkManager]:
        """
        Return the NetworkManager associated with the Pipeline if it exists.

        :return: The NetworkManager associated with the Pipeline.
        """

        return self.__get_any_manager(manager_names='network_manager')

    def get_data_manager(self) -> Optional[DataManager]:
        """
        Return the DataManager associated with the Pipeline if it exists.

        :return: The DataManager associated with the Pipeline.
        """

        return self.__get_any_manager(manager_names='data_manager')

    def get_stats_manager(self) -> Optional[StatsManager]:
        """
        Return the StatsManager associated with the Pipeline if it exists.

        :return: The StatsManager associated with the Pipeline.
        """

        return self.__get_any_manager(manager_names='stats_manager')

    def get_database_manager(self) -> Optional[DatabaseManager]:
        """
        Return the DatabaseManager associated with the Pipeline if it exists.

        :return: The DatabaseManager associated with the Pipeline.
        """

        return self.__get_any_manager(manager_names=['data_manager', 'database_manager'])

    def get_environment_manager(self) -> Optional[EnvironmentManager]:
        """
        Return the EnvironmentManager associated with the Pipeline if it exists.

        :return: The EnvironmentManager associated with the Pipeline.
        """

        return self.__get_any_manager(manager_names=['data_manager', 'environment_manager'])

    def __str__(self):

        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Pipeline type: {self.type}\n"
        description += f"    Session repository: {self.session_dir}\n"
        return description
