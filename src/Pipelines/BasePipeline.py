from typing import Dict, Optional, Any, List, Union

from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Manager.Manager import Manager
from DeepPhysX.Core.Manager.NetworkManager import NetworkManager
from DeepPhysX.Core.Manager.DataManager import DataManager
from DeepPhysX.Core.Manager.StatsManager import StatsManager
from DeepPhysX.Core.Manager.DatasetManager import DatasetManager
from DeepPhysX.Core.Manager.EnvironmentManager import EnvironmentManager


class BasePipeline:
    """
    | Base class defining Pipelines common variables.

    :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
    :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
    :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
    :param str session_name: Name of the newly created directory if session is not defined
    :param Optional[str] session_dir: Name of the directory in which to write all the necessary data
    :param Optional[str] pipeline: Values at either 'training' or 'prediction'
    """

    def __init__(self,
                 network_config: Optional[BaseNetworkConfig] = None,
                 dataset_config: Optional[BaseDatasetConfig] = None,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 session_name: str = 'default',
                 session_dir: Optional[str] = None,
                 pipeline: Optional[str] = None):

        self.name: str = self.__class__.__name__

        # Check the arguments
        if network_config is not None and not isinstance(network_config, BaseNetworkConfig):
            raise TypeError(f"[{self.name}] The network configuration must be a BaseNetworkConfig")
        if environment_config is not None and not isinstance(environment_config, BaseEnvironmentConfig):
            raise TypeError(f"[{self.name}] The environment configuration must be a BaseEnvironmentConfig")
        if dataset_config is not None and not isinstance(dataset_config, BaseDatasetConfig):
            raise TypeError(f"[{self.name}] The dataset configuration must be a BaseDatasetConfig")
        if type(session_name) != str:
            raise TypeError(f"[{self.name}] The network config must be a BaseNetworkConfig object.")
        if session_dir is not None and type(session_dir) != str:
            raise TypeError(f"[{self.name}] The session directory must be a str.")

        self.type: str = pipeline    # Either training or prediction
        self.debug: bool = False
        self.new_session: bool = True
        self.record_data: Optional[Dict[str, bool]] = None  # Can be of type {'in': bool, 'out': bool}

        # Dataset variables
        self.dataset_config: BaseDatasetConfig = dataset_config
        # Network variables
        self.network_config: BaseNetworkConfig = network_config
        # Simulation variables
        self.environment_config: BaseEnvironmentConfig = environment_config
        # Main manager
        self.manager: Optional[Manager] = None

    def get_any_manager(self, manager_names: Union[str, List[str]]) -> Optional[Any]:
        """
        | Return the desired Manager associated with the pipeline if it exists.

        :param Union[str, List[str]] manager_names: Name of the desired Manager or order of access to the desired
                                                    Manager
        :return: Manager associated with the Pipeline
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

    def get_network_manager(self) -> NetworkManager:
        """
        | Return the NetworkManager associated with the pipeline.

        :return: NetworkManager associated with the pipeline
        """

        return self.get_any_manager('network_manager')

    def get_data_manager(self) -> DataManager:
        """
        | Return the DataManager associated with the pipeline.

        :return: DataManager associated with the pipeline
        """

        return self.get_any_manager('data_manager')

    def get_stats_manager(self) -> StatsManager:
        """
        | Return the StatsManager associated with the pipeline.

        :return: StatsManager associated with the pipeline
        """

        return self.get_any_manager('stats_manager')

    def get_dataset_manager(self) -> DatasetManager:
        """
        | Return the DatasetManager associated with the pipeline.

        :return: DatasetManager associated with the pipeline
        """

        return self.get_any_manager(['data_manager', 'dataset_manager'])

    def get_environment_manager(self) -> EnvironmentManager:
        """
        | Return the EnvironmentManager associated with the pipeline.

        :return: EnvironmentManager associated with the pipeline
        """

        return self.get_any_manager(['data_manager', 'environment_manager'])
