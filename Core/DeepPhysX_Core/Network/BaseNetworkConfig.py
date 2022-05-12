from typing import Any, Optional, Type
from collections import namedtuple
from os.path import isdir

from DeepPhysX_Core.Network.BaseNetwork import BaseNetwork
from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization
from DeepPhysX_Core.Network.DataTransformation import DataTransformation

NetworkType = BaseNetwork
OptimizationType = BaseOptimization
DataTransformationType = DataTransformation


class BaseNetworkConfig:
    """
    | BaseNetworkConfig is a configuration class to parameterize and create BaseNetwork, BaseOptimization and
      DataTransformation for the NetworkManager.

    :param Type[BaseNetwork] network_class: BaseNetwork class from which an instance will be created
    :param Type[BaseOptimization] optimization_class: BaseOptimization class from which an instance will be created
    :param Type[DataTransformation] data_transformation_class: DataTransformation class from which an instance will
                                                               be created
    :param Optional[str] network_dir: Name of an existing network repository
    :param str network_name: Name of the network
    :param str network_type: Type of the network
    :param int which_network: If several networks in network_dir, load the specified one
    :param bool save_each_epoch: If True, network state will be saved at each epoch end; if False, network state
                                 will be saved at the end of the training
    :param Optional[float] lr: Learning rate
    :param bool require_training_stuff: If specified, loss and optimizer class can be not necessary for training
    :param Optional[Any] loss: Loss class
    :param Optional[Any] optimizer: Network's parameters optimizer class
    """

    def __init__(self,
                 network_class: Type[BaseNetwork] = BaseNetwork,
                 optimization_class: Type[BaseOptimization] = BaseOptimization,
                 data_transformation_class: Type[DataTransformation] = DataTransformation,
                 network_dir: Optional[str] = None,
                 network_name: str = 'Network',
                 network_type: str = 'BaseNetwork',
                 which_network: int = 0,
                 save_each_epoch: bool = False,
                 lr: Optional[float] = None,
                 require_training_stuff: bool = True,
                 loss: Optional[Any] = None,
                 optimizer: Optional[Any] = None):

        # Check network_dir type and existence
        if network_dir is not None:
            if type(network_dir) != str:
                raise TypeError(
                    f"[{self.__class__.__name__}] Wrong 'network_dir' type: str required, get {type(network_dir)}")
            if not isdir(network_dir):
                raise ValueError(f"[{self.__class__.__name__}] Given 'network_dir' does not exists: {network_dir}")
        # Check network_name type
        if type(network_name) != str:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'network_name' type: str required, get {type(network_name)}")
        # Check network_tpe type
        if type(network_type) != str:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'network_type' type: str required, get {type(network_type)}")
        # Check which_network type and value
        if type(which_network) != int:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'which_network' type: int required, get {type(which_network)}")
        if which_network < 0:
            raise ValueError(f"[{self.__class__.__name__}] Given 'which_network' value is negative")
        # Check save_each_epoch type
        if type(save_each_epoch) != bool:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'save each epoch' type: bool required, get {type(save_each_epoch)}")

        # BaseNetwork parameterization
        self.network_class: Type[BaseNetwork] = network_class
        self.network_config: namedtuple = self.make_config(config_name='network_config', network_name=network_name,
                                                           network_type=network_type)

        # BaseOptimization parameterization
        self.optimization_class: Type[BaseOptimization] = optimization_class
        self.optimization_config: namedtuple = self.make_config(config_name='optimization_config', loss=loss, lr=lr,
                                                                optimizer=optimizer)
        self.training_stuff: bool = (loss is not None) and (optimizer is not None) or (not require_training_stuff)

        # NetworkManager parameterization
        self.data_transformation_class: Type[DataTransformation] = data_transformation_class
        self.data_transformation_config: namedtuple = self.make_config(config_name='data_transformation_config')
        self.network_dir: str = network_dir
        self.which_network: int = which_network
        self.save_each_epoch: bool = save_each_epoch and self.training_stuff

    def make_config(self, config_name: str, **kwargs) -> namedtuple:
        """
        | Create a namedtuple which gathers all the parameters for an Object configuration (Network or Optimization).
        | For a child config class, only new items are required since parent's items will be added by default.

        :param str config_name: Name of the configuration to fill
        :param kwargs: Items to add to the Object configuration
        :return: Namedtuple which contains Object parameters
        """

        # Get items set as keyword arguments
        fields = tuple(kwargs.keys())
        args = tuple(kwargs.values())
        # Check if a config already exists with the same name (child class will have the parent's config by default)
        if config_name in self.__dict__:
            config = self.__getattribute__(config_name)
            for key, value in config._asdict().items():
                # Only new items are required for children, check if the parent's items are set again anyway
                if key not in fields:
                    fields += (key,)
                    args += (value,)
        # Create namedtuple with collected items
        return namedtuple(config_name, fields)._make(args)

    def create_network(self) -> NetworkType:
        """
        | Create an instance of network_class with given parameters.

        :return: BaseNetwork object from network_class and its parameters.
        """

        try:
            network = self.network_class(config=self.network_config)
        except:
            raise ValueError(
                f"[{self.__class__.__name__}] Given 'network_class' cannot be created in {self.__class__.__name__}")
        if not isinstance(network, BaseNetwork):
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'network_class' type: BaseNetwork required, get "
                f"{self.network_class}")
        return network

    def create_optimization(self) -> OptimizationType:
        """
        | Create an instance of optimization_class with given parameters.

        :return: BaseOptimization object from optimization_class and its parameters.
        """

        try:
            optimization = self.optimization_class(config=self.optimization_config)
        except:
            raise ValueError(
                f"[{self.__class__.__name__}] Given 'optimization_class' got an unexpected keyword argument 'config'")
        if not isinstance(optimization, BaseOptimization):
            raise TypeError(f"[{self.__class__.__name__}] Wrong 'optimization_class' type: BaseOptimization required, "
                            f"get {self.optimization_class}")
        return optimization

    def create_data_transformation(self) -> DataTransformationType:
        """
        | Create an instance of data_transformation_class with given parameters.

        :return: DataTransformation object from data_transformation_class and its parameters.
        """

        try:
            data_transformation = self.data_transformation_class(config=self.data_transformation_config)
        except:
            raise ValueError(
                f"[{self.__class__.__name__}] Given 'data_transformation_class' got an unexpected keyword argument "
                f"'config'")
        if not isinstance(data_transformation, DataTransformation):
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'data_transformation_class' type: DataTransformation required, "
                f"get {self.data_transformation_class}")
        return data_transformation

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseDatasetConfig object
        """

        # Todo: fields in Configs are the set in Managers or objects, then remove __str__ method
        description = "\n"
        description += f"{self.__class__.__name__}\n"
        description += f"    Network class: {self.network_class.__name__}\n"
        description += f"    Optimization class: {self.optimization_class.__name__}\n"
        description += f"    Training materials: {self.training_stuff}\n"
        description += f"    Network directory: {self.network_dir}\n"
        description += f"    Which network: {self.which_network}\n"
        description += f"    Save each epoch: {self.save_each_epoch}\n"
        return description
