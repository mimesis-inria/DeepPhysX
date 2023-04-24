from typing import Any, Optional, Type
from os.path import isdir
from numpy import sctypeDict

from DeepPhysX.Core.Network.BaseNetwork import BaseNetwork
from DeepPhysX.Core.Network.BaseOptimization import BaseOptimization
from DeepPhysX.Core.Network.BaseTransformation import BaseTransformation
from DeepPhysX.Core.Utils.configs import make_config, namedtuple


class BaseNetworkConfig:

    def __init__(self,
                 network_class: Type[BaseNetwork] = BaseNetwork,
                 optimization_class: Type[BaseOptimization] = BaseOptimization,
                 data_transformation_class: Type[BaseTransformation] = BaseTransformation,
                 network_dir: Optional[str] = None,
                 network_name: str = 'Network',
                 network_type: str = 'BaseNetwork',
                 which_network: int = -1,
                 save_each_epoch: bool = False,
                 data_type: str = 'float32',
                 lr: Optional[float] = None,
                 require_training_stuff: bool = True,
                 loss: Optional[Any] = None,
                 loss_parameters: Type[dict] = None,
                 optimizer: Optional[Any] = None,
                 optimizer_parameters: Type[dict] = None,
                 scheduler_class: Any = None,
                 scheduler_parameters: dict = None):
        """
        BaseNetworkConfig is a configuration class to parameterize and create BaseNetwork, BaseOptimization and
        BaseTransformation for the NetworkManager.

        :param network_class: BaseNetwork class from which an instance will be created.
        :param optimization_class: BaseOptimization class from which an instance will be created.
        :param data_transformation_class: BaseTransformation class from which an instance will be created.
        :param network_dir: Name of an existing network repository.
        :param network_name: Name of the network.
        :param network_type: Type of the network.
        :param which_network: If several networks in network_dir, load the specified one.
        :param save_each_epoch: If True, network state will be saved at each epoch end; if False, network state
                                will be saved at the end of the training.
        :param data_type: Type of the training data.
        :param lr: Learning rate.
        :param require_training_stuff: If specified, loss and optimizer class can be not necessary for training.
        :param loss: Loss class.
        :param optimizer: Network's parameters optimizer class.
        """

        self.name = self.__class__.__name__

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
        # Check save_each_epoch type
        if type(save_each_epoch) != bool:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'save each epoch' type: bool required, get {type(save_each_epoch)}")
        # Check data type
        if data_type not in sctypeDict:
            raise ValueError(
                f"[{self.__class__.__name__}] The following data type is not a numpy type: {data_type}")

        # BaseNetwork parameterization
        self.network_class: Type[BaseNetwork] = network_class
        self.network_config: namedtuple = make_config(configuration_object=self,
                                                      configuration_name='network_config',
                                                      network_name=network_name,
                                                      network_type=network_type,
                                                      data_type=data_type)

        # BaseOptimization parameterization
        self.optimization_class: Type[BaseOptimization] = optimization_class
        self.optimization_config: namedtuple = make_config(configuration_object=self,
                                                           configuration_name='optimization_config',
                                                           loss=loss,
                                                           lr=lr,
                                                           optimizer=optimizer,
                                                           optimizer_parameters=optimizer_parameters,
                                                           loss_parameters=loss_parameters,
                                                           scheduler_class=scheduler_class,
                                                           scheduler_parameters=scheduler_parameters)
        self.training_stuff: bool = (loss is not None) and (optimizer is not None) or (not require_training_stuff)

        # NetworkManager parameterization
        self.data_transformation_class: Type[BaseTransformation] = data_transformation_class
        self.data_transformation_config: namedtuple = make_config(configuration_object=self,
                                                                  configuration_name='data_transformation_config')

        # NetworkManager parameterization
        self.network_dir: str = network_dir
        self.which_network: int = which_network
        self.save_each_epoch: bool = save_each_epoch and self.training_stuff

    def create_network(self) -> BaseNetwork:
        """
        Create an instance of network_class with given parameters.

        :return: BaseNetwork object from network_class and its parameters.
        """

        # Create instance
        network = self.network_class(config=self.network_config)
        if not isinstance(network, BaseNetwork):
            raise TypeError(f"[{self.name}] The given 'network_class'={self.network_class} must be a BaseNetwork.")
        return network

    def create_optimization(self) -> BaseOptimization:
        """
        Create an instance of optimization_class with given parameters.

        :return: BaseOptimization object from optimization_class and its parameters.
        """

        # Create instance
        optimization = self.optimization_class(config=self.optimization_config)
        if not isinstance(optimization, BaseOptimization):
            raise TypeError(f"[{self.name}] The given 'optimization_class'={self.optimization_class} must be a "
                            f"BaseOptimization.")
        return optimization

    def create_data_transformation(self) -> BaseTransformation:
        """
        Create an instance of data_transformation_class with given parameters.

        :return: BaseTransformation object from data_transformation_class and its parameters.
        """

        # Create instance
        data_transformation = self.data_transformation_class(config=self.data_transformation_config)
        if not isinstance(data_transformation, BaseTransformation):
            raise TypeError(f"[{self.name}] The given 'data_transformation_class'={self.data_transformation_class} "
                            f"must be a BaseTransformation.")
        return data_transformation

    def __str__(self):

        description = "\n"
        description += f"{self.__class__.__name__}\n"
        description += f"    Network class: {self.network_class.__name__}\n"
        description += f"    Optimization class: {self.optimization_class.__name__}\n"
        description += f"    Training materials: {self.training_stuff}\n"
        description += f"    Network directory: {self.network_dir}\n"
        description += f"    Which network: {self.which_network}\n"
        description += f"    Save each epoch: {self.save_each_epoch}\n"
        return description
