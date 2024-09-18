from typing import Any, Optional, Type
from os.path import isdir
from numpy import sctypeDict

from DeepPhysX.networks.core.dpx_network import DPXNetwork
from DeepPhysX.networks.core.dpx_optimization import DPXOptimization
from DeepPhysX.networks.core.dpx_transformation import DPXTransformation
from DeepPhysX.utils.configs import make_config, namedtuple


class DPXNetworkConfig:

    def __init__(self,
                 network_class: Type[DPXNetwork] = DPXNetwork,
                 optimization_class: Type[DPXOptimization] = DPXOptimization,
                 data_transformation_class: Type[DPXTransformation] = DPXTransformation,
                 network_dir: Optional[str] = None,
                 network_name: str = 'networks',
                 network_type: str = 'DPXNetwork',
                 which_network: int = -1,
                 save_intermediate_state_every: int = 0,
                 data_type: str = 'float32',
                 lr: Optional[float] = None,
                 require_training_stuff: bool = True,
                 loss: Optional[Any] = None,
                 optimizer: Optional[Any] = None):
        """
        DPXNetworkConfig is a configuration class to parameterize and create DPXNetwork, DPXOptimization and
        DPXTransformation for the NetworkManager.

        :param network_class: DPXNetwork class from which an instance will be created.
        :param optimization_class: DPXOptimization class from which an instance will be created.
        :param data_transformation_class: DPXTransformation class from which an instance will be created.
        :param network_dir: Name of an existing networks repository.
        :param network_name: Name of the networks.
        :param network_type: Type of the networks.
        :param which_network: If several networks in network_dir, load the specified one.
        :param save_intermediate_state_every: Save the current state of the networks periodically.
        :param data_type: Type of the training data.
        :param lr: Learning rate.
        :param require_training_stuff: If specified, loss and optimizer class can be not necessary for training.
        :param loss: Loss class.
        :param optimizer: networks's parameters optimizer class.
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
        # Check save_intermediate_states type and value
        if type(save_intermediate_state_every) != int:
            raise TypeError(
                f"[{self.__class__.__name__}] Wrong 'save_intermediate_state_every' type: int required, get "
                f"{type(save_intermediate_state_every)}")
        # Check data type
        if data_type not in sctypeDict:
            raise ValueError(
                f"[{self.__class__.__name__}] The following data type is not a numpy type: {data_type}")

        # DPXNetwork parameterization
        self.network_class: Type[DPXNetwork] = network_class
        self.network_config: namedtuple = make_config(configuration_object=self,
                                                      configuration_name='network_config',
                                                      network_name=network_name,
                                                      network_type=network_type,
                                                      data_type=data_type)

        # DPXOptimization parameterization
        self.optimization_class: Type[DPXOptimization] = optimization_class
        self.optimization_config: namedtuple = make_config(configuration_object=self,
                                                           configuration_name='optimization_config',
                                                           loss=loss,
                                                           lr=lr,
                                                           optimizer=optimizer)
        self.training_stuff: bool = (loss is not None) and (optimizer is not None) or (not require_training_stuff)

        # NetworkManager parameterization
        self.data_transformation_class: Type[DPXTransformation] = data_transformation_class
        self.data_transformation_config: namedtuple = make_config(configuration_object=self,
                                                                  configuration_name='data_transformation_config')

        # NetworkManager parameterization
        self.network_dir: str = network_dir
        self.which_network: int = which_network
        self.save_every_epoch: int = save_intermediate_state_every and self.training_stuff

    def create_network(self) -> DPXNetwork:
        """
        Create an instance of network_class with given parameters.

        :return: DPXNetwork object from network_class and its parameters.
        """

        # Create instance
        network = self.network_class(config=self.network_config)
        if not isinstance(network, DPXNetwork):
            raise TypeError(f"[{self.name}] The given 'network_class'={self.network_class} must be a DPXNetwork.")
        return network

    def create_optimization(self) -> DPXOptimization:
        """
        Create an instance of optimization_class with given parameters.

        :return: DPXOptimization object from optimization_class and its parameters.
        """

        # Create instance
        optimization = self.optimization_class(config=self.optimization_config)
        if not isinstance(optimization, DPXOptimization):
            raise TypeError(f"[{self.name}] The given 'optimization_class'={self.optimization_class} must be a "
                            f"DPXOptimization.")
        return optimization

    def create_data_transformation(self) -> DPXTransformation:
        """
        Create an instance of data_transformation_class with given parameters.

        :return: DPXTransformation object from data_transformation_class and its parameters.
        """

        # Create instance
        data_transformation = self.data_transformation_class(config=self.data_transformation_config)
        if not isinstance(data_transformation, DPXTransformation):
            raise TypeError(f"[{self.name}] The given 'data_transformation_class'={self.data_transformation_class} "
                            f"must be a DPXTransformation.")
        return data_transformation

    def __str__(self):

        description = "\n"
        description += f"{self.__class__.__name__}\n"
        description += f"    networks class: {self.network_class.__name__}\n"
        description += f"    Optimization class: {self.optimization_class.__name__}\n"
        description += f"    Training materials: {self.training_stuff}\n"
        description += f"    networks directory: {self.network_dir}\n"
        description += f"    Which networks: {self.which_network}\n"
        description += f"    Save each epoch: {self.save_each_epoch}\n"
        return description
