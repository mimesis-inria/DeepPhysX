from typing import Any, Optional, Type
from os.path import isdir

from DeepPhysX.networks.network import Network
from DeepPhysX.networks.network_optimization import NetworkOptimization
from DeepPhysX.networks.network_transformation import NetworkTransformation
from DeepPhysX.utils.configs import make_config, namedtuple


class NetworkConfig:

    def __init__(self,
                 network_class: Type[Network] = Network,
                 optimization_class: Type[NetworkOptimization] = NetworkOptimization,
                 data_transformation_class: Type[NetworkTransformation] = NetworkTransformation,
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
        :param optimizer: Network's parameters optimizer class.
        """

        self.name = self.__class__.__name__

        # Check network_dir type and existence
        if network_dir is not None:
            if not isdir(network_dir):
                raise ValueError(f"[{self.__class__.__name__}] Given 'network_dir' does not exists: {network_dir}")

        # DPXNetwork parameterization
        self.network_class: Type[Network] = network_class
        self.network_config: namedtuple = make_config(configuration_object=self,
                                                      configuration_name='network_config',
                                                      network_name=network_name,
                                                      network_type=network_type,
                                                      data_type=data_type)

        # DPXOptimization parameterization
        self.optimization_class: Type[NetworkOptimization] = optimization_class
        self.optimization_config: namedtuple = make_config(configuration_object=self,
                                                           configuration_name='optimization_config',
                                                           loss=loss,
                                                           lr=lr,
                                                           optimizer=optimizer)
        self.training_stuff: bool = (loss is not None) and (optimizer is not None) or (not require_training_stuff)

        # NetworkManager parameterization
        self.data_transformation_class: Type[NetworkTransformation] = data_transformation_class
        self.data_transformation_config: namedtuple = make_config(configuration_object=self,
                                                                  configuration_name='data_transformation_config')

        # NetworkManager parameterization
        self.network_dir: str = network_dir
        self.which_network: int = which_network
        self.save_every_epoch: int = save_intermediate_state_every and self.training_stuff

    def create_network(self) -> Network:
        """
        Create an instance of network_class with given parameters.

        :return: DPXNetwork object from network_class and its parameters.
        """

        # Create instance
        network = self.network_class(config=self.network_config)
        if not isinstance(network, Network):
            raise TypeError(f"[{self.name}] The given 'network_class'={self.network_class} must be a DPXNetwork.")
        return network

    def create_optimization(self) -> NetworkOptimization:
        """
        Create an instance of optimization_class with given parameters.

        :return: DPXOptimization object from optimization_class and its parameters.
        """

        # Create instance
        optimization = self.optimization_class(config=self.optimization_config)
        if not isinstance(optimization, NetworkOptimization):
            raise TypeError(f"[{self.name}] The given 'optimization_class'={self.optimization_class} must be a "
                            f"DPXOptimization.")
        return optimization

    def create_data_transformation(self) -> NetworkTransformation:
        """
        Create an instance of data_transformation_class with given parameters.

        :return: DPXTransformation object from data_transformation_class and its parameters.
        """

        # Create instance
        data_transformation = self.data_transformation_class(config=self.data_transformation_config)
        if not isinstance(data_transformation, NetworkTransformation):
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
