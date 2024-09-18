from typing import Any, Optional, Type

from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig, BaseNetwork, BaseOptimization, BaseTransformation
from DeepPhysX.Torch.Network.TorchTransformation import TorchTransformation
from DeepPhysX.Torch.Network.TorchNetwork import TorchNetwork
from DeepPhysX.Torch.Network.TorchOptimization import TorchOptimization


class TorchNetworkConfig(BaseNetworkConfig):

    def __init__(self,
                 network_class: Type[TorchNetwork] = TorchNetwork,
                 optimization_class: Type[TorchOptimization] = TorchOptimization,
                 data_transformation_class: Type[TorchTransformation] = TorchTransformation,
                 network_dir: Optional[str] = None,
                 network_name: str = 'Network',
                 network_type: str = 'TorchNetwork',
                 which_network: int = 0,
                 save_each_epoch: bool = False,
                 data_type: str = 'float32',
                 lr: Optional[float] = None,
                 require_training_stuff: bool = True,
                 loss: Optional[Any] = None,
                 optimizer: Optional[Any] = None):
        """
        TorchNetworkConfig is a configuration class to parameterize and create TorchNetwork, TorchOptimization and
        TorchTransformation for the NetworkManager.

        :param network_class: TorchNetwork class from which an instance will be created.
        :param optimization_class: TorchOptimization class from which an instance will be created.
        :param data_transformation_class: TorchTransformation class from which an instance will be created.
        :param network_dir: Path to an existing network repository.
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

        BaseNetworkConfig.__init__(self,
                                   network_class=network_class,
                                   optimization_class=optimization_class,
                                   data_transformation_class=data_transformation_class,
                                   network_dir=network_dir,
                                   network_name=network_name,
                                   network_type=network_type,
                                   which_network=which_network,
                                   save_each_epoch=save_each_epoch,
                                   data_type=data_type,
                                   lr=lr,
                                   require_training_stuff=require_training_stuff,
                                   loss=loss,
                                   optimizer=optimizer)

    def create_network(self) -> BaseNetwork:
        """
        Create an instance of network_class with given parameters.

        :return: TorchNetwork object from network_class and its parameters.
        """

        return BaseNetworkConfig.create_network(self)

    def create_optimization(self) -> BaseOptimization:
        """
        Create an instance of optimization_class with given parameters.

        :return: TorchOptimization object from optimization_class and its parameters.
        """

        return BaseNetworkConfig.create_optimization(self)

    def create_data_transformation(self) -> BaseTransformation:
        """
        Create an instance of data_transformation_class with given parameters.

        :return: DataTransformation object from data_transformation_class and its parameters.
        """

        return BaseNetworkConfig.create_data_transformation(self)
