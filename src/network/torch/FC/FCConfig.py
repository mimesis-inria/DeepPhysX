from typing import Any, Optional, Type, Union, List

from DeepPhysX.Core.Utils.configs import make_config
from DeepPhysX.Torch.Network.TorchNetworkConfig import TorchNetworkConfig, TorchTransformation, TorchOptimization
from DeepPhysX.Torch.FC.FC import FC


class FCConfig(TorchNetworkConfig):

    def __init__(self,
                 optimization_class: Type[TorchOptimization] = TorchOptimization,
                 data_transformation_class: Type[TorchTransformation] = TorchTransformation,
                 network_dir: Optional[str] = None,
                 network_name: str = "FCNetwork",
                 which_network: int = 0,
                 save_each_epoch: bool = False,
                 data_type: str = 'float32',
                 lr: Optional[float] = None,
                 require_training_stuff: bool = True,
                 loss: Any = None,
                 optimizer: Any = None,
                 dim_output: int = 0,
                 dim_layers: list = None,
                 biases: Union[List[bool], bool] = True):
        """
        FCConfig is a configuration class to parameterize and create FC, TorchOptimization and TorchDataTransformation
        for the NetworkManager.

        :param optimization_class: TorchOptimization class from which an instance will be created.
        :param data_transformation_class: DataTransformation class from which an instance will be created.
        :param network_dir: Path to an existing network repository.
        :param network_name: Name of the network
        :param  which_network: If several networks in network_dir, load the specified one.
        :param save_each_epoch: If True, network state will be saved at each epoch end; if False, network state
                                will be saved at the end of the training
        :param data_type: Type of the training data.
        :param lr: Learning rate.
        :param require_training_stuff: If specified, loss and optimizer class can be not necessary for training.
        :param loss: Loss class.
        :param optimizer: Network's parameters optimizer class.
        :param dim_output: Dimension of the output.
        :param dim_layers: Size of each layer of the network.
        :param biases: Layers should have biases or not. This value can either be given as a bool for all layers or as
                       a list to detail each layer.
        """

        TorchNetworkConfig.__init__(self,
                                    network_class=FC,
                                    optimization_class=optimization_class,
                                    data_transformation_class=data_transformation_class,
                                    network_dir=network_dir,
                                    network_name=network_name,
                                    network_type='FC',
                                    which_network=which_network,
                                    save_each_epoch=save_each_epoch,
                                    data_type=data_type,
                                    require_training_stuff=require_training_stuff,
                                    lr=lr,
                                    loss=loss,
                                    optimizer=optimizer)

        # Check FC variables
        if dim_output is not None and type(dim_output) != int:
            raise TypeError(f"[{self.__class__.__name__}] Wrong 'dim_output' type: int required, get "
                            f"{type(dim_output)}")
        dim_layers = dim_layers if dim_layers else []
        if type(dim_layers) != list:
            raise TypeError(f"[{self.__class__.__name__}] Wrong 'dim_layers' type: list required, get "
                            f"{type(dim_layers)}")

        self.network_config = make_config(configuration_object=self,
                                          configuration_name='network_config',
                                          network_type='FC',
                                          dim_output=dim_output,
                                          dim_layers=dim_layers,
                                          biases=biases)
