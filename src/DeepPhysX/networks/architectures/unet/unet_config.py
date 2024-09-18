from typing import Any, List, Optional, Type

from DeepPhysX.utils.configs import make_config
from DeepPhysX.networks.core.dpx_network_config import DPXNetworkConfig
from DeepPhysX.networks.core.dpx_optimization import DPXOptimization
from DeepPhysX.networks.architectures.unet.unet_transformation import UNetTransformation
from DeepPhysX.networks.architectures.unet.unet_layers import UNet


class UNetConfig(DPXNetworkConfig):

    def __init__(self,
                 optimization_class: Type[DPXOptimization] = DPXOptimization,
                 network_dir: Optional[str] = None,
                 network_name: str = "UNetNetwork",
                 which_network: int = 0,
                 save_intermediate_state_every: bool = False,
                 data_type: str = 'float32',
                 lr: Optional[float] = None,
                 require_training_stuff: bool = True,
                 loss: Optional[Any] = None,
                 optimizer: Optional[Any] = None,
                 input_size: List[int] = None,
                 nb_dims: int = 3,
                 nb_input_channels: int = 1,
                 nb_first_layer_channels: int = 64,
                 nb_output_channels: int = 3,
                 nb_steps: int = 3,
                 two_sublayers: bool = True,
                 border_mode: str = 'valid',
                 skip_merge: bool = False,
                 data_scale: float = 1.):
        """
        UNetConfig is a configuration class to parameterize and create unet, DPXOptimization and UNetTransformation
        for the NetworkManager.

        :param optimization_class: BaseOptimization class from which an instance will be created.
        :param network_dir: Path to an existing networks repository.
        :param network_name: Name of the networks.
        :param which_network: If several networks in network_dir, load the specified one.
        :param save_each_epoch: If True, networks state will be saved at each epoch end; if False, networks state
                                will be saved at the end of the training.
        :param data_type: Type of the training data.
        :param lr: Learning rate.
        :param require_training_stuff: If specified, loss and optimizer class can be not necessary for training.
        :param loss: Loss class.
        :param optimizer: Network's parameters optimizer class.

        :param input_size: Size of the input.
        :param nb_dims: Number of dimension of data.
        :param nb_input_channels: Number of channels of the input layer.
        :param nb_first_layer_channels: Number of channels of the first layer.
        :param nb_output_channels: Number of channels of the output layer.
        :param nb_steps: Number of steps of down layers / up layers.
        :param two_sublayers: Duplicate each layer or not.
        :param border_mode: Zero-padding mode.
        :param skip_merge: Skip the crop step at each up layer or not.
        :param data_scale: Scale to apply to data.
        """

        super().__init__(network_class=UNet,
                         optimization_class=optimization_class,
                         data_transformation_class=UNetTransformation,
                         network_dir=network_dir,
                         network_name=network_name,
                         network_type='unet',
                         which_network=which_network,
                         save_intermediate_state_every=save_intermediate_state_every,
                         data_type=data_type,
                         lr=lr,
                         require_training_stuff=require_training_stuff,
                         loss=loss,
                         optimizer=optimizer)

        name = self.__class__.__name__
        # Check the input size type
        input_size = input_size if input_size else [0, 0, 0]
        if type(input_size) not in [list, tuple]:
            raise TypeError(f"[{name}] Wrong 'input_size' type: list or tuple required, get {type(input_size)}")
        # Check the number of dimensions type and value
        if type(nb_dims) != int:
            raise TypeError(f"[{name}] Wrong 'nb_dims' type: int required, get {type(nb_dims)}")
        if nb_dims not in [2, 3]:
            raise ValueError(f"[{name}] unet works either with dimension 2 or 3, get {nb_dims}")
        # Check the number of channels type and value, check the nb_of steps type and value
        for nb_channel, arg_name in zip([nb_input_channels, nb_first_layer_channels, nb_output_channels, nb_steps],
                                        ['nb_input_channels', 'nb_first_layer_channels', 'nb_output_channels',
                                         'nb_steps']):
            if type(nb_channel) != int:
                raise TypeError(f"[{name}] Wrong '{arg_name}' type: int required, get {type(nb_channel)}")
            if nb_channel < 1:
                raise ValueError(f"[{name}] '{arg_name} must be positive")
        # Check the boolean values
        for flag, flag_name in zip([two_sublayers, skip_merge], ['two_sublayers', 'skip_merge']):
            if type(flag) != bool:
                raise TypeError(f"[{name}] Wrong '{flag_name}' type: bool required, get {type(flag)}")
        # Check border mode type and value
        if type(border_mode) != str:
            raise TypeError(f"[{name}] Wrong 'border_mode' type: str required, get {type(border_mode)}")
        if border_mode not in ['valid', 'same']:
            raise ValueError(f"[{name}] 'border_mode' must be in ['valid', 'same'], get {border_mode}")
        # Check data scale type and value
        if type(data_scale) != float:
            raise TypeError(f"[{name}] Wrong 'data_scale' type: float required, get {type(data_scale)}")

        # Define specific unet configuration
        self.network_config = make_config(configuration_object=self,
                                          configuration_name='network_config',
                                          network_name=network_name,
                                          network_type='unet',
                                          data_type=data_type,
                                          nb_dims=nb_dims,
                                          nb_input_channels=nb_input_channels,
                                          nb_first_layer_channels=nb_first_layer_channels,
                                          nb_output_channels=nb_output_channels,
                                          nb_steps=nb_steps,
                                          two_sublayers=two_sublayers,
                                          border_mode=border_mode,
                                          skip_merge=skip_merge)

        # Define specific UNetDataTransformation config
        self.data_transformation_config = make_config(configuration_object=self,
                                                      configuration_name='data_transformation_config',
                                                      input_size=input_size,
                                                      nb_input_channels=nb_input_channels,
                                                      nb_output_channels=nb_output_channels,
                                                      nb_steps=nb_steps,
                                                      two_sublayers=two_sublayers,
                                                      border_mode=border_mode,
                                                      data_scale=data_scale)
