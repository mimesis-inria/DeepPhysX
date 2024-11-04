from typing import List, Tuple, Union, Any
from torch import Tensor, zeros_like
from torch.nn import Module, Conv2d, Conv3d, BatchNorm2d, BatchNorm3d, ReLU, Sequential, MaxPool2d, MaxPool3d, \
    ConvTranspose2d, ConvTranspose3d
from collections import namedtuple

from DeepPhysX.networks.core.network import Network
from DeepPhysX.networks.architectures.unet.utils import crop_and_merge


class UNetLayer(Module):

    def __init__(self,
                 nb_input_channels: int,
                 nb_output_channels: int,
                 config: namedtuple):
        """
        Create a unet Layer.

        :param nb_input_channels: Number of channels in input data.
        :param nb_output_channels: Number of channels in output data.
        :param config: Namedtuple containing unet parameters.
        """

        super().__init__()

        # Configure the unet layer
        convolution_layer: Union[Conv2d, Conv3d] = Conv2d if config.nb_dims == 2 else Conv3d
        normalization_layer: Union[BatchNorm2d, BatchNorm3d] = BatchNorm2d if config.nb_dims == 2 else BatchNorm3d
        padding: int = 0 if config.border_mode == 'valid' else 1
        kernel_size: Tuple[int, int, int] = (3,) * config.nb_dims

        # Define the unet layer: sequence of convolution, normalization and relu
        layers: List[Union[Conv2d, Conv3d, BatchNorm2d, BatchNorm3d, ReLU]] = [
            convolution_layer(in_channels=nb_input_channels,
                              out_channels=nb_output_channels,
                              kernel_size=kernel_size,
                              padding=padding),
            normalization_layer(num_features=nb_output_channels),
            ReLU()]

        # Duplicate the unet layer
        if config.two_sublayers:
            layers = layers + [convolution_layer(in_channels=nb_output_channels,
                                                 out_channels=nb_output_channels,
                                                 kernel_size=kernel_size,
                                                 padding=padding),
                               normalization_layer(num_features=nb_output_channels),
                               ReLU()]

        # Set the unet layer
        self.unet_layer = Sequential(*layers)

    def forward(self,
                input_data: Tensor) -> Tensor:
        """
        Compute a forward pass of the layer.

        :param input_data: Input tensor.
        :return: Forward pass result.
        """

        return self.unet_layer(input_data)


class EncoderDecoder:

    def __init__(self,
                 layers: Any = None,
                 nb_encoding_layers: int = 0,
                 nb_decoding_layers: int = 0):

        self.layers = layers
        self.nb_encoding_layers: int = nb_encoding_layers
        self.nb_decoding_layers: int = nb_decoding_layers if nb_decoding_layers > 0 else nb_encoding_layers

        # Rectangle networks (Each layer of the encoder and decoder has the same amount of neurones)
        if len(self.layers) == 1:
            self.encoder = layers * self.nb_encoding_layers
            self.decoder = layers * self.nb_decoding_layers

        # Symmetric Network {layers: [50, 10], nbEncodingLayers : 2, nbDecodingLayers: 2}
        # Network will have [50 - 10 - 10 - 50] neurones on each layers
        elif len(self.layers) == self.nb_encoding_layers == self.nb_decoding_layers:
            self.encoder = self.layers
            self.decoder = self.layers[::-1]

        # Asymmetric Network {layers: [50, 10, 11, 15, 73], nbEncodingLayers : 2, nbDecodingLayers: 3}
        # Network will have [50 - 10 - 11 - 15 - 73] neurones on each layers
        elif len(self.layers) > self.nb_encoding_layers:
            self.encoder = self.layers[:self.nb_encoding_layers]
            self.decoder = self.layers[self.nb_encoding_layers:]

        else:
            raise ValueError(f"[EncoderDecoder] Layers count {len(self.layers)} does not match any known pattern.")

    def setupEncoder(self) -> Sequential:
        return Sequential(*self.encoder)

    def setupDecoder(self) -> Sequential:
        return Sequential(*self.decoder)

    def executeSequential(self) -> Sequential:
        return Sequential(*[*self.encoder, *self.decoder])


class UNet(Network):

    def __init__(self,
                 config: namedtuple):
        """
        Create a unet Neural Network Architecture.

        :param config: Set of unet parameters.
        """

        super().__init__(config)

        # Configure the unet layers
        up_convolution_layer: Union[
            ConvTranspose2d, ConvTranspose3d] = ConvTranspose2d if config.nb_dims == 2 else ConvTranspose3d
        last_convolution_layer: Union[Conv2d, Conv3d] = Conv2d if config.nb_dims == 2 else Conv3d
        self.max_pool: Union[MaxPool2d, MaxPool3d] = MaxPool2d(2) if config.nb_dims == 2 else MaxPool3d(2)
        self.skip_merge: bool = config.skip_merge
        channels: int = config.nb_first_layer_channels
        up_kernel_size: Tuple[int, ...] = (2,) * config.nb_dims
        final_kernel_size: Tuple[int, ...] = (1,) * config.nb_dims

        # Define down layers : sequence of UNetLayer
        down_layers: List[UNetLayer] = [UNetLayer(nb_input_channels=config.nb_input_channels,
                                                  nb_output_channels=channels,
                                                  config=config),
                                        *[UNetLayer(nb_input_channels=channels * 2 ** i,
                                                    nb_output_channels=channels * 2 ** (i + 1),
                                                    config=config)
                                          for i in range(config.nb_steps)]]

        # Define up layers: sequence of (UpConvolutionLayer, UNetLayer)
        up_layers: List[Union[ConvTranspose2d, ConvTranspose3d, UNetLayer]] = [
            *[Sequential(up_convolution_layer(in_channels=channels * 2 ** (i + 1),
                                              out_channels=channels * 2 ** i,
                                              kernel_size=up_kernel_size,
                                              stride=up_kernel_size),
                         UNetLayer(channels * 2 ** (i + 1), channels * 2 ** i, config))
              for i in range(config.nb_steps - 1, -1, -1)]]

        # Set encoder - decoder architecture
        layers = down_layers + up_layers
        architecture: EncoderDecoder = EncoderDecoder(layers=layers,
                                                      nb_encoding_layers=config.nb_steps + 1)

        # Set the parts of the unet
        self.down: Sequential = architecture.setupEncoder()
        self.up: Sequential = architecture.setupDecoder()
        self.finalLayer: Union[Conv2d, Conv3d] = last_convolution_layer(in_channels=channels,
                                                                        out_channels=config.nb_output_channels,
                                                                        kernel_size=final_kernel_size)

    def forward(self,
                input_data: Tensor) -> Tensor:
        """
        Compute a forward pass of the Network.

        :param input_data: Input tensor.
        :return: Network prediction.
        """

        # Process down layers. Keep the outputs at each 'down' step to merge at same 'up' level.
        down_outputs = [self.down[0](input_data)]
        for unet_layer in self.down[1:]:
            down_outputs.append(unet_layer(self.max_pool(down_outputs[-1])))

        # Process up layers. Merge same level 'down' outputs.
        x = down_outputs.pop()
        for (up_conv_layer, unet_layer), down_output in zip(self.up, down_outputs[::-1]):
            same_level_down_output = zeros_like(down_output) if self.skip_merge else down_output
            x = unet_layer(crop_and_merge(same_level_down_output, up_conv_layer(x)))

        return self.finalLayer(x)

    def __str__(self) -> str:

        description = Network.__str__(self)
        description += f"    Number of dimensions: {self.config.nb_dims}\n"
        description += f"    Number of input channels: {self.config.nb_input_channels}\n"
        description += f"    Number of first layer channels: {self.config.nb_first_layer_channels}\n"
        description += f"    Number of output channels: {self.config.nb_output_channels}\n"
        description += f"    Number of encoding/decoding steps: {self.config.nb_steps}\n"
        description += f"    Two sublayers in a step: {self.config.two_sublayers}\n"
        description += f"    Border mode: {self.config.border_mode}\n"
        description += f"    Merge on same level: {not self.config.skip_merge}\n"
        description += f"    Down layers: {self.print_architecture(str(self.down))}\n"
        description += f"    Up layers: {self.print_architecture(str(self.up))}\n"
        description += f"    Final layer: {self.print_architecture(str(self.finalLayer))}"
        return description
