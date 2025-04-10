from typing import List, Tuple, Union, Any, Optional
from torch import Tensor, zeros_like, cat
from torch.nn import Module, Conv2d, Conv3d, BatchNorm2d, BatchNorm3d, ReLU, Sequential, MaxPool2d, MaxPool3d, \
    ConvTranspose2d, ConvTranspose3d
from torch.nn.functional import pad
from numpy import asarray


class UNet(Module):

    def __init__(self,
                 input_size: List[int] = None,
                 nb_dims: int = 3,
                 nb_input_channels: int = 1,
                 nb_first_layer_channels: int = 64,
                 nb_output_channels: int = 3,
                 flatten_output: bool = False,
                 nb_steps: int = 3,
                 two_sublayers: bool = True,
                 border_mode: str = 'valid',
                 skip_merge: bool = False):
        """
        Create a UNet network architecture.

        :param input_size: Shape of the input tensor.
        :param nb_dims: Number of dimensions of data.
        :param nb_input_channels: Number of channels of the first layer.
        :param nb_first_layer_channels: Number of channels of the first layer.
        :param nb_output_channels: Number of channels of the final layer.
        :param nb_steps: Number of down / up layers.
        :param two_sublayers: If True, duplicate each layer.
        :param border_mode: Padding mode.
        :param skip_merge: If True, skip the crop step at each up layer.
        """

        Module.__init__(self)

        # Configure the unet layers
        up_convolution_layer: Union[
            ConvTranspose2d, ConvTranspose3d] = ConvTranspose2d if nb_dims == 2 else ConvTranspose3d
        last_convolution_layer: Union[Conv2d, Conv3d] = Conv2d if nb_dims == 2 else Conv3d
        self.max_pool: Union[MaxPool2d, MaxPool3d] = MaxPool2d(2) if nb_dims == 2 else MaxPool3d(2)
        self.skip_merge: bool = skip_merge
        channels: int = nb_first_layer_channels
        up_kernel_size: Tuple[int, ...] = (2,) * nb_dims
        final_kernel_size: Tuple[int, ...] = (1,) * nb_dims

        # Define down layers : sequence of UNetLayer
        down_layers: List[UNetLayer] = [UNetLayer(nb_input_channels=nb_input_channels,
                                                  nb_output_channels=channels,
                                                  nb_dims=nb_dims,
                                                  border_mode=border_mode,
                                                  two_sublayers=two_sublayers),
                                        *[UNetLayer(nb_input_channels=channels * 2 ** i,
                                                    nb_output_channels=channels * 2 ** (i + 1),
                                                    nb_dims=nb_dims,
                                                    border_mode=border_mode,
                                                    two_sublayers=two_sublayers)
                                          for i in range(nb_steps)]]

        # Define up layers: sequence of (UpConvolutionLayer, UNetLayer)
        up_layers: List[Union[ConvTranspose2d, ConvTranspose3d, UNetLayer]] = [
            *[Sequential(up_convolution_layer(in_channels=channels * 2 ** (i + 1),
                                              out_channels=channels * 2 ** i,
                                              kernel_size=up_kernel_size,
                                              stride=up_kernel_size),
                         UNetLayer(nb_input_channels=channels * 2 ** (i + 1),
                                   nb_output_channels=channels * 2 ** i,
                                   nb_dims=nb_dims,
                                   border_mode=border_mode,
                                   two_sublayers=two_sublayers))
              for i in range(nb_steps - 1, -1, -1)]]

        # Set encoder - decoder architectures
        layers = down_layers + up_layers
        architecture: EncoderDecoder = EncoderDecoder(layers=layers,
                                                      nb_encoding_layers=nb_steps + 1)

        # Set the parts of the unet
        self.down: Sequential = architecture.setup_encoder()
        self.up: Sequential = architecture.setup_decoder()
        self.finalLayer: Union[Conv2d, Conv3d] = last_convolution_layer(in_channels=channels,
                                                                        out_channels=nb_output_channels,
                                                                        kernel_size=final_kernel_size)

        # Data transform config
        self.input_size: List[int] = [int(s) for s in input_size]
        self.flatten_output: bool = flatten_output
        self.nb_steps: int = nb_steps
        self.nb_output_channels: int = nb_output_channels
        self.nb_input_channels: int = nb_input_channels
        self.pad_widths: Optional[List[int]] = None
        self.inverse_pad_widths: Optional[List[int]] = None

        # Define shape transformations
        border = 4 if two_sublayers else 2
        border = 0 if border_mode == 'same' else border
        self.reverse_first_step = lambda x: x + border
        self.reverse_down_step = lambda x: (x + border) * 2
        self.reverse_up_step = lambda x: (x + border - 1) // 2 + 1


    def forward(self, input_data: Tensor) -> Tensor:
        """
        Compute a forward pass of the Network.

        :param input_data: Input tensor.
        """

        # TRANSFORM BEFORE PREDICTION
        input_data = self.transform_before_prediction(data=input_data)

        # Process down layers. Keep the outputs at each 'down' step to merge at same 'up' level.
        down_outputs = [self.down[0](input_data)]
        for unet_layer in self.down[1:]:
            down_outputs.append(unet_layer(self.max_pool(down_outputs[-1])))

        # Process up layers. Merge same level 'down' outputs.
        x = down_outputs.pop()
        for (up_conv_layer, unet_layer), down_output in zip(self.up, down_outputs[::-1]):
            same_level_down_output = zeros_like(down_output) if self.skip_merge else down_output
            x = unet_layer(crop_and_merge(same_level_down_output, up_conv_layer(x)))

        pred = self.finalLayer(x)

        # TRANSFORMS AFTER PREDICTION
        pred = self.transform_after_prediction(data=pred)

        return pred

    def transform_before_prediction(self, data: Tensor) -> Tensor:
        """
        Transform operations to apply before the forward pass.
        """

        # Transform tensor shape
        data = data.view((-1, self.input_size[2], self.input_size[1], self.input_size[0], self.nb_input_channels))
        data = data.permute((0, 4, 1, 2, 3))

        # Compute padding
        if self.pad_widths is None:
            transformed_shape = data[0].shape[1:]
            self.compute_pad_widths(transformed_shape)

        # Apply padding
        data = pad(data, self.pad_widths, mode='constant')
        return data

    def transform_after_prediction(self, data: Tensor) -> Tensor:
        """
        Transform operations to apply before the backward pass.
        """

        data = pad(data, self.inverse_pad_widths)
        data = data.permute(0, 2, 3, 4, 1)
        if self.flatten_output:
            data = data.view((-1, self.input_size[0] *self.input_size[1] * self.input_size[2], self.nb_output_channels))
        else:
            data = data.view((-1, self.input_size[0], self.input_size[1], self.input_size[2], self.nb_output_channels))
        return data

    def compute_pad_widths(self, desired_shape: List[int]) -> None:
        """
        Define padding to apply on data given the data shape and the networks architectures.

        :param desired_shape: Data shape without padding.
        """

        # Compute minimal input shape given the desired shape
        minimal_shape = asarray(desired_shape)
        for i in range(self.nb_steps):
            minimal_shape = self.reverse_up_step(minimal_shape)
        for i in range(self.nb_steps):
            minimal_shape = self.reverse_down_step(minimal_shape)
        minimal_shape = tuple(self.reverse_first_step(minimal_shape))

        # Compute padding width between shapes
        pad_widths = [((m - d) // 2, (m - d - 1) // 2 + 1) for m, d in zip(minimal_shape, desired_shape)]
        pad_widths.reverse()    # PyTorch applies padding from last dimension
        self.pad_widths, self.inverse_pad_widths = (), ()
        for p in pad_widths:
            self.pad_widths += p
            self.inverse_pad_widths += (-p[0], -p[1])   # PyTorch accepts negative padding


class UNetLayer(Module):

    def __init__(self,
                 nb_input_channels: int,
                 nb_output_channels: int,
                 nb_dims: int,
                 border_mode: str,
                 two_sublayers: bool):
        """
        Create one UNet layer.

        :param nb_input_channels: Number of channels of the first layer.
        :param nb_output_channels: Number of channels of the final layer.
        :param nb_dims: Number of dimensions of data.
        :param border_mode: Padding mode.
        :param two_sublayers: If True, duplicate each layer.
        """

        super().__init__()

        # Configure the unet layer
        convolution_layer: Union[Conv2d, Conv3d] = Conv2d if nb_dims == 2 else Conv3d
        normalization_layer: Union[BatchNorm2d, BatchNorm3d] = BatchNorm2d if nb_dims == 2 else BatchNorm3d
        padding: int = 0 if border_mode == 'valid' else 1
        kernel_size: Tuple[int] = (3,) * nb_dims

        # Define the unet layer: sequence of convolution, normalization and relu
        layers: List[Union[Conv2d, Conv3d, BatchNorm2d, BatchNorm3d, ReLU]] = [
            convolution_layer(in_channels=nb_input_channels,
                              out_channels=nb_output_channels,
                              kernel_size=kernel_size,
                              padding=padding),
            normalization_layer(num_features=nb_output_channels),
            ReLU()]

        # Duplicate the unet layer
        if two_sublayers:
            layers = layers + [convolution_layer(in_channels=nb_output_channels,
                                                 out_channels=nb_output_channels,
                                                 kernel_size=kernel_size,
                                                 padding=padding),
                               normalization_layer(num_features=nb_output_channels),
                               ReLU()]

        # Set the unet layer
        self.unet_layer = Sequential(*layers)

    def forward(self, input_data: Tensor) -> Tensor:
        """
        Compute a forward pass of the layer.

        :param input_data: Input tensor.
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

    def setup_encoder(self) -> Sequential:
        return Sequential(*self.encoder)

    def setup_decoder(self) -> Sequential:
        return Sequential(*self.decoder)

    def execute_sequential(self) -> Sequential:
        return Sequential(*[*self.encoder, *self.decoder])


def crop_slices(shape1: List[int], shape2: List[int]) -> List[slice]:
    return [slice((s1 - s2) // 2, (s1 - s2) // 2 + s2) for s1, s2 in zip(shape1, shape2)]


def crop_and_merge(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    return cat((tensor1[slices], tensor2), 1)
