from typing import List, Optional, Tuple, Dict
from torch.nn.functional import pad
from torch import reshape, Tensor
from numpy import asarray
from collections import namedtuple

from DeepPhysX.networks.core.dpx_transformation import DPXTransformation


class UNetTransformation(DPXTransformation):

    def __init__(self, config: namedtuple):
        """
        UNetDataTransformation manages data operations before and after unet predictions.

        :param config: Set of TorchTransformation parameters.
        """

        super().__init__(config)

        # Configure the transformation parameters
        self.input_size: List[int] = self.config.input_size
        self.nb_steps: int = self.config.nb_steps
        self.nb_output_channels: int = self.config.nb_output_channels
        self.nb_input_channels: int = self.config.nb_input_channels
        self.data_scale: float = self.config.data_scale
        self.pad_widths: Optional[List[int]] = None
        self.inverse_pad_widths: Optional[List[int]] = None

        # Define shape transformations
        border = 4 if self.config.two_sublayers else 2
        border = 0 if self.config.border_mode == 'same' else border
        self.reverse_first_step = lambda x: x + border
        self.reverse_down_step = lambda x: (x + border) * 2
        self.reverse_up_step = lambda x: (x + border - 1) // 2 + 1

    @DPXTransformation.check_type
    def transform_before_prediction(self, data_net: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply data operations before networks's prediction.

        :param data_net: Data used by the Network.
        :return: Transformed data_net.
        """

        # Transform tensor shape
        data_in = data_net['input']
        data_in = data_in.view((-1, self.input_size[2], self.input_size[1], self.input_size[0], self.nb_input_channels))
        data_in = data_in.permute((0, 4, 1, 2, 3))

        # Compute padding
        if self.pad_widths is None:
            transformed_shape = data_in[0].shape[1:]
            self.compute_pad_widths(transformed_shape)

        # Apply padding
        data_in = pad(data_in, self.pad_widths, mode='constant')
        return {'input': data_in}

    @DPXTransformation.check_type
    def transform_before_loss(self,
                              data_pred: Dict[str, Tensor],
                              data_opt: Dict[str, Optional[Tensor]] = None) -> Tuple[Dict[str, Tensor], Optional[Dict[str, Tensor]]]:
        """
        Apply data operations between networks's prediction and loss computation.

        :param data_pred: Data produced by the Network.
        :param data_opt: Data used by the Optimizer.
        :return: Transformed data_pred, data_opt.
        """

        # Transform ground truth
        # Transform tensor shape, apply scale

        if data_opt is not None:
            data_gt = data_opt['ground_truth']
            data_gt = reshape(data_gt,
                              (-1, self.input_size[2], self.input_size[1], self.input_size[0], self.nb_output_channels))
            data_gt = self.data_scale * data_gt
            data_opt['ground_truth'] = data_gt


        # Transform prediction
        # Apply inverse padding, permute
        data_out = data_pred['prediction']
        data_out = pad(data_out, self.inverse_pad_widths)
        data_out = data_out.permute(0, 2, 3, 4, 1)

        return {'prediction': data_out}, data_opt

    @DPXTransformation.check_type
    def transform_before_apply(self,
                               data_pred: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply data operations between loss computation and prediction apply in environment.

        :param data_pred: Data produced by the Network.
        :return: Transformed data_pred.
        """

        # Rescale prediction
        data_out = data_pred['prediction']
        data_out = data_out / self.data_scale
        return {'prediction': data_out}

    def compute_pad_widths(self,
                           desired_shape: List[int]) -> None:
        """
        Define padding to apply on data given the data shape and the networks architecture.

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

    def __str__(self):

        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Data type: {self.data_type}\n"
        description += f"    Data scale: {self.data_scale}\n"
        description += f"    Transformation before prediction: Input -> Reshape + Permute + Padding\n"
        description += f"    Transformation before loss: Ground Truth -> Reshape + Upscale\n"
        description += f"                                Prediction -> Inverse padding + Permute\n"
        description += f"    Transformation before apply: Prediction -> Downscale\n"
        return description
