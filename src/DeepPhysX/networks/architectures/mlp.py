from typing import List, Union, Optional, Tuple
from torch import Tensor
from torch.nn import Module, Sequential, PReLU, Linear


class MLP(Module):

    def __init__(self,
                 dim_layers: List[int],
                 out_shape: Optional[Tuple[int]] = None,
                 use_biases: Union[List[bool], bool] = True):
        """
        Create a Fully Connected layers Neural Network Architecture.
        """

        Module.__init__(self)

        self.dim_layers = dim_layers

        # Data fields
        # self.net_fields = ['input']
        # self.opt_fields = ['ground_truth']
        # self.pred_fields = ['prediction']

        # Convert biases to a List if not already one
        if isinstance(use_biases, list):
            if len(use_biases) != len(self.dim_layers) - 1:
                raise ValueError("Biases list length does not match layers count")
            self.biases = use_biases
        else:
            self.biases = [use_biases] * (len(self.dim_layers) - 1)

        # Init the layers
        self.layers = []
        for i, bias in enumerate(self.biases):
            self.layers.append(Linear(self.dim_layers[i], self.dim_layers[i + 1], bias))
            self.layers.append(PReLU(num_parameters=self.dim_layers[i + 1]))
        self.layers = self.layers[:-1]
        self.linear = Sequential(*self.layers)
        self.out_shape = (-1,) if out_shape is None else out_shape

    def forward(self,
                input_data: Tensor) -> Tensor:
        """
        Compute a forward pass of the Network.

        :param input_data: Input tensor.
        :return: Network prediction.
        """

        flat_input_data = input_data.view(input_data.shape[0], -1)
        res = self.linear(flat_input_data)
        return res.view(input_data.shape[0], *self.out_shape)

    # def __str__(self):
    #
    #     description = Network.__str__(self)
    #     description += self.linear.__str__() + "\n"
    #     description += f"    Layers dimensions: {self.dim_layers}\n"
    #     description += f"    Output dimension: {self.out_shape}\n"
    #     return description
