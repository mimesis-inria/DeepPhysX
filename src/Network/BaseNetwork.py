from typing import Any, Dict
from numpy import ndarray
from collections import namedtuple


class BaseNetwork:

    def __init__(self,
                 config: namedtuple):
        """
        BaseNetwork is a network class to compute predictions from input data according to actual state.

        :param namedtuple config: namedtuple containing BaseNetwork parameters
        """

        # Config
        self.device = None
        self.config = config

        # Data fields
        self.net_fields = ['input']
        self.opt_fields = ['ground_truth']
        self.pred_fields = ['prediction']

    def predict(self, data_net: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute a forward pass of the network.

        :param data_net: Data used by the Network.
        :return: Data produced by the Network.
        """

        return {'prediction': self.forward(data_net['input'])}

    def forward(self, input_data: Any) -> Any:
        """
        | Gives input_data as raw input to the neural network.

        :param Any input_data: Input tensor
        :return: Network prediction
        """

        raise NotImplementedError

    def set_train(self) -> None:
        """
        | Set the Network in train mode (compute gradient).
        """

        raise NotImplementedError

    def set_eval(self) -> None:
        """
        | Set the Network in eval mode (does not compute gradient).
        """

        raise NotImplementedError

    def set_device(self) -> None:
        """
        | Set computer device on which Network's parameters will be stored and tensors will be computed.
        """

        raise NotImplementedError

    def load_parameters(self, path: str) -> None:
        """
        | Load network parameter from path.

        :param str path: Path to Network parameters to load
        """

        raise NotImplementedError

    def get_parameters(self) -> Dict[str, Any]:
        """
        | Return the current state of Network parameters.

        :return: Network parameters
        """

        raise NotImplementedError

    def save_parameters(self, path) -> None:
        """
        | Saves the network parameters to the path location.

        :param str path: Path where to save the parameters.
        """

        raise NotImplementedError

    def nb_parameters(self) -> int:
        """
        | Return the number of parameters of the network.

        :return: Number of parameters
        """

        raise NotImplementedError

    def numpy_to_tensor(self, data: ndarray, grad: bool = True) -> Any:
        """
        | Transform and cast data from numpy to the desired tensor type.

        :param ndarray data: Array data to convert
        :param bool grad: If True, gradient will record operations on this tensor
        :return: Converted tensor
        """

        return data.astype(self.config.data_type)

    def tensor_to_numpy(self, data: Any) -> ndarray:
        """
        | Transform and cast data from tensor type to numpy.

        :param Any data: Any to convert
        :return: Converted array
        """

        return data.astype(self.config.data_type)

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseNetwork object
        """

        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Name: {self.config.network_name}\n"
        description += f"    Type: {self.config.network_type}\n"
        description += f"    Number of parameters: {self.nb_parameters()}\n"
        description += f"    Estimated size: {self.nb_parameters() * 32 * 1.25e-10} Go\n"
        return description
