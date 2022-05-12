from typing import Any, Dict
from numpy import ndarray
from collections import namedtuple


class BaseNetwork:
    """
    | BaseNetwork is a network class to compute predictions from input data according to actual state.

    :param namedtuple config: namedtuple containing BaseNetwork parameters
    """

    def __init__(self, config: namedtuple):

        # Config
        self.device = None
        self.config = config

    def predict(self, input_data: Any) -> Any:
        """
        | Same as forward

        :param Any input_data: Input tensor
        :return: Network prediction
        """

        return self.forward(input_data)

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

    def transform_from_numpy(self, data: ndarray, grad: bool = True) -> Any:
        """
        | Transform and cast data from numpy to the desired tensor type.

        :param ndarray data: Array data to convert
        :param bool grad: If True, gradient will record operations on this tensor
        :return: Converted tensor
        """

        raise NotImplementedError

    def transform_to_numpy(self, data: Any) -> ndarray:
        """
        | Transform and cast data from tensor type to numpy.

        :param Any data: Any to convert
        :return: Converted array
        """

        raise NotImplementedError

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
