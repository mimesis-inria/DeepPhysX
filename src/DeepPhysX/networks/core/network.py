from typing import Any, Dict
from os import cpu_count
from numpy import ndarray
import torch
from torch import Tensor, device, set_num_threads, load, save, as_tensor
from torch.nn import Module
from torch.cuda import is_available, empty_cache
from gc import collect as gc_collect
from collections import namedtuple


class Network(Module):

    def __init__(self, config: namedtuple):
        """
        DPXNetwork computes predictions from input data according to actual set of weights.

        :param config: Set of DPXNetwork parameters.
        """

        Module.__init__(self)

        # Config
        self.device = None
        self.config = config

        # Data fields
        self.net_fields = ['input']
        self.opt_fields = ['ground_truth']
        self.pred_fields = ['prediction']
        self.pred_norm_fields = {'prediction': 'ground_truth'}

        # Data type
        if config.data_type == 'float64':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

    def predict(self,
                data_net: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute a forward pass of the networks.

        :param data_net: Data used by the networks.
        :return: Data produced by the networks.
        """

        return {'prediction': self.forward(data_net['input'])}

    def forward(self,
                input_data: Any) -> Any:
        """
        Compute a forward pass of the networks.

        :param input_data: Input tensor.
        :return: networks prediction.
        """

        raise NotImplementedError

    def set_train(self) -> None:
        """
        Set the networks in training mode (compute gradient).
        """

        self.train()

    def set_eval(self) -> None:
        """
        Set the networks in prediction mode (does not compute gradient).
        """

        self.eval()

    def set_device(self) -> None:
        """
        Set computer device on which networks's parameters will be stored and tensors will be computed.
        """

        if is_available():
            self.device = device('cuda')
            # Garbage collector run
            gc_collect()
            empty_cache()
        else:
            self.device = device('cpu')
            set_num_threads(cpu_count() - 1)
        self.to(self.device)
        print(f"[{self.__class__.__name__}]: Device is {self.device}")

    def load_parameters(self,
                        path: str) -> None:
        """
        Load networks parameter from path.

        :param path: Path to networks parameters to load.
        """

        self.load_state_dict(load(path, map_location=self.device))

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the current state of networks parameters.

        :return: networks parameters.
        """

        raise self.state_dict()

    def save_parameters(self,
                        path: str) -> None:
        """
        Saves the networks parameters to the path location.

        :param path: Path where to save the parameters.
        """

        path = path + '.pth'
        save(self.state_dict(), path)

    def nb_parameters(self) -> int:
        """
        Return the number of parameters of the networks.

        :return: Number of parameters.
        """

        return sum(p.numel() for p in self.parameters())

    def numpy_to_tensor(self,
                        data: ndarray,
                        grad: bool = True) -> Any:
        """
        Transform and cast data from numpy to the desired tensor type.

        :param data: Array data to convert.
        :param grad: If True, gradient will record operations on this tensor.
        :return: Converted tensor.
        """

        data = as_tensor(data.astype(self.config.data_type), device=self.device)
        if grad:
            data.requires_grad_()
        return data

    def tensor_to_numpy(self,
                        data: Any) -> ndarray:
        """
        Transform and cast data from tensor type to numpy.

        :param data: Tensor to convert.
        :return: Converted array.
        """

        return data.cpu().detach().numpy().astype(self.config.data_type)

    @staticmethod
    def print_architecture(architecture) -> str:
        """
        Format the networks architecture string description.

        :return: String containing the networks architecture description.
        """

        lines = architecture.splitlines()
        architecture = ''
        for line in lines:
            architecture += '\n      ' + line
        return architecture

    def __str__(self) -> str:

        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Name: {self.config.network_name}\n"
        description += f"    Type: {self.config.network_type}\n"
        description += f"    Number of parameters: {self.nb_parameters()}\n"
        description += f"    Estimated size: {self.nb_parameters() * 32 * 1.25e-10} Go\n"
        description += f"    Device: {self.device}\n"
        return description
