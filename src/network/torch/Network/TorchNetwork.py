from typing import Dict
from os import cpu_count

import torch
from numpy import ndarray
from torch import Tensor, device, set_num_threads, load, save, as_tensor
from torch.nn import Module
from torch.cuda import is_available, empty_cache
from gc import collect as gc_collect
from collections import namedtuple

from DeepPhysX.Core.Network.BaseNetwork import BaseNetwork


class TorchNetwork(Module, BaseNetwork):

    def __init__(self,
                 config: namedtuple):
        """
        TorchNetwork computes predictions from input data according to actual set of weights.

        :param config: Set of TorchNetwork parameters.
        """

        Module.__init__(self)
        BaseNetwork.__init__(self, config)

        # Data type
        if config.data_type == 'float64':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        # Data fields
        self.net_fields = ['input']
        self.opt_fields = ['ground_truth']
        self.pred_fields = ['prediction']

    def forward(self,
                input_data: Tensor) -> Tensor:
        """
        Compute a forward pass of the Network.

        :param input_data: Input tensor.
        :return: Network prediction.
        """

        raise NotImplementedError

    def set_train(self) -> None:
        """
        Set the Network in train mode (compute gradient).
        """

        self.train()

    def set_eval(self) -> None:
        """
         Set the Network in eval mode (does not compute gradient).
         """

        self.eval()

    def set_device(self) -> None:
        """
        Set computer device on which Network's parameters will be stored and tensors will be computed.
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
        Load network parameter from path.

        :param path: Path to Network parameters to load.
        """

        self.load_state_dict(load(path, map_location=self.device))

    def get_parameters(self) -> Dict[str, Tensor]:
        """
        Return the current state of Network parameters.

        :return: Network parameters.
        """

        return self.state_dict()

    def save_parameters(self,
                        path: str) -> None:
        """
        Saves the network parameters to the path location.

        :param path: Path where to save the parameters.
        """

        path = path + '.pth'
        save(self.state_dict(), path)

    def nb_parameters(self) -> int:
        """
        Return the number of parameters of the network.

        :return: Number of parameters.
        """

        return sum(p.numel() for p in self.parameters())

    def numpy_to_tensor(self,
                        data: ndarray,
                        grad: bool = True) -> Tensor:
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
                        data: Tensor) -> ndarray:
        """
        Transform and cast data from tensor type to numpy.

        :param data: Tensor to convert.
        :return: Converted array.
        """

        return data.cpu().detach().numpy().astype(self.config.data_type)

    @staticmethod
    def print_architecture(architecture) -> str:
        """
        Format the network architecture string description.

        :return: String containing the network architecture description.
        """

        lines = architecture.splitlines()
        architecture = ''
        for line in lines:
            architecture += '\n      ' + line
        return architecture

    def __str__(self):

        description = BaseNetwork.__str__(self)
        description += f"    Device: {self.device}\n"
        return description
