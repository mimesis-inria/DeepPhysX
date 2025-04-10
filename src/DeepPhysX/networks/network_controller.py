from typing import Dict, Any, Type
from os import cpu_count
from numpy import ndarray
from torch import device, set_num_threads, load, save, as_tensor, dtype, Tensor
from torch.nn import Module
from torch.cuda import is_available, empty_cache
from gc import collect as gc_collect


class NetworkController:

    def __init__(self,
                 network_architecture: Type[Module],
                 network_kwargs: Dict[str, Any],
                 data_type: dtype):
        """
        NetworkController allows components to interact with the neural network architecture.

        :param network_architecture: The neural network architecture.
        :param network_kwargs: Dict of kwargs to create an instance of the neural network architecture.
        :param data_type: Set the Torch default data type.
        """

        self.__network: Module = network_architecture(**network_kwargs)
        self.__device = None
        self.data_type: dtype = data_type
        self.__is_ready: bool = False
        self.__is_training: bool = False

    def predict(self, *args):
        """
        Call the forward function of the network.

        :param args: Data fields to fill the forward function.
        """

        return self.__network.forward(*args)

    @property
    def is_ready(self):
        return self.__is_ready

    @property
    def is_training(self):
        return self.__is_training

    def train(self) -> None:
        """
        Set the network in training mode.
        """

        self.__is_ready = True
        self.__is_training = True
        self.__network.train()

    def eval(self) -> None:
        """
        Set the network in prediction mode.
        """

        self.__is_ready = True
        self.__network.eval()

    def set_device(self) -> None:
        """
        Define the network and tensors device - cpu or gpu.
        """

        # Set to GPU if available
        if is_available():
            self.__device = device('cuda')
            gc_collect()
            empty_cache()

        # Set the CPU
        else:
            self.__device = device('cpu')
            set_num_threads(cpu_count() - 1)

        self.__network.to(self.__device)
        print(f"[Network] Device is {self.__device}")

    def load(self, path: str) -> None:
        """
        Load a state of parameters.

        :param path: File containing the parameters of the network.
        """

        self.__network.load_state_dict(load(f=path, map_location=self.__device, weights_only=True))

    def parameters(self):
        """
        Get the parameters of the network.
        """

        return self.__network.parameters()

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dict of the network.
        """

        return self.__network.state_dict()

    def nb_parameters(self) -> int:
        """
        Get the number of parameters in the network.
        """

        return sum(p.numel() for p in self.__network.parameters())

    def save(self, path: str) -> None:
        """
        Save the current state of the network.

        :param path: File to store the parameters.
        """

        save(obj=self.__network.state_dict(), f=f'{path}.pth')

    def to_torch(self,
                 tensor: ndarray,
                 grad: bool = True) -> Tensor:
        """
        Convert numpy arrays to torch tensors.

        :param tensor: Numpy array to convert.
        :param grad: If True, record operations gradients.
        """

        tensor = as_tensor(tensor, dtype=self.data_type, device=self.__device)
        tensor.requires_grad_(grad)
        return tensor

    def to_numpy(self, tensor: Tensor) -> ndarray:
        """
        Convert torch tensors to numpy arrays.

        :param tensor: Torch tensor to convert.
        """

        return tensor.cpu().detach().numpy()
        