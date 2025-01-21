from typing import Dict, Any, Callable, Type
from os import cpu_count
from numpy import ndarray
from torch import device, set_num_threads, load, save, as_tensor
from torch.nn import Module
from torch.cuda import is_available, empty_cache
from gc import collect as gc_collect


class NetworkWrapper:

    def __init__(self,
                 network_architecture: Type[Module],
                 network_kwargs: Dict[str, Any],
                 data_type: str):

        self.__network = network_architecture(**network_kwargs)
        self.__device = None
        self.data_type = data_type
        # if config.data_type == 'float64':
        #     torch.set_default_dtype(torch.float64)
        # else:
        #     torch.set_default_dtype(torch.float32)

    def predict(self, *args):

        return self.__network.forward(*args)

    def train(self) -> None:

        self.__network.train()

    def eval(self) -> None:

        self.__network.eval()

    def set_device(self) -> None:

        if is_available():
            self.__device = device('cuda')
            gc_collect()
            empty_cache()

        else:
            self.__device = device('cpu')
            set_num_threads(cpu_count() - 1)

        self.__network.to(self.__device)
        print(f"[{self.__class__.__name__}] Device is {self.__device}")


    def load(self, path: str) -> None:

        self.__network.load_state_dict(load(f=path, map_location=self.__device, weights_only=True))

    def parameters(self):

        return self.__network.parameters()

    def state_dict(self) -> Dict[str, Any]:

        return self.__network.state_dict()

    def nb_parameters(self) -> int:

        return sum(p.numel() for p in self.__network.parameters())

    def save(self, path: str) -> None:

        save(obj=self.__network.state_dict(), f=f'{path}.pth')

    def to_torch(self, tensor: ndarray, grad: bool = True) -> Any:

        tensor = as_tensor(tensor.astype(self.data_type), device=self.__device)
        tensor.requires_grad_(grad)
        return tensor

    def to_numpy(self, tensor: Any) -> ndarray:

        return tensor.cpu().detach().numpy().astype(self.data_type)
        