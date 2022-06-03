from typing import Dict, Any
from collections import namedtuple

from DeepPhysX.Core.Network.BaseNetwork import BaseNetwork


class BaseOptimization:
    """
    | BaseOptimization is dedicated to network optimization: compute loss between prediction and target, update
      network parameters.

    :param namedtuple config: Namedtuple containing BaseOptimization parameters
    """

    def __init__(self, config: namedtuple):

        self.manager: Any = None

        # Loss
        self.loss_class = config.loss
        self.loss = None
        self.loss_value = 0.

        # Optimizer
        self.optimizer_class = config.optimizer
        self.optimizer = None
        self.lr = config.lr

    def set_loss(self) -> None:
        """
        | Initialize the loss function.
        """

        raise NotImplementedError

    def compute_loss(self, prediction: Any, ground_truth: Any, data: Dict[str, Any]) -> Dict[str, float]:
        """
        | Compute loss from prediction / ground truth.

        :param Any prediction: Tensor produced by the forward pass of the Network
        :param Any ground_truth: Ground truth tensor to be compared with prediction
        :param Dict[str, Any] data: Additional data sent as dict to compute loss value
        :return: Loss value
        """

        raise NotImplementedError

    def transform_loss(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        | Apply a transformation on the loss value using the potential additional data.

        :param Dict[str, Any] data: Additional data sent as dict to compute loss value
        :return: Transformed loss value
        """

        raise NotImplementedError

    def set_optimizer(self, net: BaseNetwork) -> None:
        """
        | Define an optimization process.

        :param BaseNetwork net: Network whose parameters will be optimized.
        """

        raise NotImplementedError

    def optimize(self) -> None:
        """
        | Run an optimization step.
        """

        raise NotImplementedError

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseOptimization object
        """

        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Loss class: {self.loss_class.__name__}\n" if self.loss_class else f"    Loss class: None\n"
        description += f"    Optimizer class: {self.optimizer_class.__name__}\n" if self.optimizer_class else \
            f"    Optimizer class: None\n"
        description += f"    Learning rate: {self.lr}\n"
        return description
