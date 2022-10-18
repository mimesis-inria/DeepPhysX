from typing import Dict, Any
from collections import namedtuple

from DeepPhysX.Core.Network.BaseNetwork import BaseNetwork


class BaseOptimization:

    def __init__(self, config: namedtuple):
        """
        BaseOptimization is dedicated to network optimization: compute loss between prediction and target, update
        network parameters.

        :param config: Namedtuple containing BaseOptimization parameters
        """

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
        Initialize the loss function.
        """

        raise NotImplementedError

    def compute_loss(self,
                     data_pred: Dict[str, Any],
                     data_opt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute loss from prediction / ground truth.

        :param data_pred: Tensor produced by the forward pass of the Network.
        :param data_opt: Ground truth tensor to be compared with prediction.
        :return: Loss value.
        """

        # WARNING: self.optimization.compute_loss(data_pred.reshape(data_gt.shape), data_gt, loss_data)
        raise NotImplementedError

    def transform_loss(self,
                       data_opt: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply a transformation on the loss value using the potential additional data.

        :param data_opt: Additional data sent as dict to compute loss value
        :return: Transformed loss value
        """

        raise NotImplementedError

    def set_optimizer(self, net: BaseNetwork) -> None:
        """
        Define an optimization process.

        :param net: Network whose parameters will be optimized.
        """

        raise NotImplementedError

    def optimize(self) -> None:
        """
        Run an optimization step.
        """

        raise NotImplementedError

    def __str__(self) -> str:

        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Loss class: {self.loss_class.__name__}\n" if self.loss_class else f"    Loss class: None\n"
        description += f"    Optimizer class: {self.optimizer_class.__name__}\n" if self.optimizer_class else \
            f"    Optimizer class: None\n"
        description += f"    Learning rate: {self.lr}\n"
        return description
