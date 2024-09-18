from typing import Dict, Any
from collections import namedtuple

from DeepPhysX.networks.core.dpx_network import DPXNetwork


class DPXOptimization:

    def __init__(self, config: namedtuple):
        """
        DPXOptimization computes loss between prediction and target and optimizes the networks parameters.

        :param config: Set of DPXOptimization parameters.
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

        if self.loss_class is not None:
            self.loss = self.loss_class()

    def compute_loss(self,
                     data_pred: Dict[str, Any],
                     data_opt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute loss from prediction / ground truth.

        :param data_pred: Tensor produced by the forward pass of the networks.
        :param data_opt: Ground truth tensor to be compared with prediction.
        :return: Loss value.
        """

        self.loss_value = self.loss(data_pred['prediction'].view(data_opt['ground_truth'].shape),
                                    data_opt['ground_truth'])
        return self.transform_loss(data_opt)

    def transform_loss(self,
                       data_opt: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply a transformation on the loss value using the potential additional data.

        :param data_opt: Additional data sent as dict to compute loss value
        :return: Transformed loss value.
        """

        return {'loss': self.loss_value.item()}

    def set_optimizer(self,
                      net: DPXNetwork) -> None:
        """
        Define an optimization process.

        :param net: networks whose parameters will be optimized.
        """

        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self) -> None:
        """
        Run an optimization step.
        """

        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def __str__(self):

        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Loss class: {self.loss_class.__name__}\n" if self.loss_class else f"    Loss class: None\n"
        description += f"    Optimizer class: {self.optimizer_class.__name__}\n" if self.optimizer_class else \
            f"    Optimizer class: None\n"
        description += f"    Learning rate: {self.lr}\n"
        return description
