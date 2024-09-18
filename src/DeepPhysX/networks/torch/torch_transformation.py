from typing import Tuple, Optional, Dict
from torch import Tensor
from collections import namedtuple

from DeepPhysX.networks.core.base_transformation import BaseTransformation


class TorchTransformation(BaseTransformation):

    def __init__(self, config: namedtuple):
        """
        TorchBaseTransformation manages data operations before and after networks predictions.

        :param config: Set of TorchTransformation parameters.
        """

        BaseTransformation.__init__(self, config)
        self.data_type = Tensor

    @BaseTransformation.check_type
    def transform_before_prediction(self,
                                    data_net: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply data operations before networks's prediction.

        :param data_net: Data used by the Network.
        :return: Transformed data_net.
        """

        return data_net

    @BaseTransformation.check_type
    def transform_before_loss(self,
                              data_pred: Dict[str, Tensor],
                              data_opt: Optional[Dict[str, Tensor]] = None) -> Tuple[Dict[str, Tensor], Optional[Dict[str, Tensor]]]:
        """
        Apply data operations between networks's prediction and loss computation.

        :param data_pred: Data produced by the Network.
        :param data_opt: Data used by the Optimizer.
        :return: Transformed data_pred, data_opt.
        """

        return data_pred, data_opt

    @BaseTransformation.check_type
    def transform_before_apply(self,
                               data_pred: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply data operations between loss computation and prediction apply in environment.

        :param data_pred: Data produced by the Network.
        :return: Transformed data_pred.
        """

        return data_pred

    def __str__(self) -> str:

        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Data type: {self.data_type}\n"
        description += f"    Transformation before prediction: Identity\n"
        description += f"    Transformation before loss: Identity\n"
        description += f"    Transformation before apply: Identity\n"
        return description
