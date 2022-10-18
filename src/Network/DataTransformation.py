from typing import Callable, Any, Optional, Tuple, Dict
from collections import namedtuple

DTYPE = Dict[str, Any]


class DataTransformation:

    def __init__(self, config: namedtuple):
        """
        DataTransformation is dedicated to data operations before and after network predictions.

        :param namedtuple config: Namedtuple containing the parameters of the network manager.
        """

        self.name = self.__class__.__name__

        self.config: Any = config
        self.data_type = any

    @staticmethod
    def check_type(func: Callable[[Any, Any], Any]):

        def inner(self, *args):
            for data in args:
                if data is not None and type(data) != self.data_type:
                    raise TypeError(f"[{self.name}] Wrong data type: {self.data_type} required, get {type(data)}")
            return func(self, *args)

        return inner

    def transform_before_prediction(self,
                                    data_net: DTYPE) -> DTYPE:
        """
        Apply data operations before network's prediction.

        :param data_net: Data used by the Network.
        :return: Transformed data_net.
        """

        return data_net

    def transform_before_loss(self,
                              data_pred: DTYPE,
                              data_opt: Optional[DTYPE] = None) -> Tuple[DTYPE, Optional[DTYPE]]:
        """
        Apply data operations between network's prediction and loss computation.

        :param data_pred: Data produced by the Network.
        :param data_opt: Data used by the Optimizer.
        :return: Transformed data_pred, data_opt.
        """

        return data_pred, data_opt

    def transform_before_apply(self,
                               data_pred: DTYPE) -> DTYPE:
        """
        Apply data operations between loss computation and prediction apply in environment.

        :param data_pred: Data produced by the Network.
        :return: Transformed data_pred.
        """

        return data_pred

    def __str__(self) -> str:
        """
        :return: String containing information about the DataTransformation object
        """

        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Data type: {self.data_type}\n"
        description += f"    Transformation before prediction: Identity\n"
        description += f"    Transformation before loss: Identity\n"
        description += f"    Transformation before apply: Identity\n"
        return description
