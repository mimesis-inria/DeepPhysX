from typing import Callable, Any, Optional, Tuple
from collections import namedtuple


class DataTransformation:
    """
    | DataTransformation is dedicated to data operations before and after network predictions.

    :param namedtuple config: Namedtuple containing the parameters of the network manager
    """

    def __init__(self, config: namedtuple):

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

    def transform_before_prediction(self, data_in: Any) -> Any:
        """
        | Apply data operations before network's prediction.

        :param Any data_in: Input data
        :return: Transformed input data
        """

        return data_in

    def transform_before_loss(self, data_out: Any, data_gt: Optional[Any] = None) -> Tuple[Any, Optional[Any]]:
        """
        | Apply data operations between network's prediction and loss computation.

        :param Any data_out: Prediction data
        :param Optional[Any] data_gt: Ground truth data
        :return: Transformed prediction data, transformed ground truth data
        """

        return data_out, data_gt

    def transform_before_apply(self, data_out: Any) -> Any:
        """
        | Apply data operations between loss computation and prediction apply in environment.

        :param Any data_out: Prediction data
        :return: Transformed prediction data
        """

        return data_out

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
