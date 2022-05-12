from functools import reduce
from operator import iconcat
from typing import List
import numpy


def flatten(a: numpy.ndarray) -> List[float]:
    """
    Return a flattened version of a numpy array

    :param numpy.ndarray a: Array to be flattened
    """
    return reduce(iconcat, a, [])
