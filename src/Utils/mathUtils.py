import math

import numpy
import torch
from numpy import array, concatenate, ndarray, append, empty
from numpy import cos, sin, sqrt, exp, pi, linspace
from typing import Union, Iterable, Callable, List

MathValue = Union[int, numpy.ndarray, float]


def next_power_of_2(n: int) -> int:
    """
    Return the next power of 2 given an int n

    :param n Minimum value for the power of 2
    """
    # if n = 0 or is a power of 2 return n
    if n and not (n & (n - 1)):
        return n
    return 1 << (int(math.log(n)) + 1)


def fibonacci_3D_sphere_sampling(number_of_points: int, step: int = 1, shift_phi: int = 0, shift_theta: int = 0) -> torch.Tensor:
    """
    Create a 3D sphere composed of (number_of_points / step) vertices shifted in both phi and theta according to the
    parameters

    :param number_of_points: Number of points of the Fibonacci sphere
    :param step: Point to skip between each vertices
    :param shift_phi: Horizontal rotational shift
    :param shift_theta: Vertical rotational shift

    :return A torch tensor containing al the vertices of the fibonacci sphere. Format [(x,y,z),(x1,y1,z1),...]
    """
    phi = (1.0 + sqrt(5.0)) / 2.0

    theta = array([])
    sphi = array([])
    cphi = array([])

    for i in range(0, number_of_points, step):
        i2 = 2.0 * i - (number_of_points - 1.0)
        theta = append(theta, 2.0 * pi * ((i2 + shift_theta) % number_of_points) / phi)
        sphi = append(sphi, i2 + shift_phi / number_of_points)
        cphi = append(cphi, sqrt((number_of_points + i2 + shift_phi) * (number_of_points - i2)) / number_of_points)

    valid_position = list(filter(lambda x: x[0] >= 0.0, [[cphi[i] * sin(theta[i]), cphi[i] * cos(theta[i]), sphi[i]] for i in range(len(sphi))]))

    return torch.tensor(valid_position)


def sigmoid(x: MathValue, shift: MathValue = 0, spread: MathValue = 1) -> MathValue:
    """
    Sigmoid valuation with spread and shift control

    :param x: Quantity to evaluate
    :param shift: Shift the value on the designed axis
    :param spread: Control the slope of the function. The highest the steepest.
    """
    return 1 / (1 + exp(-spread * x + shift))


def min_max_feature_scaling(feature: MathValue, pmin: MathValue, pmax: MathValue, epsilon: MathValue = 1e-10) -> MathValue:
    """
    Linear feature scaling in interval [min, max]

    :param feature: Feature to be scaled
    :param pmin: Minimum value of the interval
    :param pmax: Maximum value of the interval
    :param epsilon: Safety-net in order to do not divide by 0
    """
    return (feature - pmin + epsilon) / (pmax - pmin + epsilon)


def ndim_interpolation(val_min: MathValue, val_max: MathValue, count: Union[int, Iterable[int]], ignored_dim: Union[int, Iterable[int]] = None, technic: Callable[[MathValue, MathValue, MathValue], MathValue] = lambda m, M, c: m + c * M) -> numpy.ndarray:
    """
    Interpolate between val_min and val_max with axis-wise count node according to technic (default is linear).

    Exemple: All the "o" in the output graph are vertices in the output vector
            Input :
            val_min o-------------------------------
                    |                              |
                    |                              |
                    -------------------------------o val_max


            Output : count = [4, 3]
            val_min o---------o---------o----------o
                    |                              |
                    o---------o---------o----------o
                    |                              |
                    o---------o---------o----------o val_max

            Output : count = [4, 1]
            val_min o---------o---------o----------o
                    |                              |
                    |                              |
                    -------------------------------- val_max

            Output :  count = [1, 3]
            val_min o------------------------------o
                    |                              |
                    o------------------------------o
                    |                              |
                    o------------------------------o val_max

    :param val_min: Minimum value of the space
    :param val_max: Maximum value of the space
    :param count: Axis-wise number of interpolated node (0 == ignored dimension)
    :param ignored_dim: Dimension on which to do not interpolate (override count value)
    :param technic: Interpolation method. Default is linear.

    :return: A numpy.ndarray that contains the node interpolated according to the parameters
    """
    assert (type(val_min) == type(val_max))
    if isinstance(val_min, float) or isinstance(val_min, int):
        return array([technic(val_min, val_max, c) for c in linspace(0, 1, count)])

    if isinstance(val_min, ndarray):
        # if count is an int make it a list of val_min dim of itself
        #  Ex : count = 10 , val_min = [14,-8,1] -> count = [10, 10, 10]
        if isinstance(count, int):
            count = [count]*val_min.shape[0]
        # Coefficient is a list of array with the corresponding subdivision
        coefficients = []
        for i in range(val_min.shape[0]):
            if count[i] <= 0:
                coefficients.append([0.0])
            else:
                coefficients.append(linspace(0, 1, count[i]))

        # Set of all the indices
        valid_idx = {*linspace(0, val_min.shape[0] - 1, val_min.shape[0])}
        if ignored_dim is not None:
            # Return the set difference between all the index possible and the ignored ones
            # {0,1,2,3}-{0,2} = {1,3}
            valid_idx = valid_idx - {*ignored_dim}  # Transform to a list in order to get it as index set
        valid_idx = array([*valid_idx], dtype=int)
        # The computation starts here
        interp_idx = numpy.zeros(val_min.shape[0], dtype=int)
        values = empty((0, *val_min.shape))
        while interp_idx[valid_idx[-1]] < count[-1]:
            coeff = numpy.array([coefficients[i][interp_idx[i]] for i in range(val_min.shape[0])])
            values = concatenate((values, [technic(val_min, val_max, coeff)]), axis=0)
            interp_idx[valid_idx[0]] += 1
            # Check if we reached the number of subdivision for the dimension
            # if so we carry the extra value
            i = 0
            while interp_idx[valid_idx[i]] >= count[valid_idx[i]] and i < len(valid_idx)-1:
                interp_idx[valid_idx[i]] = 0
                interp_idx[valid_idx[i + 1]] += 1
                i += 1
        return values
    return numpy.empty(0)
