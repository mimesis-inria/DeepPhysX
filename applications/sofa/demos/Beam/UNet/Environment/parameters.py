"""
Parameters
Define the following set of parameters :
    * grid parameters
"""

from numpy import array
from collections import namedtuple


# Model grid parameters
grid_min = array([0., 0., 0.])
grid_max = array([15., 15., 100.])
grid_resolution = array([5, 5, 25])
grid_nb_nodes = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
grid_fixed_box = array([0., 0., 0., 15., 15., 0.])
grid = {'min': grid_min,
        'max': grid_max,
        'res': grid_resolution,
        'size': grid_min.tolist() + grid_max.tolist(),
        'nb_nodes': grid_nb_nodes,
        'fixed_box': grid_fixed_box}
p_grid = namedtuple('p_grid', grid)(**grid)
