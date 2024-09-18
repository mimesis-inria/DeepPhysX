"""
Parameters
Define the following set of parameters :
    * grid parameters
"""

from numpy import array
from collections import namedtuple


# Grid parameters
grid_min = array([0., 0., 0.])
grid_max = array([100., 15., 15.])
grid_resolution = array([25, 5, 5])
grid_nb_nodes = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
grid_fixed_box = array([0., 0., 0., 0., 15., 15.])
grid_interval = array([0., 0., -50., 0., 0., -50.])
grid = {'min': grid_min,
        'max': grid_max,
        'res': grid_resolution,
        'size': grid_min.tolist() + grid_max.tolist(),
        'nb_nodes': grid_nb_nodes,
        'fixed_box': grid_fixed_box}
p_grid = namedtuple('p_grid', grid)(**grid)
