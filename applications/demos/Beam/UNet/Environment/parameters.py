"""
Parameters
Define the following set of parameters :
    * model parameters
"""

import os
from vedo import Mesh
from collections import namedtuple


# Model parameters
grid = os.path.dirname(os.path.abspath(__file__)) + '/models/beam.obj'
grid_resolution = [5, 5, 25]
model = {'grid': grid,
         'nb_nodes': Mesh(grid).N()}
p_model = namedtuple('p_model', model)(**model)
