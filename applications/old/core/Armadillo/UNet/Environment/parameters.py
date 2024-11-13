"""
Parameters
Define the sets of parameters :
    * model parameters
    * grid parameters
    * forces parameters
"""

import os
import sys
from numpy import array
from collections import namedtuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import define_bbox, compute_grid_resolution, find_extremities

# Model
mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/armadillo.obj'
coarse_mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/armadillo_coarse.obj'
sparse_grid = os.path.dirname(os.path.abspath(__file__)) + '/models/sparse_grid.obj'
scale = 1e-3
model = {'mesh': mesh,
         'mesh_coarse': coarse_mesh,
         'sparse_grid': sparse_grid,
         'scale': scale}
p_model = namedtuple('p_model', model)(**model)

# Grid
cell_size = 0.06
min_bbox, max_bbox, _ = define_bbox(mesh, 0., scale)
bbox_size = max_bbox - min_bbox
grid_resolution = compute_grid_resolution(max_bbox, min_bbox, cell_size)
nb_cells = [g_r - 1 for g_r in grid_resolution]
grid = {'origin': min_bbox.tolist(),
        'size': bbox_size,
        'nb_cells': nb_cells,
        'resolution': grid_resolution}
p_grid = namedtuple('p_grid', grid)(**grid)

# Forces
zones = ['tail', 'r_hand', 'l_hand', 'r_ear', 'l_ear', 'muzzle']
centers, radius, amplitude = {}, {}, {}
for zone, c, rad, amp in zip(zones, find_extremities(mesh, scale), [2.5, 2.5, 2.5, 2., 2., 1.5],
                             array([15, 2.5, 2.5, 7.5, 7.5, 7.5]) * scale):
    centers[zone] = c
    radius[zone] = scale * rad
    amplitude[zone] = scale * amp
forces = {'zones': zones,
          'centers': centers,
          'radius': radius,
          'amplitude': amplitude}
p_forces = namedtuple('p_forces', forces)(**forces)
