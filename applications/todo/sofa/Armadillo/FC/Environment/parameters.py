"""
Parameters
Define the following sets of parameters :
    * model parameters
    * grid parameters
    * forces parameters
"""

import os
import sys
from numpy import array
from collections import namedtuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import compute_grid_resolution, define_bbox, find_extremities, find_fixed_box

# Model
mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/armadillo.obj'
coarse_mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/armadillo_coarse.obj'
scale = 1e-3
scale3d = 3 * [scale]
fixed_box = find_fixed_box(mesh, scale)
model = {'mesh': mesh,
         'mesh_coarse': coarse_mesh,
         'scale': scale,
         'scale3d': scale3d,
         'fixed_box': fixed_box}
p_model = namedtuple('p_model', model)(**model)

# Grid
margin_scale = 0.2
cell_size = 0.06
min_bbox, max_bbox, bbox = define_bbox(mesh, margin_scale, scale)
grid_resolution = compute_grid_resolution(max_bbox, min_bbox, cell_size)
grid = {'bbox': bbox,
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
          'amplitude': amplitude,
          'simultaneous': 1}
p_forces = namedtuple('p_forces', forces)(**forces)
