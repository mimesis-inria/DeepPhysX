import os
import sys
from collections import namedtuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import define_bbox, find_center, compute_grid_resolution

# Liver parameters
mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/liver.obj'
mesh_coarse = os.path.dirname(os.path.abspath(__file__)) + '/models/liver_coarse.obj'
sparse_grid = os.path.dirname(os.path.abspath(__file__)) + '/models/sparse_grid.obj'
scale = 1e-3
fixed_point = find_center(mesh, scale)
model = {'mesh': mesh,
         'mesh_coarse': mesh_coarse,
         'sparse_grid': sparse_grid,
         'scale': scale,
         'fixed_point': fixed_point}
p_model = namedtuple('p_liver', model)(**model)

# Grid parameters
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

# Forces parameters
nb_simultaneous_forces = 5
amplitude = 15 * scale
inter_distance_thresh = 50 * scale
forces = {'nb_simultaneous_forces': nb_simultaneous_forces,
          'amplitude': amplitude,
          'inter_distance_thresh': inter_distance_thresh}
p_forces = namedtuple('p_force', forces)(**forces)


