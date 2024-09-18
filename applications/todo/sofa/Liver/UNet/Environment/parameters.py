import os
import sys
from numpy import array
from collections import namedtuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import compute_grid_resolution, find_boundaries, define_bbox, get_nb_nodes, find_center

# Liver
mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/liver.obj'
mesh_coarse = os.path.dirname(os.path.abspath(__file__)) + '/models/liver_coarse.obj'
boundary_files = [os.path.dirname(os.path.abspath(__file__)) + '/models/vein.obj']
scale = 1e-3
scale3d = array([scale, scale, scale])
fixed_box = find_boundaries(mesh, boundary_files, scale3d)
fixed_point = find_center(mesh, scale)
nb_nodes = get_nb_nodes(mesh_coarse)
liver = {'mesh': mesh,
         'mesh_coarse': mesh_coarse,
         'scale': scale,
         'scale3d': scale3d.tolist(),
         'fixed_box': fixed_box,
         'fixed_point': fixed_point,
         'nb_nodes': nb_nodes}
p_liver = namedtuple('p_liver', liver)(**liver)

# Grid parameters
min_bbox, max_bbox, b_box = define_bbox(mesh, 0., scale3d)
bbox_size = max_bbox - min_bbox
cell_size = 0.06
grid_resolution = compute_grid_resolution(max_bbox, min_bbox, cell_size)
nb_cells = [g_r - 1 for g_r in grid_resolution]
grid = {'b_box': b_box,
        'bbox_anchor': min_bbox.tolist(),
        'bbox_size': bbox_size,
        'nb_cells': nb_cells,
        'grid_resolution': grid_resolution}
p_grid = namedtuple('p_grid', grid)(**grid)

# Forces parameters
nb_simultaneous_forces = 5
amplitude = 15 * scale
inter_distance_thresh = 50 * scale
forces = {'nb_simultaneous_forces': nb_simultaneous_forces,
          'amplitude': amplitude,
          'inter_distance_thresh': inter_distance_thresh}
p_forces = namedtuple('p_force', forces)(**forces)
