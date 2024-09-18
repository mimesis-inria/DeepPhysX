import os
import sys
from collections import namedtuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import np, compute_grid_resolution, find_boundaries, define_bbox, get_nb_nodes, find_center

# Liver parameters
mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/liver.obj'
mesh_coarse = os.path.dirname(os.path.abspath(__file__)) + '/models/liver_coarse.obj'
boundary_files = [os.path.dirname(os.path.abspath(__file__)) + '/models/vein.obj']
scale = 1e-3
scale3d = np.array([scale, scale, scale])
fixed_box = find_boundaries(mesh, boundary_files, scale3d)
fixed_point = find_center(mesh, scale)
nb_nodes = get_nb_nodes(mesh_coarse)
model = {'mesh': mesh,
         'mesh_coarse': mesh_coarse,
         'scale': scale,
         'scale3d': scale3d.tolist(),
         'fixed_box': fixed_box,
         'fixed_point': fixed_point,
         'nb_nodes': nb_nodes}
p_model = namedtuple('p_liver', model)(**model)

# Grid parameters
margin_scale = 0.1
min_bbox, max_bbox, b_box = define_bbox(mesh, margin_scale, scale3d)
cell_size = 0.06
grid_resolution = compute_grid_resolution(max_bbox, min_bbox, cell_size)
grid = {'bbox': b_box,
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
