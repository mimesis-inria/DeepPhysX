import os
import sys
from collections import namedtuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_nb_nodes, find_center, find_boundaries

# Liver parameters
mesh = os.path.dirname(os.path.abspath(__file__)) + '/models/liver.obj'
mesh_coarse = os.path.dirname(os.path.abspath(__file__)) + '/models/liver_coarse.obj'
boundary_files = [os.path.dirname(os.path.abspath(__file__)) + '/models/vein.obj']
grid = os.path.dirname(os.path.abspath(__file__)) + '/models/sparse_grid.obj'
scale = 1e-3
boundaries = find_boundaries(mesh, boundary_files, scale)
fixed_point = find_center(mesh, scale)
nb_nodes_mesh = get_nb_nodes(mesh_coarse)
nb_nodes_grid = get_nb_nodes(grid)
model = {'mesh': mesh,
         'mesh_coarse': mesh_coarse,
         'grid': grid,
         'scale': scale,
         'boundaries': boundaries,
         'fixed_point': fixed_point,
         'nb_nodes_mesh': nb_nodes_mesh,
         'nb_nodes_grid': nb_nodes_grid}
p_model = namedtuple('p_liver', model)(**model)

# Forces parameters
nb_simultaneous_forces = 5
amplitude = 15 * scale
inter_distance_thresh = 50 * scale
forces = {'nb_simultaneous_forces': nb_simultaneous_forces,
          'amplitude': amplitude,
          'inter_distance_thresh': inter_distance_thresh}
p_forces = namedtuple('p_force', forces)(**forces)
