"""
Armadillo
Simulation of an Armadillo with deformation predictions from a Fully Connected.
Training data are produced at each time step:
    * input: applied forces on each surface node
    * prediction: resulted displacement of each surface node
"""

# Python related imports
import os
import sys

import numpy as np
from vedo import Mesh
from math import pow
from time import sleep, time

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX.Core.Utils.Visualizer.GridMapping import GridMapping

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model, p_forces, p_grid


# Create an Environment as a BaseEnvironment child class
class Armadillo(BaseEnvironment):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        BaseEnvironment.__init__(self,
                                 ip_address=ip_address,
                                 port=port,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)

        # Topology
        self.mesh = None
        self.mesh_coarse = None
        self.grid = None
        self.sparse_grid = None
        self.mapping = None
        self.mapping_coarse = None
        self.grid_correspondences = None

        # Force fields
        self.forces = []
        self.areas = []
        self.g_areas = []
        self.compute_sample = None

        # Force pattern
        step = 0.05
        self.amplitudes = np.concatenate((np.arange(step, 1, step),
                                          np.arange(1, step, -step)))
        self.idx_amplitude = 0
        self.force_value = None
        self.idx_zone = 0
        self.F = None

        nb_cells = (p_grid.nb_cells[0] + 1) * (p_grid.nb_cells[1] + 1) * (p_grid.nb_cells[2] + 1)
        self.data_size = (nb_cells, 3)

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
    """

    def recv_parameters(self, param_dict):

        # Get the model definition parameters
        self.compute_sample = param_dict['compute_sample'] if 'compute_sample' in param_dict else True
        self.amplitudes[0] = 0 if self.compute_sample else 1

    def create(self):

        # Load the meshes
        self.mesh = Mesh(p_model.mesh).scale(p_model.scale)
        self.mesh_coarse = Mesh(p_model.mesh_coarse).scale(p_model.scale)
        self.sparse_grid = Mesh(p_model.sparse_grid)

        # Define regular grid
        x_reg = [p_grid.origin[0] + i * p_grid.size[0] / p_grid.nb_cells[0] for i in range(p_grid.nb_cells[0] + 1)]
        y_reg = [p_grid.origin[1] + i * p_grid.size[1] / p_grid.nb_cells[1] for i in range(p_grid.nb_cells[1] + 1)]
        z_reg = [p_grid.origin[2] + i * p_grid.size[2] / p_grid.nb_cells[2] for i in range(p_grid.nb_cells[2] + 1)]
        grid_positions = np.array([[[[x, y, z] for x in x_reg] for y in y_reg] for z in z_reg]).reshape(-1, 3)
        cell_corner = lambda x, y, z: len(x_reg) * (len(y_reg) * z + y) + x
        grid_cells = np.array([[[[[cell_corner(ix, iy, iz),
                                   cell_corner(ix + 1, iy, iz),
                                   cell_corner(ix + 1, iy + 1, iz),
                                   cell_corner(ix, iy + 1, iz),
                                   cell_corner(ix, iy, iz + 1),
                                   cell_corner(ix + 1, iy, iz + 1),
                                   cell_corner(ix + 1, iy + 1, iz + 1),
                                   cell_corner(ix, iy + 1, iz + 1)]
                                  for ix in range(p_grid.nb_cells[0])]
                                 for iy in range(p_grid.nb_cells[1])]
                                for iz in range(p_grid.nb_cells[2])]]).reshape(-1, 8)
        self.grid = Mesh([grid_positions, grid_cells])

        # Init mappings between meshes and sparse grid
        self.mapping = GridMapping(self.sparse_grid, self.mesh)
        self.mapping_coarse = GridMapping(self.sparse_grid, self.mesh_coarse)

        # Init correspondences between sparse and regular grid
        self.grid_correspondences = np.zeros(self.sparse_grid.N(), dtype=int)
        x_sparse = np.unique(self.sparse_grid.points()[:, 0])
        y_sparse = np.unique(self.sparse_grid.points()[:, 1])
        z_sparse = np.unique(self.sparse_grid.points()[:, 2])
        if len(x_reg) != len(x_sparse) or len(y_reg) != len(y_sparse) or len(z_reg) != len(z_sparse):
            raise ValueError('Grids should have the same dimension')
        d = [x_sparse[1] - x_sparse[0], y_sparse[1] - y_sparse[0], z_sparse[1] - z_sparse[0]]
        origin = [x_sparse[0], y_sparse[0], z_sparse[0]]
        for i, node in enumerate(self.sparse_grid.points()):
            p = np.array(node) - np.array(origin)
            ix = int(round(p[0] / d[0]))
            iy = int(round(p[1] / d[1]))
            iz = int(round(p[2] / d[2]))
            idx = len(x_sparse) * (len(y_sparse) * iz + iy) + ix
            self.grid_correspondences[i] = idx

        # Define force fields
        sphere = lambda x, z: sum([pow(x_i - c_i, 2) for x_i, c_i in zip(x, p_forces.centers[z])]) <= pow(p_forces.radius[z], 2)
        for zone in p_forces.zones:
            self.areas.append([])
            self.g_areas.append([])
            for i, node in enumerate(self.mesh_coarse.points()):
                if sphere(node, zone):
                    self.areas[-1].append(i)
                    self.g_areas[-1] += self.mapping_coarse.cells[i].tolist()
            self.areas[-1] = np.array(self.areas[-1])
            self.g_areas[-1] = np.unique(self.g_areas[-1])
            self.forces.append(np.zeros((len(self.areas[-1]), 3)))

    def send_visualization(self):

        # Mesh representing detailed Armadillo (object will have id = 0)
        self.factory.add_object(object_type='Mesh',
                                data_dict={'positions': self.mesh.points(),
                                           'cells': self.mesh.cells(),
                                           'wireframe': True,
                                           'c': 'orange',
                                           'at': self.instance_id})

        # Arrows representing the force fields (object will have id = 1)
        self.factory.add_object(object_type='Arrows',
                                data_dict={'positions': [0, 0, 0],
                                           'vectors': [0., 0., 0.],
                                           'c': 'green',
                                           'at': self.instance_id})

        # Sparse grid (object will have id = 2)
        self.factory.add_object(object_type='Points',
                                data_dict={'positions': self.sparse_grid.points(),
                                           'r': 2,
                                           'c': 'grey',
                                           'at': self.instance_id})

        # Return the visualization data
        return self.factory.objects_dict

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
    """

    async def step(self):

        # Compute a force sample
        if self.compute_sample:
            # Generate a new force
            if self.idx_amplitude == 0:
                self.idx_zone = np.random.randint(0, len(self.forces))
                zone = p_forces.zones[self.idx_zone]
                self.force_value = np.random.uniform(low=-1, high=1, size=(3,)) * p_forces.amplitude[zone]

            # Update current force amplitude
            self.forces[self.idx_zone] = self.force_value * self.amplitudes[self.idx_amplitude]
            self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

            # Create input array
            F = np.zeros(self.data_size)
            F[self.grid_correspondences[self.g_areas[self.idx_zone]]] = self.forces[self.idx_zone]
            self.F = np.zeros((self.mesh_coarse.N(), 3))
            self.F[self.areas[self.idx_zone]] = self.forces[self.idx_zone]

        # Load a force sample from Dataset
        else:
            sleep(0.5)
            F = np.zeros(self.data_size) if self.sample_in is None else self.sample_in
            self.F = F[self.grid_correspondences]

        # Set training data
        self.set_training_data(input_array=F.copy(),
                               output_array=np.zeros(self.data_size))

    def apply_prediction(self, prediction):

        # Reshape to correspond to sparse grid
        U = np.reshape(prediction, self.data_size)
        U_sparse = U[self.grid_correspondences]
        self.update_visual(U_sparse)

    def update_visual(self, U):

        # Apply mappings
        updated_position = self.sparse_grid.points().copy() + U
        mesh_position = self.mapping.apply(updated_position)
        mesh_coarse_position = self.mapping_coarse.apply(updated_position)

        # Update surface mesh
        self.factory.update_object_dict(object_id=0,
                                        new_data_dict={'positions': mesh_position})

        # Update arrows representing force fields
        self.factory.update_object_dict(object_id=1,
                                        new_data_dict={'positions': mesh_coarse_position if self.compute_sample else updated_position,
                                                       'vectors': 0.25 * self.F / p_model.scale})

        # Update sparse grid positions
        self.factory.update_object_dict(object_id=2,
                                        new_data_dict={'positions': updated_position})

        # Send visualization data to update
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)

    def close(self):
        # Shutdown message
        print("Bye!")
