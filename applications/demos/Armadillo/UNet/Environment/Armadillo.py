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
from numpy import ndarray, array, concatenate, arange, zeros, unique, reshape
from numpy.random import randint, uniform
from vedo import Mesh
from math import pow

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX.Core.Utils.Visualizer.GridMapping import GridMapping

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model, p_forces, p_grid


# Create an Environment as a BaseEnvironment child class
class Armadillo(BaseEnvironment):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1):

        BaseEnvironment.__init__(self,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id,
                                 instance_nb=instance_nb)

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

        # Force pattern
        step = 0.05
        self.amplitudes = concatenate((arange(step, 1, step),
                                       arange(1, step, -step)))
        self.amplitudes[0] = 0
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

    def init_database(self):

        # Define the fields of the Training database
        self.define_training_fields(fields=[('input', ndarray), ('ground_truth', ndarray)])

    def create(self):

        # Load the meshes
        self.mesh = Mesh(p_model.mesh).scale(p_model.scale)
        self.mesh_coarse = Mesh(p_model.mesh_coarse).scale(p_model.scale)
        self.sparse_grid = Mesh(p_model.sparse_grid)

        # Define regular grid
        x_reg = [p_grid.origin[0] + i * p_grid.size[0] / p_grid.nb_cells[0] for i in range(p_grid.nb_cells[0] + 1)]
        y_reg = [p_grid.origin[1] + i * p_grid.size[1] / p_grid.nb_cells[1] for i in range(p_grid.nb_cells[1] + 1)]
        z_reg = [p_grid.origin[2] + i * p_grid.size[2] / p_grid.nb_cells[2] for i in range(p_grid.nb_cells[2] + 1)]
        grid_positions = array([[[[x, y, z] for x in x_reg] for y in y_reg] for z in z_reg]).reshape(-1, 3)
        cell_corner = lambda x, y, z: len(x_reg) * (len(y_reg) * z + y) + x
        grid_cells = array([[[[[cell_corner(ix, iy, iz),
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
        self.grid_correspondences = zeros(self.sparse_grid.N(), dtype=int)
        x_sparse = unique(self.sparse_grid.points()[:, 0])
        y_sparse = unique(self.sparse_grid.points()[:, 1])
        z_sparse = unique(self.sparse_grid.points()[:, 2])
        if len(x_reg) != len(x_sparse) or len(y_reg) != len(y_sparse) or len(z_reg) != len(z_sparse):
            raise ValueError('Grids should have the same dimension')
        d = [x_sparse[1] - x_sparse[0], y_sparse[1] - y_sparse[0], z_sparse[1] - z_sparse[0]]
        origin = [x_sparse[0], y_sparse[0], z_sparse[0]]
        for i, node in enumerate(self.sparse_grid.points()):
            p = array(node) - array(origin)
            ix = int(round(p[0] / d[0]))
            iy = int(round(p[1] / d[1]))
            iz = int(round(p[2] / d[2]))
            idx = len(x_sparse) * (len(y_sparse) * iz + iy) + ix
            self.grid_correspondences[i] = idx

        # Define force fields
        sphere = lambda x, z: sum([pow(x_i - c_i, 2) for x_i, c_i in zip(x, p_forces.centers[z])]) <= pow(
            p_forces.radius[z], 2)
        for zone in p_forces.zones:
            self.areas.append([])
            self.g_areas.append([])
            for i, node in enumerate(self.mesh_coarse.points()):
                if sphere(node, zone):
                    self.areas[-1].append(i)
                    self.g_areas[-1] += self.mapping_coarse.cells[i].tolist()
            self.areas[-1] = array(self.areas[-1])
            self.g_areas[-1] = unique(self.g_areas[-1])
            self.forces.append(zeros((len(self.areas[-1]), 3)))

    def init_visualization(self):

        # Mesh representing detailed Armadillo (object will have id = 0)
        self.factory.add_mesh(positions=self.mesh.points(),
                              cells=self.mesh.cells(),
                              wireframe=True,
                              c='orange',
                              at=self.instance_id)
        # Arrows representing the force fields (object will have id = 1)
        self.factory.add_arrows(positions=array([0., 0., 0.]),
                                vectors=array([0., 0., 0.]),
                                c='green',
                                at=self.instance_id)
        # Sparse grid (object will have id = 2)
        self.factory.add_points(positions=self.sparse_grid.points(),
                                point_size=2,
                                c='grey',
                                at=self.instance_id)

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
    """

    async def step(self):

        # Generate a new force
        if self.idx_amplitude == 0:
            self.idx_zone = randint(0, len(self.forces))
            zone = p_forces.zones[self.idx_zone]
            self.force_value = uniform(low=-1, high=1, size=(3,)) * p_forces.amplitude[zone]

        # Update current force amplitude
        self.forces[self.idx_zone] = self.force_value * self.amplitudes[self.idx_amplitude]
        self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

        # Create input array
        F = zeros(self.data_size)
        F[self.grid_correspondences[self.g_areas[self.idx_zone]]] = self.forces[self.idx_zone]
        self.F = zeros((self.mesh_coarse.N(), 3))
        self.F[self.areas[self.idx_zone]] = self.forces[self.idx_zone]

        # Set training data
        self.set_training_data(input=F.copy(),
                               ground_truth=zeros(self.data_size))

    def apply_prediction(self, prediction):

        # Reshape to correspond to sparse grid
        U = reshape(prediction['prediction'], self.data_size)
        U_sparse = U[self.grid_correspondences]
        self.update_visual(U_sparse)

    def update_visual(self, U):

        # Apply mappings
        updated_position = self.sparse_grid.points().copy() + U
        mesh_position = self.mapping.apply(updated_position)
        mesh_coarse_position = self.mapping_coarse.apply(updated_position)

        # Update surface mesh
        self.factory.update_mesh(object_id=0,
                                 positions=mesh_position)

        # Update arrows representing force fields
        self.factory.update_arrows(object_id=1,
                                   positions=mesh_coarse_position,
                                   vectors=0.25 * self.F / p_model.scale)

        # Update sparse grid positions
        self.factory.update_points(object_id=2,
                                   positions=updated_position)

        # Send visualization data to update
        self.update_visualisation()

    def close(self):
        # Shutdown message
        print("Bye!")
