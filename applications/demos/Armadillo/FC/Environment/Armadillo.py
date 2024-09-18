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

from numpy import ndarray, array, concatenate, arange, zeros, reshape
from numpy.random import randint, uniform
from vedo import Mesh
from math import pow

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX.Core.Utils.Visualizer.GridMapping import GridMapping

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model, p_forces


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
        self.sparse_grid = None
        self.mapping = None
        self.mapping_coarse = None

        # Force fields
        self.forces = []
        self.areas = []

        # Force pattern
        step = 0.05
        self.amplitudes = concatenate((arange(0, 1, step),
                                       arange(1, 0, -step)))
        self.amplitudes[0] = 0
        self.idx_amplitude = 0
        self.force_value = None
        self.idx_zone = 0
        self.F = None

        # Data sizes
        self.input_size = (p_model.nb_nodes_mesh, 3)
        self.output_size = (p_model.nb_nodes_grid, 3)

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
    """

    def init_database(self):

        # Define the fields of the Training database
        self.define_training_fields(fields=[('input', ndarray), ('ground_truth', ndarray)])

    def create(self):

        # Load the mesh and the sparse grid, init the mapping between them
        self.mesh = Mesh(p_model.mesh).scale(p_model.scale)
        self.mesh_coarse = Mesh(p_model.mesh_coarse).scale(p_model.scale)
        self.sparse_grid = Mesh(p_model.sparse_grid)
        self.mapping = GridMapping(self.sparse_grid, self.mesh)
        self.mapping_coarse = GridMapping(self.sparse_grid, self.mesh_coarse)

        # Define force fields
        sphere = lambda x, y: sum([pow(x_i - y_i, 2) for x_i, y_i in zip(x, y)])
        for zone in p_forces.zones:
            # Find the spherical area
            self.areas.append([])
            for i, pts in enumerate(self.mesh_coarse.points()):
                if sphere(pts, p_forces.centers[zone]) <= pow(p_forces.radius[zone], 2):
                    self.areas[-1].append(i)
            # Init force value
            self.forces.append(zeros(3, ))

    def init_visualization(self):

        # Mesh representing detailed Armadillo (object will have id = 0)
        self.factory.add_mesh(positions=self.mesh.points(),
                              cells=self.mesh.cells(),
                              wireframe=False,
                              c='orange',
                              at=self.instance_id)
        # Arrows representing the force fields (object will have id = 1)
        self.factory.add_arrows(positions=array([0., 0., 0.]),
                                vectors=array([0., 0., 0.]),
                                c='green',
                                at=self.instance_id)
        # Points representing the grid (object will have id = 2)
        self.factory.add_points(positions=self.sparse_grid.points(),
                                point_size=1,
                                c='black',
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

        # Update force amplitude index
        self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

        # Create input array
        F = zeros(self.input_size)
        F[self.areas[self.idx_zone]] = self.forces[self.idx_zone]

        # Set training data
        self.F = F
        self.set_training_data(input=F.copy(),
                               ground_truth=zeros(self.output_size))

    def apply_prediction(self, prediction):

        # Reshape to correspond to sparse grid
        U = reshape(prediction['prediction'], self.output_size)
        self.update_visual(U)

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
