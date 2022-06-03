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
from time import sleep

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX.Core.Utils.Visualizer.GridMapping import GridMapping

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model, p_forces


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
        self.sparse_grid = None
        self.mapping = None
        self.mapping_coarse = None

        # Force fields
        self.forces = []
        self.areas = []
        self.compute_sample = True

        # Force pattern
        step = 0.05
        self.amplitudes = np.concatenate((np.arange(0, 1, step),
                                          np.arange(1, 0, -step)))
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

    def recv_parameters(self, param_dict):

        # Get the model definition parameters
        self.compute_sample = param_dict['compute_sample'] if 'compute_sample' in param_dict else True
        self.amplitudes[0] = 0 if self.compute_sample else 1

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
            self.forces.append(np.zeros(3, ))

    def send_visualization(self):

        # Mesh representing detailed Armadillo (object will have id = 0)
        self.factory.add_object(object_type="Mesh",
                                data_dict={"positions": self.mesh.points(),
                                           'cells': self.mesh.cells(),
                                           'wireframe': True,
                                           "c": "orange",
                                           "at": self.instance_id})

        # Arrows representing the force fields (object will have id = 1)
        self.factory.add_object(object_type='Arrows',
                                data_dict={'positions': [0, 0, 0],
                                           'vectors': [0, 0, 0],
                                           'c': 'green',
                                           'at': self.instance_id})

        # Points representing the grid (object will have id = 2)
        self.factory.add_object(object_type='Points',
                                data_dict={'positions': self.sparse_grid.points(),
                                           'r': 1.,
                                           'c': 'black',
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

            # Update force amplitude index
            self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

            # Create input array
            F = np.zeros(self.input_size)
            F[self.areas[self.idx_zone]] = self.forces[self.idx_zone]

        # Load a force sample from Dataset
        else:
            sleep(0.5)
            F = np.zeros(self.input_size) if self.sample_in is None else self.sample_in

        # Set training data
        self.F = F
        self.set_training_data(input_array=F.copy(),
                               output_array=np.zeros(self.output_size))

    def apply_prediction(self, prediction):

        # Reshape to correspond to sparse grid
        U = np.reshape(prediction, self.output_size)
        self.update_visual(U)

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
                                        new_data_dict={'positions': mesh_coarse_position,
                                                       'vectors': 0.25 * self.F / p_model.scale})

        # Update sparse grid positions
        self.factory.update_object_dict(object_id=2,
                                        new_data_dict={'positions': updated_position})

        # Send visualization data to update
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)

    def close(self):
        # Shutdown message
        print("Bye!")
