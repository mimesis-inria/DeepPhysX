"""
Beam
Simulation of a Beam with deformations computed by a Fully Connected Network.
Training data are produced at each time step:
    * input: applied forces on the surface
    * output: resulted displacement of the volume
"""

# Python related imports
import os
import sys

import numpy as np
from vedo import Mesh
from time import sleep

# DeepPhysX related imports
from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model


# Create an Environment as a BaseEnvironment child class
class Beam(BaseEnvironment):

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
        self.surface = None

        # Force
        self.compute_sample = True
        step = 0.1
        self.amplitudes = np.concatenate((np.arange(0, 1, step),
                                          np.arange(1, 0, -step)))
        self.idx_amplitude = 0
        self.force_value = None
        self.zone = None
        self.F = None

        # Data size
        self.data_size = (p_model.nb_nodes, 3)

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
    """

    def recv_parameters(self, param_dict):

        # Get the model definition parameters
        self.compute_sample = param_dict['compute_sample'] if 'compute_sample' in param_dict else True
        self.amplitudes[0] = 0 if self.compute_sample else 1

    def create(self):

        # Create beam mesh
        self.mesh = Mesh(p_model.grid)
        # Extract surface points
        pts = self.mesh.points().copy()
        self.surface = np.concatenate((np.argwhere(pts[:, 0] == np.min(pts[:, 0])),
                                       np.argwhere(pts[:, 0] == np.max(pts[:, 0])),
                                       np.argwhere(pts[:, 1] == np.min(pts[:, 1])),
                                       np.argwhere(pts[:, 1] == np.max(pts[:, 1])),
                                       np.argwhere(pts[:, 2] == np.min(pts[:, 2])),
                                       np.argwhere(pts[:, 2] == np.max(pts[:, 2])))).reshape(-1)
        self.surface = np.unique(self.surface)

    def send_visualization(self):

        # Mesh representing the grid (object will have id = 0)
        self.factory.add_object(object_type='Mesh',
                                data_dict={'positions': self.mesh.points(),
                                           'cells': self.mesh.cells(),
                                           'wireframe': True,
                                           'c': 'orange',
                                           'at': self.instance_id})

        # Arrows representing the force fields (object will have id = 1)
        self.factory.add_object(object_type='Arrows',
                                data_dict={'positions': [0, 0, 0],
                                           'vectors': [0, 0, 0],
                                           'c': 'green',
                                           'at': self.instance_id})

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
                # Define zone
                pts = self.mesh.points().copy()
                side = np.random.randint(0, 6)
                x_min = np.min(pts[:, 0]) if side == 0 else np.random.randint(np.min(pts[:, 0]), np.max(pts[:, 0]) - 10)
                y_min = np.min(pts[:, 1]) if side == 1 else np.random.randint(np.min(pts[:, 1]), np.max(pts[:, 1]) - 10)
                z_min = np.min(pts[:, 2]) if side == 2 else np.random.randint(np.min(pts[:, 2]), np.max(pts[:, 2]) - 10)
                x_max = np.max(pts[:, 0]) if side == 3 else np.random.randint(x_min + 10, np.max(pts[:, 0]) + 1)
                y_max = np.max(pts[:, 1]) if side == 4 else np.random.randint(y_min + 10, np.max(pts[:, 1]) + 1)
                z_max = np.max(pts[:, 2]) if side == 5 else np.random.randint(z_min + 10, np.max(pts[:, 2]) + 1)
                # Find points on surface and in box
                self.zone = np.where((pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & (pts[:, 1] >= y_min) &
                                     (pts[:, 1] <= y_max) & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max))
                self.zone = np.array(list(set(self.zone[0].tolist()).intersection(set(list(self.surface)))))
                # Force value
                F = np.random.uniform(low=-1, high=1, size=(3,)) * np.random.randint(20, 30)
                self.force_value = F

            # Update current amplitude
            self.force = self.force_value * self.amplitudes[self.idx_amplitude]
            self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

            # Create input array
            F = np.zeros(self.data_size)
            F[self.zone] = self.force

        # Load a force sample from Dataset
        else:
            sleep(0.5)
            F = np.zeros(self.data_size) if self.sample_in is None else self.sample_in

        # Set training data
        self.F = F
        self.set_training_data(input_array=F.copy(),
                               output_array=np.zeros(self.data_size))

    def apply_prediction(self, prediction):

        # Reshape to correspond to sparse grid
        U = np.reshape(prediction, self.data_size)
        self.update_visual(U)

    def update_visual(self, U):

        # Update surface mesh
        updated_mesh = self.mesh.clone().points(self.mesh.points().copy() + U)
        self.factory.update_object_dict(object_id=0,
                                        new_data_dict={'positions': updated_mesh.points().copy()})

        # Update arrows representing force fields
        self.factory.update_object_dict(object_id=1,
                                        new_data_dict={'positions': updated_mesh.points().copy(),
                                                       'vectors': 0.25 * self.F})

        # Send visualization data to update
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)

    def close(self):
        # Shutdown message
        print("Bye!")
