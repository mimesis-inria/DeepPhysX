"""
Beam
Simulation of a Beam with deformations computed by a Fully Connected networks.
Training data are produced at each time step:
    * input: applied forces on the surface
    * output: resulted displacement of the volume
"""

# Python related imports
import os
import sys

import numpy as np
from vedo import Mesh

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model


# Create an Environment as a BaseEnvironment child class
class Beam(BaseEnvironment):

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
        self.surface = None

        # Force
        step = 0.05
        self.amplitudes = np.concatenate((np.arange(step, 1, step),
                                          np.arange(1, step, -step)))
        self.amplitudes[0] = 0
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

    def init_database(self):

        # Define the fields of the Training database
        self.define_training_fields(fields=[('input', np.ndarray), ('ground_truth', np.ndarray)])

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

    def init_visualization(self):

        # Mesh representing the grid (object will have id = 0)
        self.factory.add_mesh(positions=self.mesh.points(),
                              cells=self.mesh.cells(),
                              wireframe=True,
                              c='orange',
                              at=self.instance_id)
        # Arrows representing the force fields (object will have id = 1)
        self.factory.add_arrows(positions=np.array([0., 0., 0.]),
                                vectors=np.array([0., 0., 0.]),
                                c='green',
                                at=self.instance_id)

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
    """

    async def step(self):

        # Generate a new force
        if self.idx_amplitude == 0:
            # Define zone
            pts = self.mesh.points().copy()
            side = np.random.randint(0, 11)
            x_min = np.min(pts[:, 0]) if side in (3, 4) else np.random.randint(np.min(pts[:, 0]), np.max(pts[:, 0]) - 10)
            y_min = np.min(pts[:, 1]) if side in (7, 8) else np.random.randint(np.min(pts[:, 1]), np.max(pts[:, 1]) - 10)
            z_min = np.min(pts[:, 2]) if side == 0 else np.random.randint(np.min(pts[:, 2]), np.max(pts[:, 2]) - 10)
            x_max = np.max(pts[:, 0]) if side in (5, 6) else np.random.randint(x_min + 10, np.max(pts[:, 0]) + 1)
            y_max = np.max(pts[:, 1]) if side in (9, 10) else np.random.randint(y_min + 10, np.max(pts[:, 1]) + 1)
            z_max = np.max(pts[:, 2]) if side in (1, 2, 3) else np.random.randint(z_min + 10, np.max(pts[:, 2]) + 1)
            # Find points on surface and in box
            self.zone = np.where((pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & (pts[:, 1] >= y_min) &
                                 (pts[:, 1] <= y_max) & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max))
            self.zone = np.array(list(set(self.zone[0].tolist()).intersection(set(list(self.surface)))))
            # Force value
            F = np.random.uniform(low=-1, high=1, size=(3,))
            self.force_value = (F / np.linalg.norm(F)) * np.random.randint(10, 50)

        # Update current amplitude
        force = self.force_value * self.amplitudes[self.idx_amplitude]
        self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

        # Create input array
        F = np.zeros(self.data_size)
        F[self.zone] = force

        # Set training data
        self.F = F
        self.set_training_data(input=F.copy(),
                               ground_truth=np.zeros(self.data_size))

    def apply_prediction(self, prediction):

        # Reshape to correspond to sparse grid
        U = np.reshape(prediction['prediction'], self.data_size)
        self.update_visual(U)

    def update_visual(self, U):

        # Update surface mesh
        updated_position = self.mesh.points().copy() + U
        self.factory.update_mesh(object_id=0,
                                 positions=updated_position)

        # Update arrows representing force fields
        self.factory.update_arrows(object_id=1,
                                   positions=updated_position,
                                   vectors=0.25 * self.F)

        # Send visualization data to update
        self.update_visualisation()

    def close(self):
        # Shutdown message
        print("Bye!")
