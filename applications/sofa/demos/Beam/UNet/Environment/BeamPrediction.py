"""
BeamPrediction
Simulation of a Beam with NN computed simulations.
The SOFA simulation contains the model used to apply the network predictions.
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python imports
import os
import sys
import numpy as np

# Working session imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BeamTraining import BeamTraining, p_grid


class BeamPrediction(BeamTraining):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1,
                 visualizer=None):

        BeamTraining.__init__(self,
                              as_tcp_ip_client=as_tcp_ip_client,
                              instance_id=instance_id,
                              instance_nb=instance_nb)

        self.create_model['fem'] = False
        self.visualizer = visualizer

        # Force pattern
        step = 0.1 if self.visualizer else 0.03
        self.amplitudes = np.concatenate((np.arange(0, 1, step),
                                          np.arange(1, 0, -step)))
        self.idx_range = 0
        self.force_value = None
        self.indices_value = None

    def recv_parameters(self, param_dict):
        """
        Exploit received parameters before scene creation.
        """

        # Receive visualizer option (either True for Vedo, False for SOFA GUI)
        self.visualizer = param_dict['visualizer'] if 'visualizer' in param_dict else self.visualizer

    def init_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called when creating Environment.
        """

        # Nothing to visualize if the predictions are run in SOFA GUI.
        if self.visualizer:
            # Add the mesh model (object will have id = 0)
            self.factory.add_mesh_callback(position_object='@nn.visual.OGL',
                                           at=self.instance_id,
                                           c='orange')
            # Arrows representing the force fields (object will have id = 1)
            self.factory.add_arrows_callback(position_object='@nn.visual.OGL',
                                             start_indices=self.cff.indices.value,
                                             vector_object='@nn.CFF',
                                             scale=0.25,
                                             c='green',
                                             at=self.instance_id)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset force amplitude index
        if self.idx_range == len(self.amplitudes):
            self.idx_range = 0

        # Create a new non-empty random box ROI, select nodes of the surface
        if self.idx_range == 0:
            # Define random box
            side = np.random.randint(0, 11)
            x_min = p_grid.min[0] if side in (3, 4) else np.random.randint(p_grid.min[0], p_grid.max[0] - 10)
            x_max = p_grid.max[0] if side in (5, 6) else np.random.randint(x_min + 10, p_grid.max[0] + 1)
            y_min = p_grid.min[1] if side in (7, 8) else np.random.randint(p_grid.min[1], p_grid.max[1] - 10)
            y_max = p_grid.max[1] if side in (9, 10) else np.random.randint(y_min + 10, p_grid.max[1] + 1)
            z_min = p_grid.min[2] if side == 0 else np.random.randint(p_grid.min[2], p_grid.max[2] - 10)
            z_max = p_grid.max[2] if side in (1, 2, 3) else np.random.randint(z_min + 10, p_grid.max[2] + 1)

            # Set the new bounding box
            self.root.nn.removeObject(self.cff_box)
            self.cff_box = self.root.nn.addObject('BoxROI', name='ForceBox', drawBoxes=False, drawSize=1,
                                                  box=[x_min, y_min, z_min, x_max, y_max, z_max])
            self.cff_box.init()

            # Get the intersection with the surface
            indices = list(self.cff_box.indices.value)
            indices = list(set(indices).intersection(set(self.idx_surface)))

            # Create a random force vector
            F = np.random.uniform(low=-1, high=1, size=(3,))
            K = np.random.randint(10, 50)
            F = K * (F / np.linalg.norm(F))
            # Keep value
            self.force_value = F
            self.indices_value = indices

        # Update force field
        F = self.amplitudes[self.idx_range] * self.force_value
        self.root.nn.removeObject(self.cff)
        self.cff = self.root.nn.addObject('ConstantForceField', name='CFF', showArrowSize=0.5,
                                          indices=self.indices_value, force=list(F))
        self.cff.init()
        self.idx_range += 1

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """

        # Send training data
        self.set_training_data(input=self.compute_input(),
                               ground_truth=np.array([]))

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # See the network prediction even if the solver diverged.
        return True
