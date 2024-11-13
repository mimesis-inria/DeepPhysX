"""
LiverPrediction
Simulation of a Liver with NN computed simulations.
The SOFA simulation contains the model used to apply the networks predictions.
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
from LiverTraining import LiverTraining
from parameters import p_forces, p_liver


class LiverPrediction(LiverTraining):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1,
                 visualizer=None,
                 nb_forces=3):

        LiverTraining.__init__(self,
                               as_tcp_ip_client=as_tcp_ip_client,
                               instance_id=instance_id,
                               instance_nb=instance_nb)

        self.create_model['fem'] = False
        self.visualizer = visualizer

        # Force pattern
        step = 0.1 if self.visualizer else 0.05
        self.amplitudes = np.concatenate((np.arange(0, 1, step),
                                          np.arange(1, 0, -step)))
        self.idx_range = 0
        self.nb_forces = nb_forces
        self.forces = [None] * self.nb_forces

    def init_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called whn creating Environment.
        """

        # Nothing to visualize if the predictions are run in SOFA GUI.
        if self.visualizer:
            # Add the mesh model (object will have id = 0)
            self.factory.add_mesh_callback(position_object='@nn.visual.OGL',
                                           at=self.instance_id,
                                           c='orange')
            # Point cloud for sparse grid (object will have id = 1)
            self.factory.add_points_callback(position_object='@nn.SparseGrid',
                                             point_size=2,
                                             c='grey',
                                             at=self.instance_id)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset force amplitude index
        if self.idx_range == len(self.amplitudes):
            self.idx_range = 0
            self.n_surface_mo.position.value = self.n_surface_mo.rest_position.value

        # Build and set forces vectors
        if self.idx_range == 0:

            # Pick up a random visible surface point, select the points in a centered sphere
            selected_centers = np.empty([0, 3])
            for i in range(self.nb_forces):
                # Pick up a random visible surface point, select the points in a centered sphere
                current_point = self.n_surface_mo.position.value[
                    np.random.randint(len(self.n_surface_mo.position.value))]
                # Check distance to other points
                distance_check = True
                for p in selected_centers:
                    distance = np.linalg.norm(current_point - p)
                    if distance < p_forces.inter_distance_thresh:
                        distance_check = False
                        break
                empty_indices = False

                if distance_check:
                    # Add center to the selection
                    selected_centers = np.concatenate((selected_centers, np.array([current_point])))
                    # Set sphere center
                    self.sphere[i].centers.value = [current_point]
                    # Build force vector
                    if len(self.sphere[i].indices.value) > 0:
                        f = np.random.uniform(low=-1, high=1, size=(3,))
                        self.forces[i] = (f / np.linalg.norm(f)) * p_forces.amplitude
                        self.force_field[i].indices.value = self.sphere[i].indices.array()
                        self.force_field[i].force.value = self.forces[i] * self.amplitudes[self.idx_range]
                    else:
                        empty_indices = True

                if not distance_check or empty_indices:
                    # Reset sphere position
                    self.sphere[i].centers.value = [p_liver.fixed_point.tolist()]
                    # Reset force field
                    self.force_field[i].indices.value = [0]
                    self.force_field[i].force.value = np.array([0.0, 0.0, 0.0])
                    self.forces[i] = np.array([0., 0., 0.])

        # Change force amplitude
        else:
            for i in range(self.nb_forces):
                self.force_field[i].force.value = self.forces[i] * self.amplitudes[self.idx_range]
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

        # See the networks prediction even if the solver diverged.
        return True
