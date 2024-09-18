"""
BeamTraining
Simulation of a Beam with FEM computed simulations.
The SOFA simulation contains two models of a Beam:
    * one to apply forces and compute deformations
    * one to apply the networks predictions
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python related imports
import os
import sys
from numpy import ndarray, zeros, reshape

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BeamSofa import BeamSofa, p_grid


class BeamTraining(BeamSofa):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1):
        BeamSofa.__init__(self,
                          as_tcp_ip_client=as_tcp_ip_client,
                          instance_id=instance_id,
                          instance_nb=instance_nb)

        self.create_model['nn'] = True
        self.data_size = (p_grid.nb_nodes, 3)

    def init_database(self):
        """
        Define the fields of the training dataset.
        """

        # Define the fields of the Training database
        self.define_training_fields(fields=[('input', ndarray), ('ground_truth', ndarray)])

    def init_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called when creating Environment.
        """

        # Add the mesh model (object will have id = 0)
        self.factory.add_mesh_callback(position_object='@fem.visual.OGL',
                                       at=self.instance_id,
                                       c='green')

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data.
        """

        # Send training data
        self.set_training_data(input=self.compute_input(),
                               ground_truth=self.compute_output())

    def compute_input(self):
        """
        Compute force field on the grid.
        """

        # Compute applied force on surface
        F = zeros(self.data_size)
        F[self.cff.indices.value.copy()] = self.cff.forces.value.copy()
        return F

    def compute_output(self):
        """
        Compute displacement field on the grid.
        """

        # Compute generated displacement on the grid
        U = self.f_grid_mo.position.value.copy() - self.f_grid_mo.rest_position.value.copy()
        return U

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model.
        """

        # Reshape to correspond regular grid
        U = reshape(prediction['prediction'], self.data_size)
        self.n_grid_mo.position.value = self.n_grid_mo.rest_position.array() + U
