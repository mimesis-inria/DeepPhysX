"""
EnvironmentDataset.py
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the networks and the ground truth is the mean.
SofaEnvironment compatible with DataGeneration pipeline.
Initialize and update visualization data.
"""

# Python related imports
import os
import sys
from numpy import ndarray

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from EnvironmentSofa import EnvironmentSofa


# Create an Environment as a BaseEnvironment child class
class EnvironmentDataset(EnvironmentSofa):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1,
                 constant=False,
                 data_size=(30, 3),
                 delay=False):
        EnvironmentSofa.__init__(self,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id,
                                 instance_nb=instance_nb,
                                 constant=constant,
                                 data_size=data_size,
                                 delay=delay)

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
        - create
        - init_database
        - init_visualization
    """

    def init_database(self):
        # Define the fields of the Training database
        self.define_training_fields(fields=[('input', ndarray), ('ground_truth', ndarray)])

    def init_visualization(self):
        # Point cloud (object will have id = 0)
        self.factory.add_points_callback(position_object='@input.MO',
                                         at=self.instance_id,
                                         c='blue',
                                         point_size=5)

        # Ground truth value (object will have id = 1)
        self.factory.add_points_callback(position_object='@output.MO',
                                         at=self.instance_id,
                                         c='green',
                                         point_size=10)

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
        - onAnimateBeginEvent
        - onAnimateEndEvent
        - on_step
    """

    def onAnimateBeginEvent(self, _):
        EnvironmentSofa.onAnimateBeginEvent(self, _)
        self.set_training_data(input=self.MO['input'].position.value,
                               ground_truth=self.MO['ground_truth'].position.value)
