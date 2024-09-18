"""
EnvironmentTraining.py
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the network and the ground truth is the mean.
SofaEnvironment compatible with Training pipeline.
Apply predictions, initialize and update visualization data.
"""

# Python related imports
import os
import sys
from numpy import array

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from EnvironmentDataset import EnvironmentDataset


# Create an Environment as a BaseEnvironment child class
class EnvironmentTraining(EnvironmentDataset):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1,
                 constant=False,
                 data_size=(30, 3),
                 delay=False):
        EnvironmentDataset.__init__(self,
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

        # Prediction value (object will have id = 2)
        self.factory.add_points_callback(position_object='@predict.MO',
                                         at=self.instance_id,
                                         c='pink',
                                         point_size=10)

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
        - onAnimateBeginEvent
        - onAnimateEndEvent
        - on_step
        - apply_prediction
    """

    def onAnimateEndEvent(self, _):
        # Request and apply a prediction in the Environment
        self.apply_prediction(self.get_prediction(input=self.MO['input'].position.value))

    def apply_prediction(self, prediction):
        # Update MechanicalObject
        center = prediction['prediction']
        self.MO['prediction'].position.value = center
