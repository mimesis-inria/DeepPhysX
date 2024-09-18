"""
EnvironmentPrediction.py
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the networks and the ground truth is the mean.
SofaEnvironment compatible with SOFA GUI and Prediction pipeline.
Apply predictions.
"""

# Python related imports
import os
import sys

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from EnvironmentDataset import EnvironmentDataset


# Create an Environment as a BaseEnvironment child class
class MeanEnvironmentPrediction(EnvironmentDataset):

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
        pass

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
        - onAnimateBeginEvent
        - onAnimateEndEvent
        - on_step
    """

    def onAnimateEndEvent(self, _):
        pass

    def apply_prediction(self, prediction):

        # Update MechanicalObject
        center = prediction['prediction']
        self.MO['prediction'].position.value = center
