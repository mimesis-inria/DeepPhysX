"""
Environment
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the network and the ground truth is the mean.
"""

# Python related imports
from numpy import mean, pi, array
from numpy.random import random, randint
from time import sleep

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment


# Create an Environment as a BaseEnvironment child class
class MeanEnvironment(BaseEnvironment):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None,
                 visu_db=None):

        BaseEnvironment.__init__(self,
                                 ip_address=ip_address,
                                 port=port,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager,
                                 visu_db=visu_db)

        # Define training data values
        self.input_value = array([])
        self.output_value = array([])

        # Environment parameters
        self.constant = False
        self.data_size = [30, 2]
        self.sleep = False
        self.allow_requests = True

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
    """

    def recv_parameters(self, param_dict):
        # If True, the same data is always sent so one can observe how the prediction crawl toward the ground truth.
        self.constant = param_dict['constant'] if 'constant' in param_dict else self.constant
        # Define the data size
        self.data_size = param_dict['data_size'] if 'data_size' in param_dict else self.data_size
        # If True, step will sleep a random time to simulate longer processes
        self.sleep = param_dict['sleep'] if 'sleep' in param_dict else self.sleep
        # If True, requests can be performed
        self.allow_requests = param_dict['allow_requests'] if 'allow_requests' in param_dict else self.allow_requests

    def create(self):
        # The vector is the input of the network and the ground truth is the mean.
        self.input_value = pi * random(self.data_size)
        self.output_value = mean(self.input_value, axis=0)

    def init_visualization(self):
        # Point cloud (object will have id = 0)
        self.factory.add_points(positions=self.input_value,
                                at=self.instance_id,
                                c='blue',
                                point_size=8)
        # Ground truth value (object will have id = 1)
        self.factory.add_points(positions=self.output_value,
                                at=self.instance_id,
                                c='green',
                                point_size=10)
        # Prediction value (object will have id = 2)
        self.factory.add_points(positions=self.output_value,
                                at=self.instance_id,
                                c='orange',
                                point_size=12)

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
    """

    async def step(self):
        # Compute new data
        if not self.constant:
            self.input_value = pi * random(self.data_size)
            self.output_value = mean(self.input_value, axis=0)
        # Simulate longer process
        if self.sleep:
            sleep(0.01 * randint(0, 10))
        # Send the training data
        self.set_training_data(input_array=self.input_value,
                               output_array=self.output_value)
        if self.allow_requests:
            # Request a prediction for the given input
            prediction = self.get_prediction(input_array=self.input_value)
            # Apply this prediction in Environment
            self.apply_prediction(prediction)

    def apply_prediction(self, prediction):
        # Update visualization with new input and ground truth
        if not self.constant:
            # Point cloud
            self.factory.update_points(object_id=0,
                                       positions=self.input_value)
            # Ground truth value
            self.factory.update_points(object_id=1,
                                       positions=self.output_value)
        # Update visualization with prediction
        self.factory.update_points(object_id=2,
                                   positions=prediction)
        # Send visualization data to update
        self.update_visualisation()

    def close(self):
        # Shutdown message
        print("Bye!")
