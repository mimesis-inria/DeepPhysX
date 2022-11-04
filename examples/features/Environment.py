"""
Environment
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the network and the ground truth is the mean.
"""

# Python related imports
from numpy import mean, pi, array, ndarray
from numpy.random import random, randint
from time import sleep

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment


# Create an Environment as a BaseEnvironment child class
class MeanEnvironment(BaseEnvironment):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1,
                 constant=False,
                 data_size=(30, 2),
                 delay=False,
                 allow_request=True):

        BaseEnvironment.__init__(self,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id,
                                 instance_nb=instance_nb)

        # Define training data values
        self.pcd = array([])
        self.mean = array([])

        # Environment parameters
        self.constant = constant
        self.data_size = data_size
        self.delay = delay
        self.allow_requests = allow_request

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
    """

    def create(self):

        self.pcd = pi * random(self.data_size)
        self.mean = mean(self.pcd, axis=0)

    def init_database(self):

        # Define the fields of the training Database
        self.define_training_fields(fields=[('input', ndarray), ('ground_truth', ndarray)])

    def init_visualization(self):

        # Point cloud (object will have id = 0)
        self.factory.add_points(positions=self.pcd,
                                at=self.instance_id,
                                c='blue',
                                point_size=8)
        # Ground truth value (object will have id = 1)
        self.factory.add_points(positions=self.mean,
                                at=self.instance_id,
                                c='green',
                                point_size=10)
        # Prediction value (object will have id = 2)
        self.factory.add_points(positions=self.mean,
                                at=self.instance_id,
                                c='orange',
                                point_size=12)

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
    """

    async def step(self):

        # Simulate longer process
        if self.delay:
            sleep(0.01 * randint(0, 10))

        # Compute new data
        if not self.constant:
            self.pcd = pi * random(self.data_size)
            self.mean = mean(self.pcd, axis=0)

        # Update visualization data
        if not self.constant:
            # Point cloud
            self.factory.update_points(object_id=0,
                                       positions=self.pcd)
            # Ground truth value
            self.factory.update_points(object_id=1,
                                       positions=self.mean)

        # Send the training data
        self.set_training_data(input=self.pcd,
                               ground_truth=self.mean)

        if self.allow_requests:
            # Request a prediction for the given input
            prediction = self.get_prediction(input=self.pcd)
            # Apply this prediction in Environment
            self.apply_prediction(prediction)
        else:
            self.update_visualisation()

    def apply_prediction(self, prediction):

        # Update visualization with prediction
        self.factory.update_points(object_id=2,
                                   positions=prediction['prediction'])
        # Send visualization data to update
        self.update_visualisation()

    def close(self):
        # Shutdown message
        print("Bye!")
