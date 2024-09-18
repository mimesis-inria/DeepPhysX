"""
EnvironmentSofa.py
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the network and the ground truth is the mean.
SofaEnvironment compatible with SOFA GUI.
Create scene graph and define behavior.
"""

# Python related imports
from numpy import mean, pi, array
from numpy.random import random, randint
from time import sleep

# DeepPhysX related imports
from DeepPhysX.Sofa.Environment.SofaEnvironment import SofaEnvironment


# Create an Environment as a BaseEnvironment child class
class EnvironmentSofa(SofaEnvironment):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1,
                 constant=False,
                 data_size=(30, 3),
                 delay=False):

        SofaEnvironment.__init__(self,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id,
                                 instance_nb=instance_nb)

        # MechanicalObjects container
        self.MO = {}

        # Environment parameters
        self.constant = constant
        self.data_size = data_size
        self.delay = delay

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
        - create
        - init_database
        - init_visualization
    """

    def create(self):

        # Initialize data
        pcd = pi * random(self.data_size)
        center = array([mean(pcd, axis=0)])

        # Add a MechanicalObject for the input data
        self.root.addChild('input')
        self.MO['input'] = self.root.input.addObject('MechanicalObject', name='MO', position=pcd.tolist(),
                                                     showObject=True, showObjectScale=5)
        # Add a MechanicalObject for the ground truth
        self.root.addChild('output')
        self.MO['ground_truth'] = self.root.output.addObject('MechanicalObject', name='MO', position=center.tolist(),
                                                             showObject=True, showObjectScale=10, showColor="0 200 0 255")
        # Add a MechanicalObject for the prediction
        self.root.addChild('predict')
        self.MO['prediction'] = self.root.predict.addObject('MechanicalObject', name='MO', position=[0., 0., 0.],
                                                            showObject=True, showObjectScale=10, showColor="100 0 200 255")

    def init_database(self):
        pass

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
        - onAnimateBeginEvent
        - onAnimateEndEvent
        - on_step
    """

    def onAnimateBeginEvent(self, _):

        # Compute new data
        if not self.constant:
            pcd = pi * random(self.data_size)
            center = array([mean(pcd, axis=0)])
            self.MO['input'].position.value = pcd
            self.MO['ground_truth'].position.value = center
        # Simulate longer process
        if self.delay:
            sleep(0.01 * randint(0, 10))

    def close(self):

        # Shutdown message
        print("Bye!")
