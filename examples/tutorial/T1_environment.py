"""
#01 - Implementing an Environment
DummyEnvironment: simply set the current step index as training data.
"""

# Python related imports
from numpy import array

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment


# Create an Environment as a BaseEnvironment child class
class DummyEnvironment(BaseEnvironment):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 instance_id=1, 
                 number_of_instances=1, 
                 as_tcp_ip_client=True,  
                 environment_manager=None):

        BaseEnvironment.__init__(self, 
                                 ip_address=ip_address, 
                                 port=port,
                                 instance_id=instance_id, 
                                 number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client, 
                                 environment_manager=environment_manager)

        self.nb_step = 0
        self.increment = 0

    """
    INITIALIZING ENVIRONMENT - Methods will be automatically called it this order:
       - recv_parameters: Receive a dictionary of parameters that can be set in EnvironmentConfig
       - create: Create the Environment
       - init: Initialize the Environment if required
       - send_parameters: Same as recv_parameters, Environment can send back a set of parameters if required
       - send_visualization: Send initial visualization data (see Example/CORE/Features to add visualization data)
    """

    # Optional
    def recv_parameters(self, param_dict):
        # Set data size
        self.increment = param_dict['increment'] if 'increment' in param_dict else 1

    # MANDATORY
    def create(self):
        # Nothing to create in our DummyEnvironment
        pass

    # Optional
    def init(self):
        # Nothing to init in our DummyEnvironment
        pass

    # Optional
    def send_parameters(self):
        # Nothing to send back
        return {}

    # Optional
    def send_visualization(self):
        # Nothing to visualize (see Example/CORE/Features to add visualization data)
        return {}

    """
    ENVIRONMENT BEHAVIOR - Methods will be automatically called at each simulation step in this order:
       - step: Transition in simulation state, compute training data
       - check_sample: Check if current data sample is usable
       - apply_prediction: Network prediction will be applied in Environment
       - close: Shutdown procedure when data producer is no longer used
     Some requests can be performed here:
       - get_prediction: Get an online prediction from an input array
       - update_visualization: Send updated visualization data (see Example/CORE/Features to update viewer data)
    """

    # MANDATORY
    async def step(self):
        # Setting (and sending) training data
        self.set_training_data(input_array=array([self.nb_step]),
                               output_array=array([self.nb_step]))
        self.nb_step += self.increment
        # Other data fields can be filled:
        #   - set_loss_data: Define an additional data to compute loss value (see Optimization.transform_loss)
        #   - set_additional_dataset: Add a field to the dataset

    # Optional
    def check_sample(self, check_input=True, check_output=True):
        # Nothing to check in our DummyEnvironment
        return True

    # Optional
    def apply_prediction(self, prediction):
        # Nothing to apply in our DummyEnvironment
        print(f"Prediction at step {self.nb_step - 1} = {prediction}")

    # Optional
    def close(self):
        # Shutdown procedure
        print("Bye!")
