"""
#01 - Implementing an Environment
DummyEnvironment: simply set the current step index as training data.
"""

# Python related imports
from numpy import array, ndarray

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment


# Create an Environment as a BaseEnvironment child class
class DummyEnvironment(BaseEnvironment):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1, 
                 instance_nb=1,
                 visualization_db=None):

        BaseEnvironment.__init__(self, 
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id, 
                                 instance_nb=instance_nb,
                                 visualization_db=visualization_db)

        self.step_nb: int = 0

    """
    INITIALIZING ENVIRONMENT - Methods will be automatically called it this order:
       - create: Create the Environment
       - init: Initialize the Environment if required
       - init_database: Define the training data fields
       - init_visualization: Define and send initial visualization data
    """

    # MANDATORY
    def create(self):

        # Nothing to create in our DummyEnvironment
        pass

    # Optional
    def init(self):

        # Nothing to init in our DummyEnvironment
        pass

    # MANDATORY
    def init_database(self):

        # Define the fields of the training Database
        self.define_training_fields(fields=[('input', ndarray), ('ground_truth', ndarray)])

    # Optional
    def init_visualization(self):

        # Nothing to visualize
        pass

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
        self.step_nb += 1
        self.set_training_data(input=array([self.step_nb]),
                               ground_truth=array([self.step_nb]))

    # Optional
    def check_sample(self):

        # Nothing to check in our DummyEnvironment
        return True

    # Optional
    def apply_prediction(self, prediction):

        # Nothing to apply in our DummyEnvironment
        print(f"Prediction at step {self.step_nb} = {prediction}")

    # Optional
    def close(self):

        # Shutdown procedure
        print("Bye!")
