"""
#02 - Implementing a Network
DummyNetwork: simply returns the input as output.
DummyOptimizer: nothing to optimize in our DummyNetwork.
"""

# Python related imports
from numpy import save, array

# DeepPhysX related imports
from DeepPhysX.Core.Network.BaseNetwork import BaseNetwork
from DeepPhysX.Core.Network.BaseOptimization import BaseOptimization


# Create a Network as a BaseNetwork child class
class DummyNetwork(BaseNetwork):

    def __init__(self, config):
        BaseNetwork.__init__(self, config)
        # There is no Network architecture to define in our DummyNetwork

    # MANDATORY
    def forward(self, x):
        # Return the input
        return x

    """
    The following methods should be already defined in a DeepPhysX AI package.
    This DummyNetwork has no parameters so it does not behave like a real neural network.
    """

    # MANDATORY
    def set_train(self):
        pass

    # MANDATORY
    def set_eval(self):
        pass

    # MANDATORY
    def set_device(self):
        pass

    # MANDATORY
    def load_parameters(self, path):
        pass

    # MANDATORY
    def get_parameters(self):
        pass

    # MANDATORY
    def save_parameters(self, path):
        save(path + '.npy', array([]))

    # MANDATORY
    def nb_parameters(self):
        return 0

    # MANDATORY
    def transform_from_numpy(self, x, grad=True):
        return x

    # MANDATORY
    def transform_to_numpy(self, x):
        return x


# Create an Optimization as a BaseOptimization child class
class DummyOptimization(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)

    """
    The following methods should be already defined in a DeepPhysX AI package.
    This DummyOptimization has nothing to compute.
    """

    # MANDATORY
    def set_loss(self):
        pass

    # MANDATORY
    def compute_loss(self, prediction, ground_truth, data):
        return {'loss': 0.}

    # Optional
    def transform_loss(self, data):
        pass

    # MANDATORY
    def set_optimizer(self, net):
        pass

    # MANDATORY
    def optimize(self):
        pass
