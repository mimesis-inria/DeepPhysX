"""
#02 - Implementing a network
DummyNetwork: simply returns the input as output.
DummyOptimizer: nothing to optimize in our DummyNetwork.
"""

# Python related imports
from numpy import save, array, ndarray

# DeepPhysX related imports
from DeepPhysX.Core.Network.BaseNetwork import BaseNetwork
from DeepPhysX.Core.Network.BaseOptimization import BaseOptimization
from DeepPhysX.Core.Network.BaseTransformation import BaseTransformation


# Create a network as a BaseNetwork child class
class DummyNetwork(BaseNetwork):

    def __init__(self, config):

        BaseNetwork.__init__(self, config)
        # There is no network architecture to define in our DummyNetwork

    # MANDATORY
    def forward(self, input_data):

        # Return the input
        return input_data

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
    def compute_loss(self, data_pred, data_opt):
        return {'loss': 0.}

    # Optional
    def transform_loss(self, data_opt):
        pass

    # MANDATORY
    def set_optimizer(self, net):
        pass

    # MANDATORY
    def optimize(self):
        pass


# Create a BaseTransformation as a BaseTransformation child class
class DummyTransformation(BaseTransformation):

    def __init__(self, config):

        BaseTransformation.__init__(self, config)
        self.data_type = ndarray

    def transform_before_prediction(self, data_net):

        # Do not transform the network data
        return data_net

    def transform_before_loss(self, data_pred, data_opt=None):

        # Do not transform the Prediction data and the Optimizer data
        return data_pred, data_opt

    def transform_before_apply(self, data_pred):

        # Do not transform Prediction data
        return data_pred
