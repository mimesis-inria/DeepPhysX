from typing import Any, Dict

import numpy as np

from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Network.BaseNetwork import BaseNetwork
from DeepPhysX.Core.Network.BaseOptimization import BaseOptimization
from DeepPhysX.Core.Network.DataTransformation import DataTransformation


class NumpyNetwork(BaseNetwork):

    def __init__(self, config):
        BaseNetwork.__init__(self, config)
        self.data = None
        self.p = [np.random.randn() for _ in range(self.config.nb_parameters)]

    def forward(self, x):
        self.data = x
        output = 0
        for i in range(len(self.p)):
            output += self.p[i] * (x ** i)
        return output

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def set_device(self):
        pass

    def load_parameters(self, path):
        self.p = np.load(path)

    def get_parameters(self):
        return np.array(self.p)

    def save_parameters(self, path):
        np.save(path, self.get_parameters())

    def nb_parameters(self):
        return len(self.p)

    def transform_from_numpy(self, x, grad=True):
        return x

    def transform_to_numpy(self, x):
        return x


class NumpyOptimisation(BaseOptimization):

    def __init__(self, config):
        BaseOptimization.__init__(self, config)
        self.net = None

    def set_loss(self):
        pass

    def transform_loss(self, data):
        pass

    def compute_loss(self, prediction, ground_truth, data):
        self.loss_value = {'item': (np.square(prediction - ground_truth)).mean(),
                           'grad': (2.0 * (prediction - ground_truth)).mean()}
        return self.loss_value['item']

    def set_optimizer(self, net):
        self.net = net

    def optimize(self):
        grad = self.loss_value['grad']
        data = self.net.data
        for i in range(self.net.nb_parameters()):
            grad_p_i = (grad * (data ** i)).sum()
            self.net.p[i] -= self.lr * grad_p_i


class NumpyDataTransformation(DataTransformation):

    def __init__(self, config):
        super(NumpyDataTransformation, self).__init__(config)
        self.data_type = np.ndarray

    @DataTransformation.check_type
    def transform_before_prediction(self, data_in):
        return data_in

    @DataTransformation.check_type
    def transform_before_loss(self, data_out, data_gt=None):
        return data_out, data_gt

    @DataTransformation.check_type
    def transform_before_apply(self, data_out):
        return data_out


class NumpyNetworkConfig(BaseNetworkConfig):

    def __init__(self,
                 network_class=NumpyNetwork,
                 optimization_class=NumpyOptimisation,
                 data_transformation_class=NumpyDataTransformation,
                 network_dir=None,
                 network_name='NumpyNetwork',
                 network_type='Regression',
                 which_network=0,
                 save_each_epoch=True,
                 lr=None,
                 require_training_stuff=False,
                 nb_parameters=3):
        super(NumpyNetworkConfig, self).__init__(network_class=network_class,
                                                 optimization_class=optimization_class,
                                                 data_transformation_class=data_transformation_class,
                                                 network_dir=network_dir,
                                                 network_name=network_name,
                                                 network_type=network_type,
                                                 which_network=which_network,
                                                 save_each_epoch=save_each_epoch,
                                                 lr=lr,
                                                 require_training_stuff=require_training_stuff)
        self.network_config = self.make_config('network_config', nb_parameters=nb_parameters)
