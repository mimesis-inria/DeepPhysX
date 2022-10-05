"""
gradientDescent.py
Configure Environment so that input vector is always the same, thus one can observe how the prediction crawl toward the
ground truth.
"""

# Python related imports
from torch.nn import MSELoss
from torch.optim import Adam

from SSD.Core.Rendering.VedoVisualizer import VedoVisualizer

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig

# Session related imports
from Environment import MeanEnvironment


def launch_training():
    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3
    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=MeanEnvironment,
                                               visualizer=VedoVisualizer,
                                               param_dict={'constant': True,
                                                           'data_size': [nb_points, dimension],
                                                           'sleep': False,
                                                           'allow_requests': True},
                                               as_tcp_ip_client=False)
    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(loss=MSELoss,
                              lr=1e-3,
                              optimizer=Adam,
                              dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                              dim_output=dimension)
    # Dataset configuration
    dataset_config = BaseDatasetConfig(normalize=False)
    # Create Trainer
    trainer = BaseTrainer(session_dir='sessions',
                          session_name='gradient_descent',
                          environment_config=environment_config,
                          dataset_config=dataset_config,
                          network_config=network_config,
                          nb_epochs=1,
                          nb_batches=200,
                          batch_size=1,
                          debug=True)
    # Launch the training session
    trainer.execute()


if __name__ == '__main__':
    launch_training()
