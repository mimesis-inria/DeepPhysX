"""
gradientDescent.py
Configure Environment so that input vector is always the same, thus one can observe how the prediction crawl toward the
ground truth.
"""

# Python related imports
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTraining import BaseTraining
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Visualization.VedoVisualizer import VedoVisualizer
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
                                               as_tcp_ip_client=True,
                                               number_of_thread=1,
                                               env_kwargs={'constant': True,
                                                           'data_size': [nb_points, dimension],
                                                           'delay': False,
                                                           'allow_request': True})

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(lr=1e-3,
                              loss=MSELoss,
                              optimizer=Adam,
                              dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                              dim_output=dimension)

    # Database configuration
    database_config = BaseDatabaseConfig(normalize=False)

    # Create Trainer
    trainer = BaseTraining(network_config=network_config,
                           environment_config=environment_config,
                           database_config=database_config,
                           session_dir='sessions',
                           session_name='gradient_descent',
                           epoch_nb=1,
                           batch_nb=200,
                           batch_size=1,
                           debug=True)

    # Launch the training session
    trainer.execute()


if __name__ == '__main__':
    launch_training()
