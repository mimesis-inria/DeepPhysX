"""
onlineTraining.py
Run the pipeline BaseTrainer to create a new Dataset while training the Network.
"""

# Python related imports
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTraining import BaseTraining
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig

# Session imports
from Environment.EnvironmentTraining import EnvironmentTraining


def launch_training():

    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3

    # Environment configuration
    environment_config = SofaEnvironmentConfig(environment_class=EnvironmentTraining,
                                               visualizer='vedo',
                                               as_tcp_ip_client=True,
                                               number_of_thread=4,
                                               env_kwargs={'constant': False,
                                                           'data_size': [nb_points, dimension],
                                                           'delay': False})

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(lr=1e-3,
                              loss=MSELoss,
                              optimizer=Adam,
                              dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                              dim_output=dimension)
    # Dataset configuration
    database_config = BaseDatabaseConfig(max_file_size=1,
                                         shuffle=True,
                                         normalize=False)

    # Create DataGenerator
    trainer = BaseTraining(network_config=network_config,
                           database_config=database_config,
                           environment_config=environment_config,
                           session_dir='sessions',
                           session_name='online_training',
                           epoch_nb=5,
                           batch_nb=100,
                           batch_size=10)

    # Launch the training session
    trainer.execute()


if __name__ == '__main__':
    launch_training()
