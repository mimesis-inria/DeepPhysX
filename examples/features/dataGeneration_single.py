"""
dataGeneration_single.py
Run the pipeline DataGenerator to produce a Dataset only.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseDataGeneration import BaseDataGeneration
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Visualization.VedoVisualizer import VedoVisualizer

# Session related imports
from Environment import MeanEnvironment


def launch_data_generation():

    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3

    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=MeanEnvironment,
                                               visualizer=VedoVisualizer,
                                               as_tcp_ip_client=False,
                                               env_kwargs={'constant': False,
                                                           'data_size': (nb_points, dimension),
                                                           'delay': False,
                                                           'allow_request': False})
    # Database configuration
    database_config = BaseDatabaseConfig(max_file_size=1,
                                         normalize=False)

    # Create DataGenerator
    data_generator = BaseDataGeneration(environment_config=environment_config,
                                        database_config=database_config,
                                        session_dir='sessions',
                                        session_name='data_generation',
                                        batch_nb=100,
                                        batch_size=10)

    # Launch the training session
    data_generator.execute()


if __name__ == '__main__':
    launch_data_generation()
