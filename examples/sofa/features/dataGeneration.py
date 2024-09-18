"""
dataGeneration.py
Run the pipeline DataGenerator to produce a Dataset only.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseDataGeneration import BaseDataGeneration
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Session related imports
from Environment.EnvironmentDataset import EnvironmentDataset


def launch_data_generation():

    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3

    # Environment configuration
    environment_config = SofaEnvironmentConfig(environment_class=EnvironmentDataset,
                                               visualizer='vedo',
                                               as_tcp_ip_client=True,
                                               number_of_thread=4,
                                               env_kwargs={'constant': False,
                                                           'data_size': [nb_points, dimension],
                                                           'delay': False})
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
