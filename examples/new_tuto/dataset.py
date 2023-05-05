"""
dataset.py
Run the pipeline DataGenerator to produce a Dataset only.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseDataGeneration import BaseDataGeneration
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig

# Session related imports
from environment import SpringEnvironment


if __name__ == '__main__':

    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=SpringEnvironment,
                                               visualizer=None,
                                               as_tcp_ip_client=True,
                                               number_of_thread=8,
                                               simulations_per_step=5)
    # Database configuration
    database_config = BaseDatabaseConfig(max_file_size=1,
                                         normalize=True)

    # Create DataGenerator
    data_generator = BaseDataGeneration(environment_config=environment_config,
                                        database_config=database_config,
                                        session_dir='sessions',
                                        session_name='data_generation',
                                        batch_nb=500,
                                        batch_size=32)

    # Launch the training session
    data_generator.execute()
