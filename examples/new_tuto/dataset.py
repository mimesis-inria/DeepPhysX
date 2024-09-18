from DeepPhysX.Core.Pipelines.BaseDataGeneration import BaseDataGeneration
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig

# Session related imports
from environment import SpringEnvironment


if __name__ == '__main__':

    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=SpringEnvironment,
                                               visualizer=None,
                                               as_tcp_ip_client=False,
                                               number_of_thread=1)
    # Database configuration
    database_config = BaseDatabaseConfig(normalize=True)

    # Create DataGenerator
    data_generator = BaseDataGeneration(environment_config=environment_config,
                                        database_config=database_config,
                                        session_dir='sessions',
                                        session_name='data_generation',
                                        batch_nb=1000,
                                        batch_size=32)

    # Launch the training session
    data_generator.execute()
