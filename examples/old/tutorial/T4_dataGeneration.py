"""
#04 - Data Generation
Only create a Dataset without training session.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseDataGeneration import BaseDataGeneration
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig

# Tutorial related imports
from T1_environment import DummyEnvironment


def launch_data_generation():

    # Create the Environment config
    env_config = BaseEnvironmentConfig(environment_class=DummyEnvironment,
                                       as_tcp_ip_client=True,
                                       number_of_thread=3)

    # Create the Database config
    database_config = BaseDatabaseConfig(max_file_size=1,
                                         normalize=False)

    # Create the Pipeline
    pipeline = BaseDataGeneration(environment_config=env_config,
                                  database_config=database_config,
                                  session_dir='sessions',
                                  session_name='tutorial_data_generation',
                                  new_session=True,
                                  batch_nb=20,
                                  batch_size=10)

    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_data_generation()
