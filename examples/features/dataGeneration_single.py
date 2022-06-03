"""
dataGeneration_single.py
Run the pipeline DataGenerator to produce a Dataset only.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseDataGenerator import BaseDataGenerator
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig

# Session related imports
from Environment import MeanEnvironment


def launch_data_generation():
    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3
    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=MeanEnvironment,
                                               param_dict={'constant': False,
                                                           'data_size': [nb_points, dimension],
                                                           'sleep': False,
                                                           'allow_requests': False},
                                               as_tcp_ip_client=False)
    # Dataset configuration
    dataset_config = BaseDatasetConfig(normalize=False)
    # Create DataGenerator
    data_generator = BaseDataGenerator(session_name='sessions/data_generation',
                                       environment_config=environment_config,
                                       dataset_config=dataset_config,
                                       nb_batches=500,
                                       batch_size=10)
    # Launch the training session
    data_generator.execute()


if __name__ == '__main__':
    launch_data_generation()
