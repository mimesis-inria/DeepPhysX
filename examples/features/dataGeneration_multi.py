"""
dataGeneration_multi.py
How to run several Clients in parallel to produce data faster.
For low-level application like this one, managing several subprocesses gives fewer advantages than running a single
Environment, so Environment add a random sleep to simulate longer computations.
"""

# Python related imports
from time import time

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseDataGenerator import BaseDataGenerator
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig

# Session related imports
from Environment import MeanEnvironment


def launch_data_generation(use_tcp_ip):
    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3
    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=MeanEnvironment,
                                               param_dict={'constant': False,
                                                           'data_size': [nb_points, dimension],
                                                           'sleep': True,
                                                           'allow_requests': False},
                                               as_tcp_ip_client=use_tcp_ip,
                                               number_of_thread=10)
    # Dataset configuration
    dataset_config = BaseDatasetConfig(normalize=False)
    # Create DataGenerator
    data_generator = BaseDataGenerator(session_name='sessions/data_generation_compare',
                                       environment_config=environment_config,
                                       dataset_config=dataset_config,
                                       nb_batches=20,
                                       batch_size=10)
    # Launch the training session
    start_time = time()
    data_generator.execute()
    return time() - start_time


if __name__ == '__main__':
    # Run single process
    single_process_time = launch_data_generation(use_tcp_ip=False)
    # Run multiprocess
    multi_process_time = launch_data_generation(use_tcp_ip=True)
    # Show results
    print(f"\nSINGLE PROCESS VS MULTIPROCESS"
          f"\n    Single process elapsed time: {round(single_process_time, 2)}s"
          f"\n    Multiprocess elapsed time:   {round(multi_process_time, 2)}s")
