"""
dataset.py
Run the data generation session to produce a Dataset only.
Use 'python3 dataset.py' to produce training Dataset (default).
Use 'python3 dataset.py -v' to produce validation Dataset.
"""

# Python related imports
import os
import sys

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseDataGeneration import BaseDataGeneration
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Session related imports
from download import LiverDownloader
LiverDownloader().get_session('run')
from Environment.LiverTraining import LiverTraining

# Dataset parameters
nb_batches = {'training': 500, 'validation': 50}
batch_size = {'training': 32,  'validation': 10}


def launch_data_generation(dataset_dir, dataset_mode):

    # Environment configuration
    environment_config = SofaEnvironmentConfig(environment_class=LiverTraining,
                                               visualizer='vedo',
                                               as_tcp_ip_client=True,
                                               number_of_thread=4)

    # Dataset configuration
    database_config = BaseDatabaseConfig(existing_dir=dataset_dir,
                                         max_file_size=1,
                                         mode=dataset_mode,
                                         normalize=True)

    # Create DataGenerator
    data_generator = BaseDataGeneration(environment_config=environment_config,
                                        database_config=database_config,
                                        session_dir='sessions',
                                        session_name='liver_data_user',
                                        batch_nb=nb_batches[dataset_mode],
                                        batch_size=batch_size[dataset_mode])

    # Launch the data generation session
    data_generator.execute()


if __name__ == '__main__':

    # Define dataset
    user_session = 'sessions/liver_data_user'
    dataset = user_session if os.path.exists(user_session) else None

    # Get dataset mode
    mode = 'training'
    if len(sys.argv) > 1:
        if sys.argv[1] != '-v':
            print("Script option must be '-v' to produce validation dataset."
                  "By default, training dataset is produced.")
            quit(0)
        mode = 'validation'

    # Launch pipeline
    launch_data_generation(dataset, mode)
