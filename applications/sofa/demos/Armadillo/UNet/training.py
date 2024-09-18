"""
training.py
Launch the training session with a VedoVisualizer.
Use 'python3 training.py' to run the pipeline with existing samples from a Dataset (default).
Use 'python3 training.py <nb_thread>' to run the pipeline with newly created samples in Environment.
"""

# Python related imports
import os.path
import sys
import torch

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTraining import BaseTraining
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Torch.UNet.UNetConfig import UNetConfig
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from download import ArmadilloDownloader

ArmadilloDownloader().get_session('run')
from Environment.ArmadilloTraining import ArmadilloTraining
from Environment.parameters import grid_resolution

# Training parameters
nb_epochs = 200
nb_batch = 1000
batch_size = 16
lr = 1e-5


def launch_trainer(dataset_dir, nb_env):

    # Environment config
    environment_config = SofaEnvironmentConfig(environment_class=ArmadilloTraining,
                                               visualizer='vedo',
                                               number_of_thread=nb_env)

    # UNet config
    network_config = UNetConfig(lr=lr,
                                loss=torch.nn.MSELoss,
                                optimizer=torch.optim.Adam,
                                input_size=grid_resolution,
                                nb_dims=3,
                                nb_input_channels=3,
                                nb_first_layer_channels=128,
                                nb_output_channels=3,
                                nb_steps=3,
                                two_sublayers=True,
                                border_mode='same',
                                skip_merge=False, )

    # Dataset config
    database_config = BaseDatabaseConfig(existing_dir=dataset_dir,
                                         max_file_size=1,
                                         shuffle=True,
                                         normalize=True)

    # Trainer
    trainer = BaseTraining(network_config=network_config,
                           database_config=database_config,
                           environment_config=environment_config if dataset_dir is None else None,
                           session_dir='sessions',
                           session_name='armadillo_training_user',
                           epoch_nb=nb_epochs,
                           batch_nb=nb_batch,
                           batch_size=batch_size)

    # Launch the training session
    trainer.execute()


if __name__ == '__main__':

    # Define dataset
    dpx_session = 'sessions/armadillo_dpx'
    user_session = 'sessions/armadillo_data_user'
    # Take user dataset by default
    dataset = user_session if os.path.exists(user_session) else dpx_session

    # Get nb_thread options
    nb_thread = 1
    if len(sys.argv) > 1:
        dataset = None
        try:
            nb_thread = int(sys.argv[1])
        except ValueError:
            print("Script option must be an integer <nb_sample> for samples produced in Environment(s)."
                  "Without option, samples are loaded from an existing Dataset.")
            quit(0)

    # Check missing data
    session_name = 'train' if dataset is None else 'train_data'
    ArmadilloDownloader().get_session(session_name)

    # Launch pipeline
    launch_trainer(dataset, nb_thread)
