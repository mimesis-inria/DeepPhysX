"""
training_pipeline.py
Launch the training session with a VedoVisualizer.
Use 'python3 training_pipeline.py' to run the pipeline with existing samples from a Dataset (default).
Use 'python3 training_pipeline.py <nb_thread>' to run the pipeline with newly created samples in Environment.
"""

# Python related imports
import os.path
import sys
from torch.optim import Adam
from torch.nn import MSELoss
from numpy import multiply

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTraining import BaseTraining
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from download import LiverDownloader

LiverDownloader().get_session('run')
from Environment.LiverTraining import LiverTraining

# Training parameters
nb_epochs = 400
nb_batch = 500
batch_size = 32
lr = 1e-5


def launch_trainer(dataset_dir, nb_env):

    # Environment config
    environment_config = SofaEnvironmentConfig(environment_class=LiverTraining,
                                               visualizer='vedo',
                                               number_of_thread=nb_env)

    # Get the data size
    env = environment_config.create_environment()
    env.create()
    env.init()
    input_size, output_size = env.input_size, env.output_size
    env.close()
    del env

    # mlp config
    nb_hidden_layers = 3
    nb_neurons = multiply(*input_size)
    nb_final_neurons = multiply(*output_size)
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers)] + [nb_final_neurons]
    network_config = FCConfig(lr=lr,
                              loss=MSELoss,
                              optimizer=Adam,
                              dim_output=3,
                              dim_layers=layers_dim,
                              biases=True)

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
                           session_name='liver_training_user',
                           epoch_nb=nb_epochs,
                           batch_nb=nb_batch,
                           batch_size=batch_size,
                           debug=True)

    # Launch the training session
    trainer.execute()


if __name__ == '__main__':

    # Define dataset
    dpx_session = 'sessions/liver_dpx'
    user_session = 'sessions/liver_data_user'
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
                  "By default, samples are loaded from an existing Dataset.")
            quit(0)

    # Check missing data
    session_name = 'train' if dataset is None else 'train_data'
    LiverDownloader().get_session(session_name)

    # Launch pipeline
    launch_trainer(dataset, nb_thread)
