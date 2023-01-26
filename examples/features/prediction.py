"""
prediction.py
Run the pipeline BaseRunner to check the predictions of the trained network.
"""

# Python related imports
import os

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BasePrediction import BasePrediction
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig


# Session imports
from Environment import MeanEnvironment


def launch_prediction(session):

    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3

    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=MeanEnvironment,
                                               visualizer='open3d',
                                               env_kwargs={'constant': False,
                                                           'data_size': [nb_points, dimension],
                                                           'delay': True,
                                                           'allow_request': False})

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                              dim_output=dimension)

    # Create DataGenerator
    trainer = BasePrediction(environment_config=environment_config,
                             network_config=network_config,
                             session_dir='sessions',
                             session_name=session,
                             step_nb=100)

    # Launch the training session
    trainer.execute()


if __name__ == '__main__':

    is_offline_session = os.path.exists(os.path.join(os.getcwd(), 'sessions/offline_training'))
    is_online_session = os.path.exists(os.path.join(os.getcwd(), 'sessions/online_training'))

    if not is_online_session and not is_offline_session:
        print("Trained Network required, 'sessions/online_training' or 'sessions/offline_training' not found. "
              "Run onlineTraining.py script first.")
        from onlineTraining import launch_training
        launch_training()
        session_name = 'online_training'
    else:
        session_name = 'online_training' if is_online_session else 'offline_training'

    launch_prediction(session_name)
