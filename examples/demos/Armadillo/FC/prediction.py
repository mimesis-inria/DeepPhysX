"""
prediction.py
Launch the prediction session in a VedoVisualizer.
"""

# Python related imports
import os
import sys

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BasePrediction import BasePrediction
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download import ArmadilloDownloader
# ArmadilloDownloader().get_session('all')
from Environment.Armadillo import Armadillo
from Environment.parameters import p_model


def launch_runner():

    # Environment config
    environment_config = BaseEnvironmentConfig(environment_class=Armadillo,
                                               visualizer='vedo')

    # FC config
    nb_hidden_layers = 3
    nb_neurons = p_model.nb_nodes_mesh * 3
    nb_final_neurons = p_model.nb_nodes_grid * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers)] + [nb_final_neurons]
    network_config = FCConfig(dim_layers=layers_dim,
                              dim_output=3,
                              biases=True)

    # Dataset config
    database_config = BaseDatabaseConfig(normalize=True)

    # Runner
    runner = BasePrediction(network_config=network_config,
                            database_config=database_config,
                            environment_config=environment_config,
                            session_dir='sessions',
                            session_name='armadillo_dpx',
                            step_nb=500)
    runner.execute()


if __name__ == '__main__':

    launch_runner()
