"""
interactive.py
Launch the interactive prediction session in a VedoVisualizer.
"""

# Python related imports
import os
import sys

# DeepPhysX related imports
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX.Torch.FC.FCConfig import FCConfig

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download import ArmadilloDownloader
ArmadilloDownloader().get_session('predict')
from Environment.ArmadilloInteractive import Armadillo
from Environment.parameters import p_model


def launch_runner():

    # Environment config
    env_config = BaseEnvironmentConfig(environment_class=Armadillo,
                                       as_tcp_ip_client=False)

    # FC config
    nb_hidden_layers = 3
    nb_neurons = p_model.nb_nodes_mesh * 3
    nb_final_neurons = p_model.nb_nodes_grid * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers)] + [nb_final_neurons]
    net_config = FCConfig(network_name='armadillo_FC',
                          dim_output=3,
                          dim_layers=layers_dim,
                          biases=True)

    # Dataset config
    dataset_config = BaseDatasetConfig()

    # Runner
    runner = BaseRunner(session_dir='sessions',
                        session_name='armadillo_dpx',
                        dataset_config=dataset_config,
                        environment_config=env_config,
                        network_config=net_config,
                        nb_steps=1)
    runner.execute()
    runner.close()


if __name__ == '__main__':

    launch_runner()
