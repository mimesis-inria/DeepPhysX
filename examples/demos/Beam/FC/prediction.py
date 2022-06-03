"""
prediction.py
Launch the prediction session in a VedoVisualizer.
"""

# Python related imports
import os
import sys

# DeepPhysX related imports
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX.Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX.Torch.FC.FCConfig import FCConfig

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download import BeamDownloader
BeamDownloader().get_session('valid_data')
from Environment.Beam import Beam
from Environment.parameters import p_model


def launch_runner():

    # Environment config
    env_config = BaseEnvironmentConfig(environment_class=Beam,
                                       visualizer=VedoVisualizer,
                                       as_tcp_ip_client=False,
                                       param_dict={'compute_sample': True})

    # FC config
    nb_hidden_layers = 3
    nb_neurons = p_model.nb_nodes * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers)] + [nb_neurons]
    net_config = FCConfig(network_name='beam_FC',
                          dim_output=3,
                          dim_layers=layers_dim,
                          biases=True)

    # Dataset config
    dataset_config = BaseDatasetConfig(normalize=True,
                                       dataset_dir='sessions/beam_dpx',
                                       use_mode='Validation')

    # Runner
    runner = BaseRunner(session_dir='sessions',
                        session_name='beam_dpx',
                        dataset_config=dataset_config,
                        environment_config=env_config,
                        network_config=net_config,
                        nb_steps=500)
    runner.execute()
    runner.close()


if __name__ == '__main__':

    launch_runner()
