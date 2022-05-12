"""
prediction.py
Launch the prediction session in a VedoVisualizer.
"""

# Python related imports
import os
import sys

# DeepPhysX related imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX_Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX_Torch.UNet.UNetConfig import UNetConfig

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download import BeamDownloader
BeamDownloader().get_session('valid_data')
from Environment.Beam import Beam
from Environment.parameters import grid_resolution


def launch_runner():

    # Environment config
    env_config = BaseEnvironmentConfig(environment_class=Beam,
                                       visualizer=VedoVisualizer,
                                       as_tcp_ip_client=False,
                                       param_dict={'compute_sample': True})

    # UNet config
    net_config = UNetConfig(network_name='beam_UNet',
                            input_size=grid_resolution.tolist(),
                            nb_dims=3,
                            nb_input_channels=3,
                            nb_first_layer_channels=128,
                            nb_output_channels=3,
                            nb_steps=3,
                            two_sublayers=True,
                            border_mode='same',
                            skip_merge=False)

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
                        nb_steps=0)
    runner.execute()
    runner.close()


if __name__ == '__main__':

    launch_runner()
