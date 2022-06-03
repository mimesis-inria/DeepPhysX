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
from DeepPhysX.Torch.UNet.UNetConfig import UNetConfig

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download import ArmadilloDownloader
ArmadilloDownloader().get_session('valid_data')
from Environment.Armadillo import Armadillo
from Environment.parameters import grid_resolution


def launch_runner():

    # Environment config
    env_config = BaseEnvironmentConfig(environment_class=Armadillo,
                                       visualizer=VedoVisualizer,
                                       as_tcp_ip_client=False,
                                       param_dict={'compute_sample': True})

    # UNet config
    net_config = UNetConfig(network_name='armadillo_UNet',
                            input_size=grid_resolution,
                            nb_dims=3,
                            nb_input_channels=3,
                            nb_first_layer_channels=128,
                            nb_output_channels=3,
                            nb_steps=3,
                            two_sublayers=True,
                            border_mode='same',
                            skip_merge=False)

    # Dataset config
    dataset_config = BaseDatasetConfig(dataset_dir='sessions/armadillo_dpx',
                                       normalize=True,
                                       use_mode='Validation')
    # Runner
    runner = BaseRunner(session_dir='sessions',
                        session_name='armadillo_dpx',
                        dataset_config=dataset_config,
                        environment_config=env_config,
                        network_config=net_config,
                        nb_steps=0)
    runner.execute()
    runner.close()


if __name__ == '__main__':

    launch_runner()
