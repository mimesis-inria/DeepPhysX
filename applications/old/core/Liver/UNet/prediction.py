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
from DeepPhysX.Torch.UNet.UNetConfig import UNetConfig

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download import LiverDownloader
LiverDownloader().get_session('all')
from Environment.Liver import Liver
from Environment.parameters import grid_resolution


def launch_runner():

    # Environment config
    environment_config = BaseEnvironmentConfig(environment_class=Liver,
                                               visualizer='vedo',
                                               env_kwargs={'nb_forces': 3})

    # unet config
    network_config = UNetConfig(input_size=grid_resolution,
                                nb_dims=3,
                                nb_input_channels=3,
                                nb_first_layer_channels=128,
                                nb_output_channels=3,
                                nb_steps=3,
                                two_sublayers=True,
                                border_mode='same',
                                skip_merge=False)

    # Dataset config
    database_config = BaseDatabaseConfig(normalize=True)

    # Runner
    runner = BasePrediction(network_config=network_config,
                            database_config=database_config,
                            environment_config=environment_config,
                            session_dir='sessions',
                            session_name='liver_dpx',
                            step_nb=500)
    runner.execute()


if __name__ == '__main__':

    launch_runner()
