"""
prediction.py
Launch the prediction session in a SOFA GUI with only predictions of the network.
Use 'python3 prediction.py' to render predictions in a SOFA GUI (default).
Use 'python3 validation.py -v' to render predictions with Vedo.
"""

# Python related imports
import os
import sys

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Pipelines.BasePrediction import BasePrediction
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig
from DeepPhysX.Sofa.Pipeline.SofaPrediction import SofaPrediction
from DeepPhysX.Torch.UNet.UNetConfig import UNetConfig

# Session related imports
from download import BeamDownloader
from Environment.BeamPrediction import BeamPrediction
from Environment.parameters import grid_resolution


def create_runner(visualizer=False):

    # Environment config
    environment_config = SofaEnvironmentConfig(environment_class=BeamPrediction,
                                               visualizer='vedo' if visualizer else None,
                                               env_kwargs={'visualizer': visualizer})

    # UNet config
    network_config = UNetConfig(input_size=grid_resolution.tolist(),
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

    # Define trained network session
    dpx_session = 'beam_dpx'
    user_session = 'beam_training_user'
    # Take user session by default
    session_name = user_session if os.path.exists('sessions/' + user_session) else dpx_session

    # Runner
    if visualizer:
        return BasePrediction(network_config=network_config,
                              database_config=database_config,
                              environment_config=environment_config,
                              session_dir='sessions',
                              session_name=session_name,
                              step_nb=100)
    else:
        return SofaPrediction(network_config=network_config,
                              database_config=database_config,
                              environment_config=environment_config,
                              session_dir='sessions',
                              session_name=session_name,
                              step_nb=-1)


if __name__ == '__main__':

    # Check missing data
    BeamDownloader().get_session('predict')

    # Get option
    visualizer = False
    if len(sys.argv) > 1:
        # Check script option
        if sys.argv[1] != '-v':
            print("Script option must be '-v' to visualize predictions in a Vedo window."
                  "By default, prediction are rendered in a SOFA GUI.")
            quit(0)
        visualizer = True

    if visualizer:

        # Create and launch runner
        runner = create_runner(visualizer)
        runner.execute()

    else:

        # Create SOFA runner
        runner = create_runner()

        # Launch SOFA GUI
        Sofa.Gui.GUIManager.Init("main", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(runner.root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(runner.root)
        Sofa.Gui.GUIManager.closeGUI()

        # Manually close the runner (security if stuff like additional dataset need to be saved)
        runner.close()

        # Delete unwanted files
        for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
            if '.ini' in file or '.log' in file:
                os.remove(file)
