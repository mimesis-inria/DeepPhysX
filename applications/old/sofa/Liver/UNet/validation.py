"""
validation.py
Launch the prediction session in a SOFA GUI. Compare the two models.
Use 'python3 validation.py' to run the pipeline with existing samples from a Dataset (default).
Use 'python3 validation.py -e' to run the pipeline with newly created samples in Environment.
"""

# Python related imports
import os
import sys

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX.Sofa.Pipeline.SofaPrediction import SofaPrediction
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig
from DeepPhysX.Torch.UNet.UNetConfig import UNetConfig

# Session related imports
from download import LiverDownloader

LiverDownloader().get_session('run')
from Environment.LiverValidation import LiverValidation
from Environment.parameters import grid_resolution


def create_runner(dataset_dir):

    # Environment config
    environment_config = SofaEnvironmentConfig(environment_class=LiverValidation,
                                               load_samples=dataset_dir is not None,
                                               env_kwargs={'compute_sample': dataset_dir is None})

    # unet config
    network_config = UNetConfig(save_each_epoch=True,
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
                                         shuffle=True,
                                         normalize=True,
                                         mode=None if dataset_dir is None else 'validation')

    # Define trained networks session
    dpx_session = 'liver_dpx'
    user_session = 'liver_training_user'
    # Take user session by default
    session_name = user_session if os.path.exists('sessions/' + user_session) else dpx_session

    # Runner
    return SofaPrediction(environment_config=environment_config,
                          network_config=network_config,
                          database_config=database_config,
                          session_dir='sessions',
                          session_name=session_name,
                          nb_steps=500)


if __name__ == '__main__':

    # Define dataset
    dpx_session = 'sessions/liver_dpx'
    user_session = 'sessions/liver_data_user'
    # Take user dataset by default
    dataset = user_session if os.path.exists(user_session) else dpx_session

    # Get option
    if len(sys.argv) > 1:
        # Check script option
        if sys.argv[1] != '-e':
            print("Script option must be '-e' for samples produced in Environment(s)."
                  "By default, samples are loaded from an existing Dataset.")
            quit(0)
        dataset = None

    # Check missing data
    session_name = 'valid' if dataset is None else 'valid_data'
    LiverDownloader().get_session(session_name)

    # Create SOFA runner
    runner = create_runner(dataset)

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
