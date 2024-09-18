"""
prediction.py
Run the pipeline BaseRunner to check the predictions of the trained networks.
"""

# Python related imports
import os

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX.Sofa.Pipeline.SofaPrediction import SofaPrediction
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig

# Session imports
from Environment.EnvironmentPrediction import MeanEnvironmentPrediction


def create_runner(session):

    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3

    # Environment configuration
    environment_config = SofaEnvironmentConfig(environment_class=MeanEnvironmentPrediction,
                                               env_kwargs={'constant': False,
                                                           'data_size': [nb_points, dimension],
                                                           'delay': True})

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                              dim_output=dimension)

    # Create SofaRunner
    return SofaPrediction(environment_config=environment_config,
                          network_config=network_config,
                          session_dir='sessions',
                          session_name=session)


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

    # Create SOFA runner
    runner = create_runner(session_name)

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
