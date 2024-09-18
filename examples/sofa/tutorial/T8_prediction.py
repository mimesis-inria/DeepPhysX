"""
#08 - Prediction
Launch a running session in a SOFA GUI.
"""

# Python related imports
import os

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Sofa.Pipeline.SofaPrediction import SofaPrediction
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Session related imports
from T1_environment import DummyEnvironment
from T2_network import DummyNetwork


def create_runner():
    # Create the Environment config
    env_config = SofaEnvironmentConfig(environment_class=DummyEnvironment)

    # Create the Network config
    net_config = BaseNetworkConfig(network_class=DummyNetwork)

    # Runner
    return SofaPrediction(network_config=net_config,
                          environment_config=env_config,
                          session_dir='sessions',
                          session_name='tutorial_online_training',
                          step_nb=20)


if __name__ == '__main__':

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
