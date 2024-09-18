"""
#04 - Run Sofa Gui
Launch the Environment in a SOFA GUI.
The launched Environment should not perform requests during steps.
"""

# Python related imports
import os

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from T1_environment import DummyEnvironment


def create_environment():

    # Create the Environment config
    env_config = SofaEnvironmentConfig(environment_class=DummyEnvironment)

    # Create DummyEnvironment
    env = env_config.create_environment()
    env.create()
    env.init()
    return env


if __name__ == '__main__':

    # Create Environment
    environment = create_environment()

    # Launch Sofa GUI
    Sofa.Gui.GUIManager.Init(program_name="main", gui_name="qglviewer")
    Sofa.Gui.GUIManager.createGUI(environment.root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(environment.root)
    Sofa.Gui.GUIManager.closeGUI()

    # Delete log files
    for file in os.listdir(os.getcwd()):
        if '.ini' in file or '.log' in file:
            os.remove(file)
