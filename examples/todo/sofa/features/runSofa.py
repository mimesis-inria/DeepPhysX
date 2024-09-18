"""
runSofa.py
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
from Environment.EnvironmentSofa import EnvironmentSofa


def create_environment():

    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3

    # Create SofaEnvironment configuration
    env_config = SofaEnvironmentConfig(environment_class=EnvironmentSofa,
                                       env_kwargs={'constant': False,
                                                   'data_size': [nb_points, dimension],
                                                   'delay': False})

    # Create Environment
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

    # Close environment
    environment.close()

    # Delete log files
    for file in os.listdir(os.getcwd()):
        if '.ini' in file or '.log' in file:
            os.remove(file)
