from os import listdir, remove

import Sofa.Gui

from simulations.beam_simulation import BeamSimulation


# Create the simulation
simu = BeamSimulation()
simu.create()
Sofa.Simulation.initRoot(simu.root)

# Launch the Sofa Gui
Sofa.Gui.GUIManager.Init(program_name="main", gui_name="qglviewer")
Sofa.Gui.GUIManager.createGUI(simu.root, __file__)
Sofa.Gui.GUIManager.SetDimension(1200, 800)
Sofa.Gui.GUIManager.MainLoop(simu.root)
Sofa.Gui.GUIManager.closeGUI()

# Delete log files
for file in listdir():
    if file.endswith('.ini') or file.endswith('.log'):
        remove(file)
