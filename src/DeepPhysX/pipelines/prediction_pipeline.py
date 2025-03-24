from typing import Optional
from os.path import join, exists

from DeepPhysX.simulation.simulation_manager import SimulationManager
from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.networks.network_manager import NetworkManager
from DeepPhysX.utils.path import get_session_dir

try:
    import Sofa
    import Sofa.Gui
except ImportError:
    pass


class PredictionPipeline:

    def __init__(self,
                 network_manager: NetworkManager,
                 simulation_manager: SimulationManager,
                 database_manager: Optional[DatabaseManager] = None,
                 session_dir: str = 'sessions',
                 session_name: str = 'training',
                 step_nb: int = -1,
                 record: bool = False):
        """
        PredictionPipeline implements the main loop that uses a Network predictions in the numerical Simulation.

        :param network_manager: Manager for the neural Network.
        :param simulation_manager: Manager for the numerical Simulation.
        :param database_manager: Manager for the Database.
        :param session_dir: Path to the directory that contains the DeepPhysX session repositories.
        :param session_name: Name of the current session repository.
        :param step_nb: Number of step of predictions tu run (set to -1 for infinite).
        :param record: Save the produced samples in the Database.
        """

        # Define the session repository
        self.session_dir = get_session_dir(session_dir, False)
        if not exists(path := join(self.session_dir, session_name)):
            raise ValueError(f"[{self.__class__.__name__}] The following directory does not exist: {path}")

        # Create a DatabaseManager
        self.database_manager = database_manager
        self.database_manager.init_prediction_pipeline(session=join(self.session_dir, session_name),
                                                       produce_data=record)

        # Create a SimulationManager
        self.simulation_manager = simulation_manager
        self.simulation_manager.init_prediction_pipeline()
        self.simulation_manager.connect_to_database(database_path=(self.database_manager.database_dir, 'dataset'),
                                                    normalize_data=self.database_manager.normalize)

        # Create a NetworkManager
        self.network_manager = network_manager
        self.network_manager.init_prediction_pipeline(session=path)
        self.network_manager.connect_to_database(database_path=(self.database_manager.database_dir, 'dataset'),
                                                 normalize_data=self.database_manager.normalize)
        self.network_manager.link_clients(1)

        self.simulation_manager.connect_to_network_manager(network_manager=self.network_manager)
        self.simulation_manager.connect_to_database_manager(database_manager=self.database_manager)

        # Prediction variables
        self.step_nb = step_nb
        self.step_id = 0
        self.produce_data = record

    def execute(self) -> None:
        """
        Launch the prediction Pipeline.
        """

        while self.step_id < self.step_nb if self.step_nb > 0 else True:
            self.step_id += 1

            # Get data from Dataset
            if self.simulation_manager.load_samples:
                self.data_lines = self.database_manager.get_data(batch_size=1)
                self.simulation_manager.dispatch_batch(data_lines=self.data_lines,
                                                       animate=True,
                                                       request_prediction=True,
                                                       save_data=self.produce_data)
            # Get data from Environment
            else:
                self.data_lines = self.simulation_manager.get_data(animate=True,
                                                                   request_prediction=True,
                                                                   save_data=self.produce_data)
                if self.produce_data:
                    self.database_manager.add_data(self.data_lines)

            # Check viewer closed event
            if self.step_nb < 0 and not self.simulation_manager.is_viewer_open():
                break

        # Close managers
        self.simulation_manager.close()
        self.database_manager.close()
        self.network_manager.close()

    def __str__(self):

        description = "\n"
        description += f"# {self.__class__.__name__}\n"
        description += f"    Session repository: {self.session_dir}\n"
        description += f"   Number of step: {self.step_nb}\n"
        return description


class SofaPredictionPipeline(Sofa.Core.Controller, PredictionPipeline):

    def __init__(self,
                 network_manager: NetworkManager,
                 simulation_manager: SimulationManager,
                 database_manager: Optional[DatabaseManager] = None,
                 session_dir: str = 'sessions',
                 session_name: str = 'training',
                 step_nb: int = -1,
                 record: bool = False,
                 *args, **kwargs):
        """
        SofaPredictionPipeline allows to run the main prediction loop in a SOFA GUI.

        :param network_manager: Manager for the neural Network.
        :param simulation_manager: Manager for the numerical Simulation.
        :param database_manager: Manager for the Database.
        :param session_dir: Path to the directory that contains the DeepPhysX session repositories.
        :param session_name: Name of the current session repository.
        :param step_nb: Number of step of predictions tu run (set to -1 for infinite).
        :param record: Save the produced samples in the Database.
        """

        Sofa.Core.Controller.__init__(self, name='DPX_Pipeline', *args, **kwargs)
        simulation_manager.use_viewer = False
        PredictionPipeline.__init__(self,
                                    network_manager=network_manager,
                                    simulation_manager=simulation_manager,
                                    database_manager=database_manager,
                                    session_dir=session_dir,
                                    session_name=session_name,
                                    step_nb=step_nb,
                                    record=record)

        self.root: Sofa.Core.Node = self.simulation_manager.simulation_controller.simulation.root
        self.root.addObject(self)

    def execute(self) -> None:
        """
        Launch the SOFA GUI and the prediction Pipeline.
        """

        # Launch the GUI
        Sofa.Gui.GUIManager.Init("main", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(self.root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(self.root)
        Sofa.Gui.GUIManager.closeGUI()

        # Close managers
        self.simulation_manager.close()
        self.database_manager.close()
        self.network_manager.close()


    def onAnimateEndEvent(self, event):
        """
        Event callback for SOFA.
        """

        # Get and apply the prediction at each time step.
        if self.step_id < self.step_nb if self.step_nb > 0 else True:
            self.step_id += 1
            self.simulation_manager.get_data(animate=False,
                                             request_prediction=True,
                                             save_data=self.produce_data)
