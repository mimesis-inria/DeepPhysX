from os.path import join, sep, exists
from vedo import ProgressBar

from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.simulation.simulation_manager import SimulationManager
from DeepPhysX.utils.path import create_dir, get_session_dir


class DataPipeline:

    def __init__(self,
                 simulation_manager: SimulationManager,
                 database_manager: DatabaseManager,
                 new_session: bool = True,
                 session_dir: str = 'sessions',
                 session_name: str = 'data_generation',
                 batch_nb: int = 0,
                 batch_size: int = 0):
        """
        DataPipeline implements the main loop that produces data from a numerical simulation.

        :param simulation_manager: Manager for the numerical Simulation.
        :param database_manager: Manager for the Database.
        :param new_session: If True, a new repository is created for the session.
        :param session_dir: Path to the directory that contains the DeepPhysX session repositories.
        :param session_name: Name of the current session repository.
        :param batch_nb: Number of batches to produce.
        :param batch_size: Number of samples to produce per batch.
        """

        # Create a new session if required
        session_dir = get_session_dir(session_dir, new_session)
        new_session = new_session or not exists(join(session_dir, session_name))
        if new_session:
            session_name = create_dir(session_dir=session_dir,
                                      session_name=session_name).split(sep)[-1]

        # Create a DatabaseManager
        self.database_manager = database_manager
        self.database_manager.init_data_pipeline(session=join(session_dir, session_name),
                                                 new_session=new_session)

        # Create a SimulationManager
        self.simulation_manager = simulation_manager
        self.simulation_manager.init_data_pipeline(batch_size=batch_size)
        self.simulation_manager.connect_to_database(database_path=(self.database_manager.database_dir, 'dataset'),
                                                    normalize_data=self.database_manager.normalize)

        # Data generation variables
        self.__batch_nb = batch_nb
        self.__progress_bar = ProgressBar(start=0, stop=self.__batch_nb, c='orange', title="Data Generation")

        # Description
        self.__desc = {'repository': session_dir,
                       'batch_nb': self.__batch_nb,
                       'batch_size': batch_size}

    def execute(self) -> None:
        """
        Launch the data generation Pipeline.
        """

        batch_id = 0
        while batch_id < self.__batch_nb:

            lines_id = self.simulation_manager.get_data(animate=True)
            self.database_manager.add_data(data_lines=lines_id)

            batch_id += 1
            self.__progress_bar.print()

        self.database_manager.close()
        self.simulation_manager.close()

    def __str__(self):

        desc = "\n"
        desc += f"# DATA PIPELINE\n"
        desc += f"    Session repository: {self.__desc['repository']}\n"
        desc += f"    Number of batches: {self.__desc['batch_nb']}\n"
        desc += f"    Number of sample per batch: {self.__desc['batch_size']}\n"
        return desc
