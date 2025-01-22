from os.path import join, sep, exists
from vedo import ProgressBar

from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.simulation.simulation_manager import SimulationManager, SimulationConfig
from DeepPhysX.utils.path import create_dir, get_session_dir


class DataPipeline:

    def __init__(self,
                 simulation_config: SimulationConfig,
                 database_manager: DatabaseManager,
                 new_session: bool = True,
                 session_dir: str = 'sessions',
                 session_name: str = 'data_generation',
                 batch_nb: int = 0,
                 batch_size: int = 0):
        """
        """

        # Create a new session if required
        self.session_dir = get_session_dir(session_dir, new_session)
        self.new_session = new_session or not exists(join(self.session_dir, session_name))
        if self.new_session:
            session_name = create_dir(session_dir=self.session_dir,
                                      session_name=session_name).split(sep)[-1]

        # Create a DatabaseManager
        self.database_manager = database_manager
        # self.database_manager = DatabaseManager(config=database_config,
        #                                         session=join(self.session_dir, session_name))
        self.database_manager.init_data_pipeline(session=join(self.session_dir, session_name),
                                                 new_session=self.new_session)

        # Create a SimulationManager
        self.simulation_manager = SimulationManager(config=simulation_config,
                                                    pipeline='data_generation',
                                                    session=join(self.session_dir, session_name),
                                                    produce_data=True,
                                                    batch_size=batch_size)
        self.simulation_manager.connect_to_database(database_path=self.database_manager.get_database_path(),
                                                    normalize_data=self.database_manager.normalize)

        # Data generation variables
        self.batch_nb: int = batch_nb
        self.batch_id: int = 0
        self.batch_size = batch_size
        self.progress_bar = ProgressBar(start=0, stop=self.batch_nb, c='orange', title="Data Generation")

    def execute(self) -> None:
        """
        Launch the data generation Pipeline.
        """

        while self.batch_id < self.batch_nb:

            lines_id = self.simulation_manager.get_data(animate=True)
            self.database_manager.add_data(data_lines=lines_id)

            self.batch_id += 1
            self.progress_bar.print()

        self.database_manager.close()
        self.simulation_manager.close()

    def __str__(self):

        description = "\n"
        description += f"# {self.__class__.__name__}\n"
        description += f"    Session repository: {self.session_dir}\n"
        description += f"    Number of batches: {self.batch_nb}\n"
        description += f"    Number of sample per batch: {self.batch_size}\n"
        return description
