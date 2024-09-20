from os.path import join, sep, exists
from vedo import ProgressBar

from DeepPhysX.pipelines.core.base_pipeline import BasePipeline
from DeepPhysX.database.database_manager import DatabaseManager, DatabaseConfig, DatabaseHandler
from DeepPhysX.simulation.core.environment_manager import EnvironmentManager, BaseEnvironmentConfig
from DeepPhysX.utils.path import create_dir


class DataGeneration(BasePipeline):

    def __init__(self,
                 environment_config: BaseEnvironmentConfig,
                 database_config: DatabaseConfig,
                 new_session: bool = True,
                 session_dir: str = 'sessions',
                 session_name: str = 'data_generation',
                 batch_nb: int = 0,
                 batch_size: int = 0):
        """
        BaseDataGeneration implements the main loop that only produces and stores data (no networks training).

        :param database_config: Configuration object with the parameters of the Database.
        :param environment_config: Configuration object with the parameters of the Environment.
        :param new_session: If True, a new repository will be created for this session.
        :param session_dir: Path to the directory which contains your DeepPhysX session repositories.
        :param session_name: Name of the current session repository.
        :param batch_nb: Number of batches to produce.
        :param batch_size: Number of samples in a single batch.
        """

        BasePipeline.__init__(self,
                              database_config=database_config,
                              environment_config=environment_config,
                              new_session=new_session,
                              session_dir=session_dir,
                              session_name=session_name,
                              pipeline='data_generation')

        # Create a new session if required
        if not self.new_session:
            self.new_session = not exists(join(self.session_dir, self.session_name))
        if self.new_session:
            self.session_name = create_dir(session_dir=self.session_dir,
                                           session_name=self.session_name).split(sep)[-1]

        # Create Managers
        self.database_manager = DatabaseManager(database_config=database_config,
                                                pipeline=self.type,
                                                session=join(self.session_dir, self.session_name),
                                                new_session=self.new_session,
                                                produce_data=True)
        self.environment_manager = EnvironmentManager(environment_config=environment_config,
                                                      pipeline=self.type,
                                                      session=join(self.session_dir, self.session_name),
                                                      produce_data=True,
                                                      batch_size=batch_size)
        self.environment_manager.connect_to_database(**self.database_manager.get_database_paths())

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

            lines_id = self.environment_manager.get_data(animate=True)
            self.database_manager.add_data(data_lines=lines_id)

            self.batch_id += 1
            self.progress_bar.print()

        self.database_manager.close()
        self.environment_manager.close()

    def __str__(self):

        description = BasePipeline.__str__(self)
        description += f"    Number of batches: {self.batch_nb}\n"
        description += f"    Number of sample per batch: {self.batch_size}\n"
        return description
