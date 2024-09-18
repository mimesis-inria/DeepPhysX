from typing import Optional
from os.path import join, sep, exists
from vedo import ProgressBar

from DeepPhysX.pipelines.core.base_pipeline import BasePipeline
from DeepPhysX.pipelines.core.data_manager import DataManager
from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.simulation.core.base_environment_config import BaseEnvironmentConfig
from DeepPhysX.utils.path import create_dir


class BaseDataGeneration(BasePipeline):

    def __init__(self,
                 environment_config: BaseEnvironmentConfig,
                 database_config: Optional[DatabaseConfig] = None,
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

        # Create a DataManager
        self.data_manager = DataManager(pipeline=self,
                                        database_config=database_config,
                                        environment_config=environment_config,
                                        session=join(self.session_dir, self.session_name),
                                        new_session=self.new_session,
                                        produce_data=True,
                                        batch_size=batch_size)

        # Data generation variables
        self.batch_nb: int = batch_nb
        self.batch_id: int = 0
        self.batch_size = batch_size
        self.progress_bar = ProgressBar(start=0, stop=self.batch_nb, c='orange', title="Data Generation")

    def execute(self) -> None:
        """
        Launch the data generation Pipeline.
        Each event is already implemented for a basic Pipeline but can also be rewritten via inheritance to describe a
        more complex Pipeline.
        """

        self.data_generation_begin()
        while self.batch_condition():
            self.batch_begin()
            self.batch_produce()
            self.batch_count()
            self.batch_end()
        self.data_generation_end()

    def data_generation_begin(self) -> None:
        """
        Called once at the beginning of the data generation Pipeline.
        """

        pass

    def batch_condition(self) -> bool:
        """
        Check the batch number condition.
        """

        return self.batch_id < self.batch_nb

    def batch_begin(self) -> None:
        """
        Called once at the beginning of a batch production.
        """

        pass

    def batch_produce(self) -> None:
        """
        Trigger the data production.
        """

        self.data_manager.get_data()

    def batch_count(self) -> None:
        """
        Increment the batch counter.
        """

        self.batch_id += 1

    def batch_end(self) -> None:
        """
        Called once at the end of a batch production.
        """

        self.progress_bar.print()

    def data_generation_end(self) -> None:
        """
        Called once at the beginning of the data generation Pipeline.
        """

        self.data_manager.close()

    def __str__(self):

        description = BasePipeline.__str__(self)
        description += f"    Number of batches: {self.batch_nb}\n"
        description += f"    Number of sample per batch: {self.batch_size}\n"
        return description
