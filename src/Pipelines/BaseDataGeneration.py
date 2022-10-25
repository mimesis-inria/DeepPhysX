from typing import Optional
from os.path import join, sep, exists
from sys import stdout

from DeepPhysX.Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Core.Manager.DataManager import DataManager
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Utils.progressbar import Progressbar
from DeepPhysX.Core.Utils.path import get_first_caller, create_dir


class BaseDataGeneration(BasePipeline):

    def __init__(self,
                 environment_config: BaseEnvironmentConfig,
                 database_config: Optional[BaseDatabaseConfig] = None,
                 session_dir: str = 'sessions',
                 session_name: str = 'data_generation',
                 new_session: bool = True,
                 batch_nb: int = 0,
                 batch_size: int = 0):
        """
        BaseDataGeneration implements the main loop that only produces and stores data (no Network training).

        :param database_config: Specialisation containing the parameters of the dataset manager.
        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param session_dir: Relative path to the directory which contains sessions directories.
        :param session_name: Name of the new the session directory.
        :param new_session: If True, the session will be run in a new repository.
        :param batch_nb: Number of batches to produce.
        :param batch_size: Number of samples in a single batch.
        """

        BasePipeline.__init__(self,
                              database_config=database_config,
                              environment_config=environment_config,
                              session_dir=session_dir,
                              session_name=session_name,
                              new_session=new_session,
                              pipeline='data_generation')

        # Define the session repository
        root = get_first_caller()
        session_dir = join(root, session_dir)

        # Create a new session if required
        if not new_session:
            new_session = not exists(session_dir)
        if new_session:
            session_name = create_dir(session_dir=session_dir,
                                      session_name=session_name).split(sep)[-1]

        # Create a DataManager
        self.data_manager = DataManager(database_config=database_config,
                                        environment_config=environment_config,
                                        session=join(session_dir, session_name),
                                        new_session=new_session,
                                        pipeline='data_generation',
                                        produce_data=True,
                                        batch_size=batch_size)

        # Data generation variables
        self.batch_nb: int = batch_nb
        self.batch_id: int = 0
        self.progress_bar = Progressbar(start=0, stop=self.batch_id, c='orange', title="Data Generation")

    def execute(self) -> None:
        """
        Launch the data generation Pipeline.
        Each event is already implemented for a basic pipeline but can also be rewritten via inheritance to describe a
        more complex pipeline.
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
        Called once at the beginning of the data generation pipeline.
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

        stdout.write("\033[K")
        self.progress_bar.print(counts=self.batch_id + 1)

    def data_generation_end(self) -> None:
        """
        Called once at the beginning of the data generation pipeline.
        """

        self.data_manager.close()
