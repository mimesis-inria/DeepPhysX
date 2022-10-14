from typing import Optional
from os.path import join, sep
from sys import stdout

from DeepPhysX.Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Core.Manager.DataManager import DataManager
from DeepPhysX.Core.Database.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Utils.progressbar import Progressbar
from DeepPhysX.Core.Utils.path import get_first_caller, create_dir


class BaseDataGenerator(BasePipeline):

    def __init__(self,
                 environment_config: BaseEnvironmentConfig,
                 dataset_config: Optional[BaseDatasetConfig] = None,
                 session_dir: str = 'sessions',
                 session_name: str = 'data_generation',
                 batch_nb: int = 0,
                 batch_size: int = 0):
        """
        BaseDataGenerator implements the main loop that only produces and stores data (no Network training).

        :param dataset_config: Specialisation containing the parameters of the dataset manager.
        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param session_dir: Relative path to the directory which contains sessions directories.
        :param session_name: Name of the new the session directory.
        :param batch_nb: Number of batches to produce.
        :param batch_size: Number of samples in a single batch.
        """

        BasePipeline.__init__(self,
                              dataset_config=dataset_config,
                              environment_config=environment_config,
                              session_dir=session_dir,
                              session_name=session_name,
                              pipeline='data_generation')

        # Define the session repository
        root = get_first_caller()
        session_dir = join(root, session_dir)

        # Configure 'new_session' flags
        # Option 1: existing_dir == None --> new_session = True
        # Option 2: existing_dir == session_dir/session_name --> new_session = False
        # Option 3: existing_dir != session_dir/session_name --> new_session = True
        new_session = True
        if dataset_config is not None and dataset_config.existing_dir is not None and \
                join(session_dir, session_name) == join(root, dataset_config.existing_dir):
            new_session = False

        # Create a new session if required
        if new_session:
            session_name = create_dir(session_dir=session_dir,
                                      session_name=session_name).split(sep)[-1]

        # Data generation variables
        self.batch_nb: int = batch_nb
        self.batch_id: int = 0
        self.progress_bar = Progressbar(start=0, stop=self.batch_id, c='orange', title="Data Generation")

        # Create a DataManager
        self.data_manager = DataManager(dataset_config=dataset_config,
                                        environment_config=environment_config,
                                        session=join(session_dir, session_name),
                                        new_session=new_session,
                                        is_training=False,
                                        produce_data=True,
                                        batch_size=batch_size)

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
        Called once at the beginning of a batch production.
        """

        stdout.write("\033[K")
        self.progress_bar.print(counts=self.batch_id + 1)

    def data_generation_end(self) -> None:
        """
        Called once at the beginning of the data generation pipeline.
        """

        self.data_manager.close()
