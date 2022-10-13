from os.path import join, sep, exists
from sys import stdout

from DeepPhysX.Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Core.Manager.DataManager import DataManager
from DeepPhysX.Core.Database.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Utils.progressbar import Progressbar
from DeepPhysX.Core.Utils.path import get_first_caller, create_dir


class BaseDataGenerator(BasePipeline):

    def __init__(self,
                 dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig,
                 session_dir: str = 'sessions',
                 session_name: str = 'data_generation',
                 batch_nb: int = 0,
                 batch_size: int = 0):
        """
        BaseDataGenerator implement the main loop that only produces and stores data (no Network training).

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

        # Create a DataManager
        self.data_manager = DataManager(dataset_config=dataset_config,
                                        environment_config=environment_config,
                                        session=join(session_dir, session_name),
                                        new_session=new_session,
                                        is_training=False,
                                        produce_data=True,
                                        batch_size=batch_size)
        self.nb_batch: int = batch_nb
        self.progress_bar = Progressbar(start=0, stop=self.nb_batch, c='orange', title="Data Generation")

    def execute(self) -> None:
        """
        Launch the data generation Pipeline.
        """

        # Produce each batch of data
        for i in range(self.nb_batch):

            # Produce a batch
            self.data_manager.get_data()

            # Update progress bar
            stdout.write("\033[K")
            self.progress_bar.print(counts=i + 1)

        # Close DataManager
        self.data_manager.close()
