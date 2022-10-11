import os.path
from os.path import join as osPathJoin
from os.path import basename
from sys import stdout

from DeepPhysX.Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Core.Manager.DataManager import DataManager
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Utils.progressbar import Progressbar
from DeepPhysX.Core.Utils.pathUtils import create_dir, get_first_caller


class BaseDataGenerator(BasePipeline):
    """
    | BaseDataGenerator implement a minimalist execute function that simply produce and store data without
      training a neural network.

    :param BaseDatasetConfig dataset_config: Specialisation containing the parameters of the dataset manager
    :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
    :param str session_name: Name of the newly created directory if session is not defined
    :param int nb_batches: Number of batches
    :param int batch_size: Size of a batch
    :param bool record_input: True if the input must be stored
    :param bool record_output: True if the output must be stored
    """

    def __init__(self,
                 dataset_config: BaseDatasetConfig,
                 environment_config: BaseEnvironmentConfig,
                 session_name: str = 'default',
                 nb_batches: int = 0,
                 batch_size: int = 0,
                 record_input: bool = True,
                 record_output: bool = True):

        BasePipeline.__init__(self,
                              dataset_config=dataset_config,
                              environment_config=environment_config,
                              session_name=session_name,
                              pipeline='dataset')

        # Init session repository
        dataset_dir = dataset_config.dataset_dir
        if dataset_dir is not None:
            if dataset_dir[-1] == "/":
                dataset_dir = dataset_dir[:-1]
            if dataset_dir[-8:] == "/dataset":
                dataset_dir = dataset_dir[:-8]
            if osPathJoin(get_first_caller(), session_name) != osPathJoin(get_first_caller(), dataset_dir):
                dataset_dir = None
            elif not os.path.exists(osPathJoin(get_first_caller(), dataset_dir)):
                dataset_dir = None
        if dataset_dir is None:
            session_dir = create_dir(osPathJoin(get_first_caller(), session_name), dir_name=session_name)
            session_name = (session_name if session_name is not None else basename(session_dir)).split("/")[-1]
        else:
            session_dir = osPathJoin(get_first_caller(), dataset_dir)
            session_name = (session_name if session_name is not None else basename(session_dir)).split("/")[-1]

        # Create a DataManager directly
        self.data_manager = DataManager(manager=self,
                                        dataset_config=dataset_config,
                                        environment_config=environment_config,
                                        session_name=session_name,
                                        session_dir=session_dir,
                                        new_session=True,
                                        offline=True,
                                        record_data={'input': record_input, 'output': record_output},
                                        batch_size=batch_size)
        self.nb_batch: int = nb_batches
        self.progress_bar = Progressbar(start=0, stop=self.nb_batch, c='orange', title="Data Generation")

    def execute(self) -> None:
        """
        | Run the data generation and recording process.
        """

        for i in range(self.nb_batch):
            # Produce a batch
            self.data_manager.get_data()
            # Update progress bar
            stdout.write("\033[K")
            self.progress_bar.print(counts=i + 1)
        # Close manager
        self.data_manager.close()
