from typing import Any, Dict, Tuple, List, Optional, Union
from os.path import isfile, isdir, abspath, join
from os import listdir, symlink, sep
from json import dump as json_dump
from json import load as json_load
from numpy import load, squeeze, ndarray, concatenate, float64

from SSD.Core.Storage.Database import Database

from DeepPhysX.Core.Database.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Database.BaseDataset import BaseDataset
from DeepPhysX.Core.Utils.path import create_dir, copy_dir
from DeepPhysX.Core.Utils.jsonUtils import CustomJSONEncoder


class DatasetManager:

    def __init__(self,
                 dataset_config: Optional[BaseDatasetConfig] = None,
                 session: str = 'sessions/default',
                 data_manager: Optional[Any] = None,
                 new_session: bool = True,
                 is_training: bool = True,
                 produce_data: bool = True):
        """
        DatasetManager handle all operations with input / output files. Allows saving and read tensors from files.

        :param dataset_config: Specialisation containing the parameters of the dataset manager
        :param data_manager: DataManager that handles the DatasetManager
        :param new_session: Define the creation of new directories to store data
        :param is_training: True if the session is done offline
        :param produce_data: True if this session is a network training
        """

        self.name: str = self.__class__.__name__

        # Manager variables
        dataset_config = BaseDatasetConfig() if dataset_config is None else dataset_config
        self.data_manager: Optional[Any] = data_manager
        self.dataset_dir: str = join(session, 'dataset')
        self.database: Optional[Database] = None

        # Dataset parameters
        self.max_file_size: int = dataset_config.max_file_size
        self.shuffle: bool = dataset_config.shuffle
        self.produce_data = produce_data
        self.normalize: bool = dataset_config.normalize
        self.recompute_normalization: bool = dataset_config.recompute_normalization

        # Dataset modes
        self.modes: List[str] = ['training', 'validation', 'running']
        self.mode: str = 'training' if produce_data else 'running'
        self.mode = self.mode if dataset_config.mode is None else dataset_config.mode

        # Dataset partitions
        session_name = session.split(sep)[-1]
        self.partition_template: Dict[str, str] = {mode: f'{session_name}_{mode}_' + '{}' for mode in self.modes}
        self.partitions: Dict[str, List[str]] = {mode: [] for mode in self.modes}
        self.partition_index: Dict[str, int] = {mode: 0 for mode in self.modes}

        # Dataset json file
        self.json_default: Dict[str, Dict[str, Any]] = {'data_shape': {},
                                                        'nb_samples': {mode: [] for mode in self.modes},
                                                        'partitions': {mode: [] for mode in self.modes},
                                                        'normalization': {}}
        self.json_content: Dict[str, Dict[str, Any]] = self.json_default.copy()

        # Produce training data
        if produce_data:
            # Produce training data in a new session
            if new_session:
                # Produce training data in a new session from scratch
                # --> Create a new  '/dataset' directory
                if dataset_config.existing_dir is None:
                    create_dir(session_dir=session, session_name='dataset')
                    self.init_directory()
                # Produce training data in a new session from an existing Dataset
                # --> Copy the 'existing_dir/dataset' directory then load the 'session/dataset' directory
                else:
                    copy_dir(src_dir=dataset_config.existing_dir, dest_dir=session, sub_folders='dataset')
                    self.load_directory()
            # Produce training data in an existing session
            # --> Load the 'session/dataset' directory
            else:
                self.load_directory()
        # Load training data
        else:
            # Load training data in a new session
            # --> Link to the 'existing_dir/dataset' directory the load the 'session/dataset' directory
            if new_session:
                symlink(src=join(dataset_config.existing_dir, 'dataset'),
                        dst=join(session, 'dataset'))
                self.load_directory()
            # Load training data in an existing session
            # --> Load the 'session/dataset' directory
            else:
                self.load_directory()

    def init_directory(self):
        """

        """

        partition_path = self.partition_template[self.mode].format(self.partition_index[self.mode])
        self.database = Database(database_dir=self.dataset_dir,
                                 database_name=partition_path).new()
        self.database.create_table(table_name='Sync',
                                   storing_table=False,
                                   fields=[('env', int), ('net', int)])
        self.database.create_table(table_name='Training')
        self.database.create_table(table_name='Additional')
        self.partition_index[self.mode] += 1

    def load_directory(self):

        # 1. Check the directory existence to prevent bugs
        if not isdir(self.dataset_dir):
            raise Warning(f"[{self.name}] Impossible to load Dataset from {self.dataset_dir}.")

        # 2. Get the .json description file
        json_found = False
        if isfile(join(self.dataset_dir, 'dataset.json')):
            json_found = True
            with open(join(self.dataset_dir, 'dataset.json')) as json_file:
                self.json_content = json_load(json_file)

        # 3. Load partitions for each mode
        self.partitions = self.json_content['partitions'] if json_found else self.search_partitions()

        # 4. Update json file if not found
        if not json_found or self.json_content == self.json_default:
            self.search_partitions_info()
            self.update_json(update_partitions_lists=True)
        if self.recompute_normalization or (
                self.normalize and self.json_content['normalization'] == self.json_default['normalization']):
            self.update_json(update_normalization=True)

        # 5. Load data from partitions
        self.load_partitions()

    def search_partitions(self):
        """

        """

        raw_partitions = {mode: [f for f in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, f))
                                 and f.endswith('.db') and f.__contains__(mode)] for mode in self.modes}
        return {mode: sorted(raw_partitions[mode]) for mode in self.modes}

    def search_partitions_info(self):
        pass

    def load_partitions(self):
        pass

    def update_json(self, update_shapes: bool = False, update_nb_samples: bool = False,
                    update_partitions_lists: bool = False, update_normalization: bool = False) -> None:
        pass

    def close(self):

        if self.normalize and self.produce_data:
            self.update_json(update_normalization=True)
        self.database.close()
