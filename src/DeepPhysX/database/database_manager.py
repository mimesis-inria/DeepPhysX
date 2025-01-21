from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple
from os.path import isdir, join, dirname, exists
from os import symlink, makedirs
import json
from numpy import arange, ndarray, array
from numpy.random import shuffle

from SSD.core import Database

from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.utils.path import copy_dir
from DeepPhysX.utils.jsonUtils import CustomJSONEncoder


class DatabaseManager:

    def __init__(self,
                 config: DatabaseConfig,
                 session: str = 'sessions/default'):
        """
        DatabaseManager handle all operations the Database.

        :param config: Configuration object with the parameters of the Database.
        :param session: Path to the session repository.
        """

        # Database repository
        self.database_dir: str = join(session, 'dataset')
        self.modes: List[str] = ['train', 'test', 'run']
        self.mode: str = ''
        self.json_content: Dict[str, Dict[str, Any]] = {'nb_samples': {mode: 0 for mode in self.modes},
                                                        'fields': {}}

        # Pipeline parameters
        self.pipeline: str = ''
        self.config: DatabaseConfig = config

        # Database instances
        self.__db = Database(database_dir=self.database_dir, database_name='dataset')
        self.__exchange = Database(database_dir=self.database_dir, database_name='temp')

        # Samples indexing
        self.sample_indices: ndarray = array([])
        self.sample_id: int = 0
        self.first_add = True

        # Dataset parameters
        # self.normalize: bool = database_config.normalize
        # self.total_nb_sample: int = 0

    ################
    # Init methods #
    ################

    def __init(self):
        self.__exchange.new(remove_existing=True)
        self.__exchange.create_table(table_name='data')

    def init_data_pipeline(self, new_session: bool) -> None:
        """
        Init method for the data pipeline.

        :param new_session: If True, the session is done in a new repository.
        """

        # Define the Database mode
        self.mode = 'train' if self.config.mode is None else self.config.mode
        self.pipeline = 'data'

        # Init Database repository for a new session
        if new_session:
            # Generate new data from scratch --> create a new directory
            if self.config.existing_dir is None:
                makedirs(self.database_dir)
                self.__create()
            # Use existing data in a new session --> copy and load the existing directory
            else:
                copy_dir(src_dir=self.config.existing_dir, dest_dir=dirname(self.database_dir), sub_folders='dataset')
                self.__load()

        # Init Database repository for an existing session
        else:
            self.__load()

        self.__init()

    def init_training_pipeline(self,
                               new_session: bool,
                               produce_data: bool) -> None:
        """
        Init method for the training pipeline.

        :param new_session: If True, the session is done in a new repository.
        :param produce_data: If True, this session will store data in the Database.
        """

        # Define the Database mode
        self.mode = 'train'
        self.pipeline = 'training'

        # Online training pipeline: create data
        if produce_data:

            # Init Database repository for a new session
            if new_session:
                # Generate new data from scratch --> create a new directory
                if self.config.existing_dir is None:
                    makedirs(self.database_dir)
                    self.__create()
                # Use existing data in a new session --> copy and load the existing directory
                else:
                    copy_dir(src_dir=self.config.existing_dir, dest_dir=dirname(self.database_dir), sub_folders='dataset')
                    self.__load()

            # Init Database repository for an existing session
            else:
                self.__load()

        # Load data
        else:

            # Init Database repository for a new session --> link and load the existing Database directory
            if new_session:
                symlink(src=join(self.config.existing_dir, 'dataset'), dst=join(dirname(self.database_dir), 'dataset'))
                self.__load()

            # Init Database repository for an existing session
            else:
                self.__load()

        self.__init()

    def init_prediction_pipeline(self, produce_data: bool) -> None:
        """
        Init method for the prediction pipeline.

        :param produce_data: If True, this session will store data in the Database.
        """

        # Define the Database mode
        self.mode = 'run' if self.config.mode is None else self.config.mode
        self.mode = 'run' if produce_data else self.mode
        self.pipeline = 'prediction'

        # Init Database repository
        self.__load()

        self.__init()

    #########################
    # Repository Management #
    #########################

    def __create(self) -> None:
        """
        Create a new Database.
        """

        # Create a new Database
        self.__db.new()

        # Create a new Table for each mode
        for mode in self.modes:
            self.__db.create_table(table_name=mode, fields=('env_id', int))

        # Create the json information file
        self.__update_json()

    def __load(self) -> None:
        """
        Load an existing Database.
        """

        # Load the existing Database
        if not isdir(self.database_dir):
            raise Warning(f"[{self.__class__.__name__}] The path {self.database_dir} does not exist.")
        self.__db.load()

        # Get the json information file
        if exists(join(self.database_dir, 'dataset.json')):
            with open(join(self.database_dir, 'dataset.json'), 'r') as json_file:
                self.json_content = json.load(json_file)
        else:
            self.__init_json()

        # Index partitions
        self.index_samples()

        # Check normalization
        if self.config.normalize or self.config.recompute_normalization:
            self.compute_normalization()

    def get_database_path(self) -> Tuple[str, str]:

        return self.__db.get_path()

    #########################
    # Json information file #
    #########################

    def __init_json(self) -> None:

        # Get the number of samples for each mode
        for table in self.modes:
            self.json_content['nb_samples'][table] = self.__db.nb_lines(table_name=table)

        # Get the fields architecture
        self.json_content['fields'] = {}
        for field in self.__db.get_architecture()['Train']:
            field_name = field.split(' ')[0]
            if field_name not in ['id', 'env_id']:
                info = {'type': field.split(' ')[1][1:-1]}
                if info['type'] == 'NUMPY':
                    data = self.__db.get_line(table_name='Train', fields=field_name)
                    info['shape'] = data[field_name].shape
                info['normalize'] = [0., 1.]
                self.json_content['fields'][field_name] = info

        # Save json file
        self.__update_json()

    def __update_json(self) -> None:
        """
        Update the JSON info file with the current Database information.
        """

        # Overwrite json file
        with open(join(self.database_dir, 'dataset.json'), 'w') as json_file:
            json.dump(self.json_content, json_file, indent=3, cls=CustomJSONEncoder)

    #########################
    # Database index access #
    #########################

    def index_samples(self) -> None:
        """
        Create a new indexing list of samples. Samples are identified by [partition_id, line_id].
        """

        # Create the indices for each sample such as [partition_id, line_id]
        nb_sample = self.json_content['nb_samples'][self.mode]
        self.sample_indices = arange(1, nb_sample + 1)
        # Init current sample position
        self.sample_id = 0
        # Shuffle the indices if required
        if self.config.shuffle:
            shuffle(self.sample_indices)

    def add_data(self, data_lines: Optional[List[int]] = None) -> None:
        """
        Manage new lines adding in the Database.

        :param data_lines: Indices of the newly added lines.
        """

        # 1.1. Init partitions information on the first sample
        if self.first_add:
            self.first_add = False
            self.__db.load()
            self.__exchange.load()
            self.__init_json()
        self.json_content['nb_samples'][self.mode] = self.__db.nb_lines(table_name=self.mode)

        # 1.2. Update the normalization coefficients if required
        if self.config.normalize and self.mode == 'train' and self.pipeline == 'training' and data_lines is not None:
            self.compute_batch_normalization(data_lines=data_lines)

        # 1. Update the json file
        self.__update_json()

    def get_data(self,
                 batch_size: int) -> List[List[int]]:
        """
        Select a batch of indices to read in the Database.

        :param batch_size: Number of sample in a single batch.
        """

        # 1. Check if dataset is loaded and if the current sample is not the last
        if self.sample_id >= len(self.sample_indices):
            self.index_samples()

        # 2. Update dataset index and get a batch of data
        idx = self.sample_id
        self.sample_id += batch_size
        lines = self.sample_indices[idx:self.sample_id].tolist()

        # 3. Ensure the batch has the good size
        if len(lines) < batch_size:
            lines += self.get_data(batch_size=batch_size - len(lines))

        return lines

    def change_mode(self, mode: str) -> None:
        """
        Change the current Database mode.

        :param mode: Name of the Database mode.
        """

        self.mode = mode

    ######################
    # Data normalization #
    ######################

    def compute_normalization(self) -> None:
        """
        Compute the mean and the standard deviation of all the training samples for each data field.
        """

        data_to_normalize = self.__db.get_lines(table_name='train',
                                                fields=list(self.json_content['fields'].keys()),
                                                batched=True)
        del data_to_normalize['id']
        for field_name, data in data_to_normalize.items():
            self.json_content['fields'][field_name]['normalize'] = [array(data).mean(), array(data).std()]
        self.__update_json()

    def compute_batch_normalization(self, data_lines: List[int]) -> None:
        """
        Compute the mean and the standard deviation of the batched samples for each data field.
        """

        data_to_normalize = self.__db.get_lines(table_name='train',
                                                fields=list(self.json_content['fields'].keys()),
                                                lines_id=data_lines,
                                                batched=True)
        del data_to_normalize['id']
        for field_name, data in data_to_normalize.items():
            self.json_content['fields'][field_name]['normalize'] = [array(data).mean(), array(data).std()]
        self.__update_json()

    ####################
    # Manager behavior #
    ####################

    def close(self):
        """
        Launch the closing procedure of the DatabaseManager.
        """

        # Compute final normalization if required
        if self.config.normalize and self.pipeline == 'data':
            self.compute_normalization()

        # Close Database partitions
        self.__db.close()
        self.__exchange.close(erase_file=True)

    def __str__(self):

        description = "\n"
        description += f"# {self.__class__.__name__}\n"
        description += f"    Dataset Repository: {self.database_dir}\n"
        return description
