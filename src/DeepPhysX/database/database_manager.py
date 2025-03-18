from typing import Any, Dict, List, Optional
from os.path import isdir, join, dirname, exists, sep, isabs, abspath
from os import symlink, makedirs
from numpy import arange, ndarray, array
from numpy.random import shuffle
import json

from SSD.core import Database

from DeepPhysX.utils.path import copy_dir
from DeepPhysX.utils.jsonUtils import CustomJSONEncoder


class DatabaseManager:

    def __init__(self,
                 existing_dir: Optional[str] = None,
                 shuffle_data: bool = True,
                 normalize: bool = True,
                 recompute_normalization: bool = False):
        """
        DatabaseManager handles the Database files, the data writing and reading access, the data normalisation and
        shuffle.

        :param existing_dir: Path to an existing Database repository.
        :param shuffle_data: If True, data is shuffled when a batch is taken.
        :param normalize: If True, the data will be normalized using standard score.
        :param recompute_normalization: If True, compute the normalisation coefficients.
        """

        # Database repository variables
        self.database_dir: str = ''
        if existing_dir is not None:
            if not isdir(existing_dir):
                raise ValueError(f"[] The given 'existing_dir'={existing_dir} does not exist.")
            if len(existing_dir.split(sep)) > 1 and existing_dir.split(sep)[-1] == 'dataset':
                existing_dir = join(*existing_dir.split(sep)[:-1])
            if not isabs(existing_dir):
                existing_dir = abspath(existing_dir)
        self.existing_dir: Optional[str] = existing_dir

        # Database instances variables
        self.__db: Optional[Database] = None
        self.__exchange: Optional[Database] = None

        # Database tables variables
        self.mode: str = ''
        self.modes: List[str] = ['train', 'test', 'run']
        self.json_content: Dict[str, Dict[str, Any]] = {'nb_samples': {mode: 0 for mode in self.modes},
                                                        'fields': {}}

        # Data access variables
        self.pipeline: str = ''
        self.first_add = True
        self.sample_id: int = 0
        self.sample_indices: ndarray = array([])
        self.shuffle: bool = shuffle_data
        self.normalize: bool = normalize
        self.recompute_normalization: bool = recompute_normalization

    ################
    # Init methods #
    ################

    def init_data_pipeline(self, session: str, new_session: bool) -> None:
        """
        Init the DatabaseManager for the data generation pipeline.

        :param session: Path to the session repository.
        :param new_session: If True, a new repository is created for the session.
        """

        # Create the Database
        self.database_dir = join(session, 'dataset')
        self.__db = Database(database_dir=self.database_dir, database_name='dataset')

        # Configure the Database for the current pipeline
        self.mode = 'train'
        self.pipeline = 'data'

        # Case 1: Use a new Database repository
        if new_session:

            # Case 1.1: Generate data from scratch --> create a new repository
            if self.existing_dir is None:
                makedirs(self.database_dir)
                self.__create()

            # Case 1.2: Add data to an existing Database --> copy and load the existing repository
            else:
                copy_dir(src_dir=self.existing_dir, dest_dir=dirname(self.database_dir), sub_folders='dataset')
                self.__load()

        # Case 2: Use an existing Database repository
        else:
            self.__load()

        # Create the exchange Database
        self.__exchange = Database(database_dir=self.database_dir, database_name='temp')
        self.__exchange.new(remove_existing=True)
        self.__exchange.create_table(table_name='data')

    def init_training_pipeline(self,
                               session: str,
                               new_session: bool,
                               produce_data: bool) -> None:
        """
        Init the DatabaseManager for the training pipeline.

        :param session: Path to the session repository.
        :param new_session: If True, a new repository is created for the session.
        :param produce_data: If True, this session will store data in the Database.
        """

        # Create the Database
        self.database_dir = join(session, 'dataset')
        self.__db = Database(database_dir=self.database_dir, database_name='dataset')

        # Configure the Database for the current pipeline 
        self.mode = 'train'
        self.pipeline = 'training'

        # Case 1: Online training pipeline: create data
        if produce_data:

            # Init Database repository for a new session
            if new_session:
                # Generate new data from scratch --> create a new directory
                if self.existing_dir is None:
                    makedirs(self.database_dir)
                    self.__create()
                # Use existing data in a new session --> copy and load the existing directory
                else:
                    copy_dir(src_dir=self.existing_dir, dest_dir=dirname(self.database_dir), sub_folders='dataset')
                    self.__load()

            # Init Database repository for an existing session
            else:
                self.__load()

        # Case 2: Offline training pipeline: load data
        else:

            # Init Database repository for a new session --> link and load the existing Database directory
            if new_session:
                symlink(src=join(self.existing_dir, 'dataset'), dst=self.database_dir)
                self.__load()

            # Init Database repository for an existing session
            else:
                self.__load()

        # Create the exchange Database
        self.__exchange = Database(database_dir=self.database_dir, database_name='temp')
        self.__exchange.new(remove_existing=True)
        self.__exchange.create_table(table_name='data')

    def init_prediction_pipeline(self, session: str, produce_data: bool) -> None:
        """
        Init method for the prediction pipeline.

        :param session:
        :param produce_data: If True, this session will store data in the Database.
        """

        # Create the Database
        self.database_dir = join(session, 'dataset')
        self.__db = Database(database_dir=self.database_dir, database_name='dataset')

        # Configure the Database for the current pipeline
        self.mode = 'run'
        self.pipeline = 'prediction'

        # Load data
        self.__load()

        # Create the exchange Database
        self.__exchange = Database(database_dir=self.database_dir, database_name='temp')
        self.__exchange.new(remove_existing=True)
        self.__exchange.create_table(table_name='data')

    @staticmethod
    def __check_init(foo):
        """
        Wrapper to check that an 'init_*_pipeline' method was called before to use the DatabaseManager.
        """

        def wrapper(self, *args, **kwargs):
            if self.__db is None:
                raise ValueError(f"[DatabaseManager] The manager is not completely initialized; please use one of the "
                                 f"'init_*_pipeline' methods.")
            return foo(self, *args, **kwargs)

        return wrapper

    #########################
    # Repository Management #
    #########################

    def __create(self) -> None:
        """
        Create a new Database.
        """

        # Create a new Database with a new Table for each mode
        self.__db.new()
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
        self.__index_samples()

        # Check normalization
        if self.normalize or self.recompute_normalization:
            self.compute_normalization()

    #########################
    # Json information file #
    #########################

    def __init_json(self) -> None:
        """
        Initialize the JSON information file.
        """

        # Get the number of samples for each mode
        for table in self.modes:
            self.json_content['nb_samples'][table] = self.__db.nb_lines(table_name=table)

        # Get the fields architectures
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
        Update the JSON information file with the current Database information.
        """

        # Overwrite json file
        with open(join(self.database_dir, 'dataset.json'), 'w') as json_file:
            json.dump(self.json_content, json_file, indent=3, cls=CustomJSONEncoder)

    #########################
    # Database index access #
    #########################

    def __index_samples(self) -> None:
        """
        Create a new indexing list of samples.
        """

        # Create the indexing list
        self.sample_indices = arange(1, self.json_content['nb_samples'][self.mode] + 1)
        self.sample_id = 0

        # Shuffle the indices if required
        if self.shuffle:
            shuffle(self.sample_indices)

    @__check_init
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
        if self.normalize and self.mode == 'train' and self.pipeline == 'training' and data_lines is not None:
            self.__compute_batch_normalization(data_lines=data_lines)

        # 1. Update the json file
        self.__update_json()

    @__check_init
    def get_data(self, batch_size: int) -> List[int]:
        """
        Select a batch of indices to read in the Database.

        :param batch_size: Number of sample in a single batch.
        """

        # 1. Check if dataset is loaded and if the current sample is not the last
        if self.sample_id >= len(self.sample_indices):
            self.__index_samples()

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

        # TODO: check if the change of this mode really behaves correctly, then turn the mode variable to private
        self.mode = mode

    ######################
    # Data normalization #
    ######################

    def compute_normalization(self) -> None:
        """
        Compute the mean and the standard deviation of all the training samples for each data field.
        """

        # Get the normalization coefficient for each data field
        for field_name in self.json_content['fields'].keys():

            # Load the data to normalize
            data = self.__db.get_lines(table_name='train', fields=field_name, batched=True)[field_name]

            # Get the normalization coefficients
            self.json_content['fields'][field_name]['normalize'] = [array(data).mean(), array(data).std()]

        # Update the json information file
        self.__update_json()

    def __compute_batch_normalization(self, data_lines: List[int]) -> None:
        """
        Compute the mean and the standard deviation of the batched samples for each data field.
        """

        # Get the normalization coefficient for each data field
        for field_name in self.json_content['fields'].keys():

            # Load the batch to normalize
            data = self.__db.get_lines(table_name='train', fields=field_name, lines_id=data_lines, batched=True)[field_name]

            # Get the normalization coefficients
            self.json_content['fields'][field_name]['normalize'] = [array(data).mean(), array(data).std()]

        # Update the json information file
        self.__update_json()

    ####################
    # Manager behavior #
    ####################

    def close(self):
        """
        Launch the closing procedure of the DatabaseManager.
        """

        # Compute final normalization if required
        if self.normalize and self.pipeline == 'data':
            self.compute_normalization()

        # Close Database partitions
        self.__db.close()
        self.__exchange.close(erase_file=True)

    def __str__(self):

        desc = "\n"
        desc += f"# DATABASE MANAGER\n"
        desc += f"    Dataset Repository: {self.database_dir}\n"
        return desc
