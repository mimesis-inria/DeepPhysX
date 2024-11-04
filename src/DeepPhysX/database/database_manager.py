from typing import Any, Dict, List, Optional, Tuple
from os.path import isfile, isdir, join
from os import listdir, symlink, sep, remove, rename, makedirs
from json import dump as json_dump
from json import load as json_load
from numpy import arange, ndarray, array, abs, mean, sqrt, empty, concatenate
from numpy.random import shuffle, normal

from SSD.core import Database

from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.utils.path import copy_dir
from DeepPhysX.utils.jsonUtils import CustomJSONEncoder


class DatabaseManager:

    def __init__(self,
                 database_config: DatabaseConfig,
                 pipeline: str = '',
                 new_session: bool = True,
                 session: str = 'sessions/default',
                 produce_data: bool = True):
        """
        DatabaseManager handle all operations the Database.

        :param database_config: Configuration object with the parameters of the Database.
        :param pipeline: Type of the Pipeline.
        :param new_session: If True, the session is done in a new repository.
        :param session: Path to the session repository.
        :param produce_data: If True, this session will store data in the Database.
        """

        self.__name: str = self.__class__.__name__

        # Session variables
        self.pipeline: str = pipeline
        self.database_dir: str = join(session, 'dataset')

        # Dataset parameters
        self.shuffle: bool = database_config.shuffle
        self.produce_data = produce_data
        self.normalize: bool = database_config.normalize
        self.total_nb_sample: int = 0
        self.recompute_normalization: bool = database_config.recompute_normalization

        # Dataset modes
        self.modes: List[str] = ['train', 'test', 'run']

        if pipeline == 'data_generation':
            self.mode: str = 'train' if database_config.mode is None else database_config.mode
        elif pipeline == 'training':
            self.mode: str = 'train'
        else:
            self.mode: str = 'run' if database_config.mode is None else database_config.mode
            self.mode = 'run' if produce_data else self.mode

        # Dataset
        self.DB = Database(database_dir=self.database_dir, database_name='dataset')

        # Dataset indexing
        self.sample_indices: ndarray = empty((0, 2), dtype=int)
        self.sample_id: int = 0
        self.first_add = True

        # Dataset json file
        self.json_default: Dict[str, Dict[str, Any]] = {'nb_samples': {mode: 0 for mode in self.modes},
                                                        'fields': {}}
        self.json_content: Dict[str, Dict[str, Any]] = self.json_default.copy()

        # DataGeneration case
        if self.pipeline == 'data_generation':

            # Generate data in a new session
            if new_session:
                # Generate data from scratch --> create a new directory
                if database_config.existing_dir is None:
                    makedirs(join(session, 'dataset'))
                    self.create()
                # Complete a Database in a new session --> copy and load the existing directory
                else:
                    copy_dir(src_dir=database_config.existing_dir, dest_dir=session, sub_folders='dataset')
                    self.load()
            # Complete a Database in the same session --> load the directory
            else:
                self.load()

        # Training case
        elif self.pipeline == 'training':

            # Generate data
            if produce_data:
                # Generate data in a new session
                if new_session:
                    # Generate data from scratch --> create a new directory
                    if database_config.existing_dir is None:
                        makedirs(join(session, 'dataset'))
                        self.create()
                    # Complete a Database in a new session --> copy and load the existing directory
                    else:
                        copy_dir(src_dir=database_config.existing_dir, dest_dir=session, sub_folders='dataset')
                        self.load()
                # Complete a Database in the same directory --> load the directory
                else:
                    self.load()

            # Load data
            else:
                # Load data in a new session  --> link and load the existing directory
                if new_session:
                    symlink(src=join(database_config.existing_dir, 'dataset'),
                            dst=join(session, 'dataset'))
                    self.load()
                # Load data in the same session  --> load the directory
                else:
                    self.load()

        # Prediction case
        else:
            self.load()

        # Finally create an exchange database
        self.exchange = Database(database_dir=self.database_dir,
                                 database_name='temp').new(remove_existing=True)
        self.exchange.create_table(table_name='data')

    ##########################################################################################
    ##########################################################################################
    #                                   Partitions Management                                #
    ##########################################################################################
    ##########################################################################################

    def create(self) -> None:
        """
        Create a new partition of the Database.
        """

        # 2. Create the Database
        self.DB.new()
        for mode in self.modes:
            self.DB.create_table(table_name=mode,
                                 fields=('env_id', int))

        self.json_content['nb_samples'][self.mode] = self.DB.nb_lines(table_name=self.mode)
        self.update_json()

    def load(self) -> None:
        """
        Get the Database information from the json file (partitions, samples, etc).
        Load all the partitions or create one if necessary.
        """

        # 1. Check the directory existence to prevent bugs
        if not isdir(self.database_dir):
            raise Warning(f"[{self.__name}] Impossible to load Dataset from {self.database_dir}.")
        self.DB.load()

        # 2. Get the .json description file
        json_found = False
        if isfile(join(self.database_dir, 'dataset.json')):
            json_found = True
            with open(join(self.database_dir, 'dataset.json')) as json_file:
                self.json_content = json_load(json_file)

        # 3. Update json file if not found
        if not json_found or self.json_content == self.json_default:
            # Get the number of samples
            for table in ['train', 'test', 'run']:
                self.json_content['nb_samples'][table] = self.DB.nb_lines(table_name=table)
            # Get the fields architecture
            self.json_content['fields'] = self.get_database_architecture()
            self.update_json()

        # 6. Index partitions
        self.index_samples()

        # 7. Check normalization
        for field_name in self.json_content['fields'].keys():
            if (self.normalize and self.json_content['fields'][field_name]['normalize'] == [0., 1.]) or self.recompute_normalization:
                self.json_content['fields'][field_name]['normalize'] = self.compute_normalization(field_name=field_name)
                self.update_json()

    # def get_partition_objects(self) -> List[Database]:
    #     """
    #     Get the list of partitions of the Database for the current mode.
    #     """
    #
    #     return self.partitions[self.mode]

    # def get_partition_names(self) -> List[List[str]]:
    #     """
    #     Get the list of partition paths of the Database for the current mode.
    #     """
    #
    #     return [db.get_path() for db in self.partitions[self.mode]]

    def change_mode(self,
                    mode: str) -> None:
        """
        Change the current Database mode.

        :param mode: Name of the Database mode.
        """

        pass

    ##########################################################################################
    ##########################################################################################
    #                                 JSON Information file                                  #
    ##########################################################################################
    ##########################################################################################

    def get_database_architecture(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the Tables and Fields structure of the Database.
        """

        fields = {}
        for field in self.DB.get_architecture()['Train']:
            field_name = field.split(' ')[0]
            if field_name not in ['id', 'env_id']:
                info = {'type': field.split(' ')[1][1:-1]}
                if info['type'] == 'NUMPY':
                    data = self.DB.get_line(table_name=self.mode, fields=field_name)
                    info['shape'] = data[field_name].shape
                info['normalize'] = [0., 1.]
                fields[field_name] = info
        return fields

    def update_json(self) -> None:
        """
        Update the JSON info file with the current Database information.
        """

        # Overwrite json file
        with open(join(self.database_dir, 'dataset.json'), 'w') as json_file:
            json_dump(self.json_content, json_file, indent=3, cls=CustomJSONEncoder)

    ##########################################################################################
    ##########################################################################################
    #                               Database access and edition                              #
    ##########################################################################################
    ##########################################################################################

    def get_database_paths(self) -> Dict[str, Tuple[str, str]]:

        return {'database': self.DB.get_path(),
                'exchange_db': self.exchange.get_path()}

    # def connect_handler(self,
    #                     handler: DatabaseHandler) -> None:
    #     """
    #     Add and init a new DatabaseHandler to the list.
    #
    #     :param handler: New DatabaseHandler.
    #     """
    #
    #     handler.init(storing_partitions=self.get_partition_objects(),
    #                  exchange_db=self.exchange)
    #     self.database_handlers.append(handler)

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
        if self.shuffle:
            shuffle(self.sample_indices)

    def add_data(self,
                 data_lines: Optional[List[int]] = None) -> None:
        """
        Manage new lines adding in the Database.

        :param data_lines: Indices of the newly added lines.
        """

        # 1. Update the json file
        self.json_content['nb_samples'][self.mode] = self.DB.nb_lines(table_name=self.mode)
        # 1.1. Init partitions information on the first sample
        if self.first_add:
            self.first_add = False
            self.DB.load()
            self.exchange.load()
            self.json_content['fields'] = self.get_database_architecture()
        self.update_json()

        # 1.2. Update the normalization coefficients if required
        if self.normalize and self.mode == 'training' and self.pipeline == 'training' and data_lines is not None:
            self.json_content['normalization'] = self.update_normalization(data_lines=data_lines)
            self.update_json()

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

    ##########################################################################################
    ##########################################################################################
    #                              Data normalization computation                            #
    ##########################################################################################
    ##########################################################################################

    @property
    def normalization(self) -> Optional[Dict[str, List[float]]]:
        """
        Get the normalization coefficients.
        """

        if not self.normalize:
            return None
        normalize = {}
        for field_name in self.json_content['fields']:
            normalize[field_name] = self.json_content['fields'][field_name]['normalize']
        return normalize

    def compute_normalization(self, field_name: str) -> List[float]:
        """
        Compute the mean and the standard deviation of all the training samples for each data field.
        """

        data_to_normalize = array(self.DB.get_lines(table_name='train',
                                                    fields=field_name,
                                                    batched=True)[field_name])
        return [data_to_normalize.mean(), data_to_normalize.std()]

    def update_normalization(self,
                             data_lines: List[int]) -> Dict[str, List[float]]:
        """
        Update the mean and the standard deviation of all the training samples with newly added samples for each data
        field.

        :param data_lines: Indices of the newly added lines.
        """

        # 1. Get the previous normalization coefficients and number of samples
        previous_normalization = self.normalization
        if previous_normalization is None:
            return self.compute_normalization()
        new_normalization = previous_normalization.copy()
        previous_nb_samples = self.total_nb_sample
        self.total_nb_sample += len(data_lines)

        # 2. Compute the global mean of samples for each field
        fields = list(previous_normalization.keys())
        data_to_normalize = self.partitions[self.mode][-1].get_lines(table_name='Training',
                                                                     fields=fields,
                                                                     lines_id=data_lines,
                                                                     batched=True)
        for field in fields:
            data = array(data_to_normalize[field])
            m = (previous_nb_samples / self.total_nb_sample) * previous_normalization[field][0] + \
                (len(data_lines) / self.total_nb_sample) * data.mean()
            new_normalization[field][0] = m

        # 3. Compute standard deviation of samples for each field
        stds = {field: [] for field in fields}
        nb_samples = []
        # 3.1. Recompute the standard deviation for each partition with the new mean value
        for partition in self.partitions['training']:
            data_to_normalize = self.load_partitions_fields(partition=partition,
                                                            fields=fields)
            nb_samples.append(data_to_normalize['id'][-1])
            for field in fields:
                data = array(data_to_normalize[field])
                stds[field].append(mean(abs(data - new_normalization[field][0]) ** 2))
        # 3.2. Compute the global standard deviation
        for field in fields:
            new_normalization[field][1] = sqrt(sum([(n / sum(nb_samples)) * std
                                                    for n, std in zip(nb_samples, stds[field])]))

        return new_normalization

    @staticmethod
    def load_partitions_fields(partition: Database,
                               fields: List[str]) -> Dict[str, ndarray]:
        """
        Load all the samples from a Field of a Table in the Database.

        :param partition: Database partition to load.
        :param fields: Data Fields to get.
        """

        partition.load()
        return partition.get_lines(table_name='Training',
                                   fields=fields,
                                   batched=True)

    @staticmethod
    def load_normalization(session_dir: str) -> Dict[str, List[float]]:

        with open(join(session_dir, 'dataset', 'dataset.json')) as json_file:
            json_content = json_load(json_file)
        return json_content['normalization']

    ##########################################################################################
    ##########################################################################################
    #                                     Manager behavior                                   #
    ##########################################################################################
    ##########################################################################################

    def close(self):
        """
        Launch the closing procedure of the DatabaseManager.
        """

        # Compute final normalization if required
        if self.normalize and self.pipeline == 'data_generation':
            self.compute_normalization()

        # Close Database partitions
        self.DB.close()
        self.exchange.close(erase_file=True)

    def __str__(self):

        description = "\n"
        description += f"# {self.__name}\n"
        description += f"    Dataset Repository: {self.database_dir}\n"
        return description
