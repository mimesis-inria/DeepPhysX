from typing import Any, Dict, List, Optional
from os.path import isfile, isdir, join
from os import listdir, symlink, sep
from json import dump as json_dump
from json import load as json_load
from numpy import arange, ndarray, array, abs, mean, sqrt
from numpy.random import shuffle

from SSD.Core.Storage.Database import Database

from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Utils.path import create_dir, copy_dir, get_first_caller
from DeepPhysX.Core.Utils.jsonUtils import CustomJSONEncoder


class DatabaseManager:

    def __init__(self,
                 database_config: Optional[BaseDatabaseConfig] = None,
                 session: str = 'sessions/default',
                 data_manager: Optional[Any] = None,
                 new_session: bool = True,
                 pipeline: str = '',
                 produce_data: bool = True):
        """
        DatabaseManager handle all operations with input / output files. Allows saving and read tensors from files.

        :param database_config: Specialisation containing the parameters of the dataset manager
        :param data_manager: DataManager that handles the DatabaseManager
        :param new_session: Define the creation of new directories to store data
        :param is_training: True if the session is done offline
        :param produce_data: True if this session is a network training
        """

        self.name: str = self.__class__.__name__

        # Manager variables
        database_config = BaseDatabaseConfig() if database_config is None else database_config
        self.data_manager: Optional[Any] = data_manager
        self.dataset_dir: str = join(session, 'dataset')
        self.database: Optional[Database] = None
        self.pipeline: str = pipeline
        root = get_first_caller()

        # Dataset parameters
        self.max_file_size: int = database_config.max_file_size
        self.shuffle: bool = database_config.shuffle
        self.produce_data = produce_data
        self.normalize: bool = database_config.normalize
        self.total_nb_sample: int = 0
        self.recompute_normalization: bool = database_config.recompute_normalization

        # Dataset modes
        self.modes: List[str] = ['training', 'validation', 'prediction']
        self.mode: str = 'training' if produce_data or pipeline == 'training' else 'prediction'
        self.mode = self.mode if database_config.mode is None else database_config.mode

        # Dataset partitions
        session_name = session.split(sep)[-1]
        self.partition_template: Dict[str, str] = {mode: f'{session_name}_{mode}_' + '{}' for mode in self.modes}
        self.partitions: Dict[str, List[str]] = {mode: [] for mode in self.modes}
        self.partition_index: Dict[str, int] = {mode: 0 for mode in self.modes}
        self.current_partition: str = ''

        # Dataset indexing
        self.shuffle_pattern: ndarray = arange(0)
        self.current_sample: int = 1
        self.first_add = True

        # Dataset json file
        self.json_default: Dict[str, Dict[str, Any]] = {'partitions': {mode: [] for mode in self.modes},
                                                        'nb_samples': {mode: [] for mode in self.modes},
                                                        'architecture': {},
                                                        'data_shape': {},
                                                        'normalization': {}}
        self.json_content: Dict[str, Dict[str, Any]] = self.json_default.copy()

        # DataGeneration case
        if self.pipeline == 'data_generation':
            # Generate data in a new session
            if new_session:
                # Generate data from scratch --> create a new directory
                if database_config.existing_dir is None:
                    create_dir(session_dir=session, session_name='dataset')
                    self.create_partition()
                # Complete a Database in a new session --> copy and load the existing directory
                else:
                    copy_dir(src_dir=join(root, database_config.existing_dir), dest_dir=session,
                             sub_folders='dataset')
                    self.load_directory(last_partition=True)
            # Complete a Database in the same session --> load the directory
            else:
                self.load_directory(last_partition=True)

        # Training and prediction cases
        else:

            # Generate data
            if produce_data:
                # Generate data in a new session
                if new_session:
                    # Generate data from scratch --> create a new directory
                    if database_config.existing_dir is None:
                        create_dir(session_dir=session, session_name='dataset')
                        self.create_partition()
                    # Complete a Database in a new session --> copy and load the existing directory
                    else:
                        copy_dir(src_dir=join(root, database_config.existing_dir), dest_dir=session,
                                 sub_folders='dataset')
                        self.load_directory(last_partition=False)
                # Complete a Database in the same directory --> load the directory
                else:
                    self.load_directory(last_partition=False)

            # Load data
            else:
                # Load data in a new session  --> link and load the existing directory
                if new_session:
                    symlink(src=join(root, database_config.existing_dir, 'dataset'),
                            dst=join(session, 'dataset'))
                    self.load_directory(last_partition=False)
                # Load data in the same session  --> load the directory
                else:
                    self.load_directory(last_partition=False)

    def create_partition(self):
        """

        """

        self.current_partition = self.partition_template[self.mode].format(self.partition_index[self.mode])
        self.database = Database(database_dir=self.dataset_dir,
                                 database_name=self.current_partition).new()
        self.database.create_table(table_name='Training')
        self.database.create_table(table_name='Additional')
        self.partitions[self.mode].append(self.current_partition)

    def additional_partition(self):

        # 1. If a DB exists, get all the Fields to re-create them
        fields = {}
        types = {'INT': int, 'FLOAT': float, 'STR': str, 'BOOL': bool, 'NUMPY': ndarray}
        if self.partition_index[self.mode] > 0:
            for table_name in self.database.get_tables():
                fields[table_name] = []
                F = self.database.get_fields(table_name=table_name,
                                             only_names=False)
                for field in [f for f in F if f not in ['id', '_dt_']]:
                    fields[table_name].append((field, types[F[field].field_type]))
            if 'Prediction' in fields.keys():
                self.database.remove_table(table_name='Prediction')

        # 2. Create a new Database
        self.partition_index[self.mode] += 1
        self.create_partition()

        # 3. Re-create the Fields if this is not the first partition
        if self.partition_index[self.mode] > 1:
            for table_name in fields.keys():
                self.database.create_fields(table_name=table_name,
                                            fields=fields[table_name])

        # 4. Tell the other components to communicate on a new DB
        self.data_manager.change_database()

    def load_directory(self,
                       last_partition: bool):
        """

        """

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

        # 5. Load the Database
        if len(self.partitions[self.mode]) == 0:
            self.create_partition()
        else:
            self.partition_index[self.mode] = len(self.partitions[self.mode]) - 1 if last_partition else 0
            self.current_partition = self.partitions[self.mode][self.partition_index[self.mode]]
            self.database = Database(database_dir=self.dataset_dir,
                                     database_name=self.current_partition).load()

        # 6. Shuffle Database indices
        if self.shuffle:
            self.shuffle_samples()

    def load_next_partition(self):

        # 1. Define next partition
        self.partition_index[self.mode] = (self.partition_index[self.mode] + 1) % len(self.partitions[self.mode])
        self.current_partition = self.partitions[self.mode][self.partition_index[self.mode]]
        self.database = Database(database_dir=self.dataset_dir,
                                 database_name=self.current_partition).load()

        # 2. Shuffle
        if self.shuffle:
            self.shuffle_samples()

    def search_partitions(self):
        """

        """

        raw_partitions = {mode: [f for f in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, f))
                                 and f.endswith('.db') and f.__contains__(mode)] for mode in self.modes}
        return {mode: sorted(raw_partitions[mode]) for mode in self.modes}

    def search_partitions_info(self):
        """

        """

        # TODO: fill json

        pass

    def update_json(self,
                    update_partitions_lists: bool = False,
                    update_nb_samples: bool = False,
                    update_architecture: bool = False,
                    update_shapes: bool = False,
                    update_normalization: bool = False) -> None:
        """


        """

        # Update partitions lists
        if update_partitions_lists:
            self.json_content['partitions'] = self.partitions

        # Update number of samples
        if update_nb_samples:
            if len(self.json_content['nb_samples'][self.mode]) == self.partition_index[self.mode] + 1:
                self.json_content['nb_samples'][self.mode][-1] = self.nb_samples
            else:
                self.json_content['nb_samples'][self.mode].append(self.nb_samples)

        # Update DB architecture
        if update_architecture:
            architecture = self.database.get_architecture()
            if 'Prediction' in architecture.keys():
                del architecture['Prediction']
            for fields in architecture.values():
                for field in fields.copy():
                    if field.split(' ')[0] in ['id', '_dt_']:
                        fields.remove(field)
            self.json_content['architecture'] = architecture

        # Update data shapes
        if update_shapes:
            for table_name, fields in self.json_content['architecture'].items():
                if self.database.nb_lines(table_name=table_name) > 0:
                    data = self.database.get_line(table_name=table_name)
                    for field in fields:
                        if 'NUMPY' in field:
                            field_name = field.split(' ')[0]
                            self.json_content['data_shape'][f'{table_name}.{field_name}'] = data[field_name].shape

        # Update normalization coefficients
        if update_normalization:
            self.json_content['normalization'] = self.compute_normalization()

        # Overwrite json file
        with open(join(self.dataset_dir, 'dataset.json'), 'w') as json_file:
            json_dump(self.json_content, json_file, indent=3, cls=CustomJSONEncoder)

    @property
    def nb_samples(self) -> int:
        return self.database.nb_lines(table_name='Training')

    def shuffle_samples(self):
        """

        """

        self.shuffle_pattern = arange(self.nb_samples)
        if self.shuffle:
            shuffle(self.shuffle_pattern)

    def add_data(self,
                 data_lines: Optional[List[int]] = None):
        """

        """

        # 1. Update the json file
        self.update_json(update_nb_samples=True)
        if self.first_add:
            self.update_json(update_partitions_lists=True, update_shapes=True, update_architecture=True)
            self.first_add = False
        if self.normalize and self.mode == 'training' and self.pipeline == 'training' and data_lines is not None:
            self.json_content['normalization'] = self.update_normalization(data_lines=data_lines)
            self.update_json()

        # 2. Check the size of the partition
        if self.max_file_size is not None:
            if self.database.memory_size > self.max_file_size:
                self.additional_partition()
                self.update_json(update_partitions_lists=True, update_nb_samples=True)

        # 3. Update sample counter
        self.current_sample = self.nb_samples + 1

    def get_data(self,
                 batch_size: int) -> List[int]:
        """

        """

        # 1. Check if dataset is loaded and if the current sample is not the last
        if self.current_sample > self.nb_samples:
            self.load_next_partition()
            self.current_sample = 1

        # 2. Update dataset index
        idx = self.current_sample
        self.current_sample += batch_size

        # 3. Get a batch of data
        if self.shuffle:
            return list(self.shuffle_pattern[idx:self.current_sample])
        return list(arange(idx, self.current_sample))

    def close(self):
        """

        """

        if self.normalize and self.pipeline == 'data_generation':
            self.update_json(update_normalization=True)
        self.database.close()

    def change_mode(self, mode: int) -> None:
        """

        """

        pass

    def compute_normalization(self) -> Dict[str, List[float]]:
        """

        """

        # Get the fields to normalize
        fields = []
        for field in self.json_content['data_shape']:
            table_name, field_name = field.split('.')
            fields += [field_name] if table_name == 'Training' else []

        # Init result
        normalization = {field: [0., 0.] for field in fields}

        # Compute the mean for each field
        means = {field: [] for field in fields}
        nb_samples = []
        for partition in self.partitions['training']:
            data_to_normalize = self.load_partitions_fields(partition=partition,
                                                            fields=fields)
            nb_samples.append(data_to_normalize['id'][-1])
            for field in fields:
                data = array(data_to_normalize[field])
                means[field].append(data.mean())
        for field in fields:
            normalization[field][0] = sum([(n / sum(nb_samples)) * m
                                           for n, m in zip(nb_samples, means[field])])

        # Compute the standard deviation for each field
        stds = {field: [] for field in fields}
        for partition in self.partitions['training']:
            data_to_normalize = self.load_partitions_fields(partition=partition,
                                                            fields=fields)
            for field in fields:
                data = array(data_to_normalize[field])
                stds[field].append(mean(abs(data - normalization[field][0]) ** 2))
        for field in fields:
            normalization[field][1] = sqrt(sum([(n / sum(nb_samples)) * std
                                                for n, std in zip(nb_samples, stds[field])]))

        return normalization

    def update_normalization(self,
                             data_lines: List[int]) -> Dict[str, List[float]]:

        previous_normalization = self.normalization
        previous_nb_samples = self.total_nb_sample
        self.total_nb_sample += len(data_lines)

        # First update
        if previous_normalization is None:
            return self.compute_normalization()
        new_normalization = previous_normalization.copy()

        # Compute the mean for each field
        fields = list(previous_normalization.keys())
        data_to_normalize = self.database.get_lines(table_name='Training',
                                                    fields=fields,
                                                    lines_id=data_lines,
                                                    batched=True)
        for field in fields:
            data = array(data_to_normalize[field])
            m = (previous_nb_samples / self.total_nb_sample) * previous_normalization[field][0] + \
                (len(data_lines) / self.total_nb_sample) * data.mean()
            new_normalization[field][0] = m

        # Compute standard deviation for each field
        stds = {field: [] for field in fields}
        nb_samples = []
        for partition in self.partitions['training']:
            data_to_normalize = self.load_partitions_fields(partition=partition,
                                                            fields=fields)
            nb_samples.append(data_to_normalize['id'][-1])
            for field in fields:
                data = array(data_to_normalize[field])
                stds[field].append(mean(abs(data - new_normalization[field][0]) ** 2))
        for field in fields:
            new_normalization[field][1] = sqrt(sum([(n / sum(nb_samples)) * std
                                                    for n, std in zip(nb_samples, stds[field])]))

        return new_normalization

    @property
    def normalization(self) -> Dict[str, List[float]]:
        return None if self.json_content['normalization'] == {} else self.json_content['normalization']

    def load_partitions_fields(self,
                               partition: str,
                               fields: List[str]):
        db = Database(database_dir=self.dataset_dir,
                      database_name=partition).load()
        return db.get_lines(table_name='Training',
                            fields=fields,
                            batched=True)

    def __str__(self):

        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Dataset Repository: {self.dataset_dir}\n"
        size = f"No limits" if self.max_file_size is None else f"{self.max_file_size * 1e-9} Go"
        description += f"    Partitions size: {size}\n"
        return description
