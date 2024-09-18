from typing import Any, Dict, List, Optional
from os.path import isfile, isdir, join
from os import listdir, symlink, sep, remove, rename, makedirs
from json import dump as json_dump
from json import load as json_load
from numpy import arange, ndarray, array, abs, mean, sqrt, empty, concatenate
from numpy.random import shuffle

from SSD.Core.Storage.database import Database

from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.database.database_handler import DatabaseHandler
from DeepPhysX.utils.path import copy_dir
from DeepPhysX.utils.jsonUtils import CustomJSONEncoder


class DatabaseManager:

    def __init__(self,
                 database_config: Optional[DatabaseConfig] = None,
                 data_manager: Optional[Any] = None,
                 pipeline: str = '',
                 new_session: bool = True,
                 session: str = 'sessions/default',
                 produce_data: bool = True):
        """
        DatabaseManager handle all operations with input / output files. Allows saving and read tensors from files.

        :param database_config: Configuration object with the parameters of the Database.
        :param data_manager: DataManager that handles the DatabaseManager.
        :param pipeline: Type of the Pipeline.
        :param new_session: If True, the session is done in a new repository.
        :param session: Path to the session repository.
        :param produce_data: If True, this session will store data in the Database.
        """

        self.name: str = self.__class__.__name__

        # Session variables
        self.pipeline: str = pipeline
        self.data_manager: Optional[Any] = data_manager
        self.database_dir: str = join(session, 'dataset')
        self.database_handlers: List[DatabaseHandler] = []
        database_config = DatabaseConfig() if database_config is None else database_config

        # Dataset parameters
        self.max_file_size: int = database_config.max_file_size
        self.shuffle: bool = database_config.shuffle
        self.produce_data = produce_data
        self.normalize: bool = database_config.normalize
        self.total_nb_sample: int = 0
        self.recompute_normalization: bool = database_config.recompute_normalization

        # Dataset modes
        self.modes: List[str] = ['training', 'validation', 'prediction']
        if pipeline == 'data_generation':
            self.mode: str = 'training' if database_config.mode is None else database_config.mode
        elif pipeline == 'training':
            self.mode: str = 'training'
        else:
            self.mode: str = 'prediction' if database_config.mode is None else database_config.mode
            self.mode = 'prediction' if produce_data else self.mode

        # Dataset partitions
        session_name = session.split(sep)[-1]
        self.partition_template: Dict[str, str] = {mode: f'{session_name}_{mode}_' + '{}' for mode in self.modes}
        self.partition_index: Dict[str, int] = {mode: 0 for mode in self.modes}
        self.partition_names: Dict[str, List[str]] = {mode: [] for mode in self.modes}
        self.partitions: Dict[str, List[Database]] = {mode: [] for mode in self.modes}

        # Dataset indexing
        self.sample_indices: ndarray = empty((0, 2), dtype=int)
        self.sample_id: int = 0
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
                    makedirs(join(session, 'dataset'))
                    self.create_partition()
                # Complete a Database in a new session --> copy and load the existing directory
                else:
                    copy_dir(src_dir=database_config.existing_dir, dest_dir=session, sub_folders='dataset')
                    self.load_directory(rename_partitions=True)
            # Complete a Database in the same session --> load the directory
            else:
                self.load_directory()

        # Training case
        elif self.pipeline == 'training':

            # Generate data
            if produce_data:
                # Generate data in a new session
                if new_session:
                    # Generate data from scratch --> create a new directory
                    if database_config.existing_dir is None:
                        makedirs(join(session, 'dataset'))
                        self.create_partition()
                    # Complete a Database in a new session --> copy and load the existing directory
                    else:
                        copy_dir(src_dir=database_config.existing_dir, dest_dir=session, sub_folders='dataset')
                        self.load_directory()
                # Complete a Database in the same directory --> load the directory
                else:
                    self.load_directory()

            # Load data
            else:
                # Load data in a new session  --> link and load the existing directory
                if new_session:
                    symlink(src=join(database_config.existing_dir, 'dataset'),
                            dst=join(session, 'dataset'))
                    self.load_directory()
                # Load data in the same session  --> load the directory
                else:
                    self.load_directory()

        # Prediction case
        else:
            self.load_directory()

        # Finally create an exchange database
        self.exchange = Database(database_dir=self.database_dir,
                                 database_name='Exchange').new(remove_existing=True)
        self.exchange.create_table(table_name='Exchange')

    ##########################################################################################
    ##########################################################################################
    #                                   Partitions Management                                #
    ##########################################################################################
    ##########################################################################################

    def load_directory(self,
                       rename_partitions: bool = False) -> None:
        """
        Get the Database information from the json file (partitions, samples, etc).
        Load all the partitions or create one if necessary.

        :param rename_partitions: If True, the existing partitions should be renamed to match the session name.
        """

        # 1. Check the directory existence to prevent bugs
        if not isdir(self.database_dir):
            raise Warning(f"[{self.name}] Impossible to load Dataset from {self.database_dir}.")

        # 2. Get the .json description file
        json_found = False
        if isfile(join(self.database_dir, 'dataset.json')):
            json_found = True
            with open(join(self.database_dir, 'dataset.json')) as json_file:
                self.json_content = json_load(json_file)

        # 3. Update json file if not found
        if not json_found or self.json_content == self.json_default:
            self.search_partitions_info()
            self.update_json()

        # 4. Load partitions for each mode
        self.partition_names = self.json_content['partitions']
        self.partition_index = {mode: len(self.partition_names[mode]) for mode in self.modes}
        if rename_partitions:
            for mode in self.modes:
                current_name = self.partition_template[mode].split(f'_{mode}_')[0]
                for i, name in enumerate(self.partition_names[mode]):
                    if name.split(f'_{mode}_')[0] != current_name:
                        self.partition_names[mode][i] = current_name + f'_{mode}_{i}'
                        rename(src=join(self.database_dir, f'{name}.db'),
                               dst=join(self.database_dir, f'{self.partition_names[mode][i]}.db'))

        # 5. Load the partitions
        for mode in self.modes:
            for name in self.partition_names[mode]:
                db = Database(database_dir=self.database_dir,
                              database_name=name).load()
                self.partitions[mode].append(db)
        if len(self.partitions[self.mode]) == 0:
            self.create_partition()
        elif self.max_file_size is not None and self.partitions[self.mode][-1].memory_size > self.max_file_size \
                and self.produce_data:
            self.create_partition()

        # 6. Index partitions
        self.index_samples()

        # 7. Check normalization
        if self.recompute_normalization or (
                self.normalize and self.json_content['normalization'] == self.json_default['normalization']):
            self.json_content['normalization'] = self.compute_normalization()
            self.update_json()

    def create_partition(self) -> None:
        """
        Create a new partition of the Database.
        """

        # 1. Define the partition name
        partition_name = self.partition_template[self.mode].format(self.partition_index[self.mode])
        self.partition_names[self.mode].append(partition_name)

        # 2. Create the Database partition
        db = Database(database_dir=self.database_dir,
                      database_name=partition_name).new()
        db.create_table(table_name='Training')
        db.create_table(table_name='Additional')
        self.partitions[self.mode].append(db)

        # 3. If the partition is an additional one, create all fields
        if self.partition_index[self.mode] > 0:
            # Get fields
            fields = {}
            types = {'INT': int, 'FLOAT': float, 'STR': str, 'BOOL': bool, 'NUMPY': ndarray}
            for table_name in self.partitions[self.mode][0].get_tables():
                fields[table_name] = []
                F = self.partitions[self.mode][0].get_fields(table_name=table_name,
                                                             only_names=False)
                for field in [f for f in F if f not in ['id', '_dt_']]:
                    fields[table_name].append((field, types[F[field].field_type]))
            # Re-create them
            for table_name in fields.keys():
                self.partitions[self.mode][-1].create_fields(table_name=table_name,
                                                             fields=fields[table_name])
        else:
            self.partitions[self.mode][-1].create_fields(table_name='Training',
                                                         fields=('env_id', int))
            self.partitions[self.mode][-1].create_fields(table_name='Additional',
                                                         fields=('env_id', int))

        # 4. Update the partitions in handlers
        for handler in self.database_handlers:
            handler.update_list_partitions(self.partitions[self.mode][-1])
        self.partition_index[self.mode] += 1
        self.json_content['partitions'] = self.partition_names
        self.get_nb_samples()
        self.update_json()

    def get_partition_objects(self) -> List[Database]:
        """
        Get the list of partitions of the Database for the current mode.
        """

        return self.partitions[self.mode]

    def get_partition_names(self) -> List[List[str]]:
        """
        Get the list of partition paths of the Database for the current mode.
        """

        return [db.get_path() for db in self.partitions[self.mode]]

    def remove_empty_partitions(self):
        """
        Remove every empty partitions of the Database.
        """

        for mode in self.modes:
            # Check if the last partition for the mode is empty
            if len(self.partitions[mode]) > 0 and self.partitions[mode][-1].nb_lines(table_name='Training') == 0:
                # Erase partition file
                path = self.partitions[mode].pop(-1).get_path()
                remove(join(path[0], f'{path[1]}.db'))
                # Remove from information
                self.partition_names[mode].pop(-1)
                self.partition_index[mode] -= 1
                self.json_content['nb_samples'][mode].pop(-1)
                self.json_content['partitions'] = self.partition_names
                self.update_json()

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

    def search_partitions_info(self) -> None:
        """
        Get the information about the Database manually if the json file is not found.
        """

        # 1. Get all the partitions
        raw_partitions = {mode: [f for f in listdir(self.database_dir) if isfile(join(self.database_dir, f))
                                 and f.endswith('.db') and f.__contains__(mode)] for mode in self.modes}
        raw_partitions = {mode: [f.split('.')[0] for f in raw_partitions[mode]] for mode in self.modes}
        self.json_content['partitions'] = {mode: sorted(raw_partitions[mode]) for mode in self.modes}

        # 2. Get the number of samples
        for mode in self.modes:
            for name in self.json_content['partitions'][mode]:
                db = Database(database_dir=self.database_dir,
                              database_name=name).load()
                self.json_content['nb_samples'][mode].append(db.nb_lines(table_name='Training'))
                db.close()

        # 3. Get the Database architecture
        self.json_content['architecture'] = self.get_database_architecture()
        self.first_add = False

        # 4. Get the data shapes
        self.json_content['data_shape'] = self.get_data_shapes()

    def get_database_architecture(self) -> Dict[str, List[str]]:
        """
        Get the Tables and Fields structure of the Database.
        """

        # Get a training or validation partition
        if len(self.json_content['partitions']['training']) != 0:
            db = Database(database_dir=self.database_dir,
                          database_name=self.json_content['partitions']['training'][0]).load()
        elif len(self.json_content['partitions']['validation']) != 0:
            db = Database(database_dir=self.database_dir,
                          database_name=self.json_content['partitions']['validation'][0]).load()
        else:
            return {}

        # Get the architecture, keep relevant fields only
        architecture = db.get_architecture()
        for fields in architecture.values():
            for field in fields.copy():
                if field.split(' ')[0] in ['id', '_dt_']:
                    fields.remove(field)
        db.close()

        return architecture

    def get_data_shapes(self) -> Dict[str, List[int]]:
        """
        Get the shape of data Fields.
        """

        # Get a training or validation partition
        if len(self.json_content['partitions']['training']) != 0:
            db = Database(database_dir=self.database_dir,
                          database_name=self.json_content['partitions']['training'][0]).load()
        elif len(self.json_content['partitions']['validation']) != 0:
            db = Database(database_dir=self.database_dir,
                          database_name=self.json_content['partitions']['validation'][0]).load()
        else:
            return {}

        # Get the data shape for each numpy Field
        shapes = {}
        for table_name, fields in self.json_content['architecture'].items():
            if db.nb_lines(table_name=table_name) > 0:
                data = db.get_line(table_name=table_name)
                for field in fields:
                    if 'NUMPY' in field:
                        field_name = field.split(' ')[0]
                        shapes[f'{table_name}.{field_name}'] = data[field_name].shape
        db.close()

        return shapes

    def get_nb_samples(self) -> None:
        """
        Get the number of sample in each partition.
        """

        nb_samples = self.partitions[self.mode][-1].nb_lines(table_name='Training')
        if len(self.json_content['nb_samples'][self.mode]) == self.partition_index[self.mode]:
            self.json_content['nb_samples'][self.mode][-1] = nb_samples
        else:
            self.json_content['nb_samples'][self.mode].append(nb_samples)

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

    def connect_handler(self,
                        handler: DatabaseHandler) -> None:
        """
        Add and init a new DatabaseHandler to the list.

        :param handler: New DatabaseHandler.
        """

        handler.init(storing_partitions=self.get_partition_objects(),
                     exchange_db=self.exchange)
        self.database_handlers.append(handler)

    def index_samples(self) -> None:
        """
        Create a new indexing list of samples. Samples are identified by [partition_id, line_id].
        """

        # Create the indices for each sample such as [partition_id, line_id]
        for i, nb_sample in enumerate(self.json_content['nb_samples'][self.mode]):
            partition_indices = empty((nb_sample, 2), dtype=int)
            partition_indices[:, 0] = i
            partition_indices[:, 1] = arange(1, nb_sample + 1)
            self.sample_indices = concatenate((self.sample_indices, partition_indices))
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
        self.get_nb_samples()
        self.update_json()
        # 1.1. Init partitions information on the first sample
        if self.first_add:
            for handler in self.database_handlers:
                handler.load()
            self.json_content['partitions'] = self.partition_names
            self.json_content['architecture'] = self.get_database_architecture()
            self.json_content['data_shape'] = self.get_data_shapes()
            self.update_json()
            self.first_add = False
        # 1.2. Update the normalization coefficients if required
        if self.normalize and self.mode == 'training' and self.pipeline == 'training' and data_lines is not None:
            self.json_content['normalization'] = self.update_normalization(data_lines=data_lines)
            self.update_json()

        # 2. Check the size of the current partition
        if self.max_file_size is not None:
            if self.partitions[self.mode][-1].memory_size > self.max_file_size:
                self.create_partition()

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

        if self.json_content['normalization'] == {} or not self.normalize:
            return None
        return self.json_content['normalization']

    def compute_normalization(self) -> Dict[str, List[float]]:
        """
        Compute the mean and the standard deviation of all the training samples for each data field.
        """

        # 1. Get the fields to normalize
        fields = []
        for field in self.json_content['data_shape']:
            table_name, field_name = field.split('.')
            fields += [field_name] if table_name == 'Training' else []
        normalization = {field: [0., 0.] for field in fields}

        # 2. Compute the mean of samples for each field
        means = {field: [] for field in fields}
        nb_samples = []
        # 2.1. Compute the mean for each partition
        for partition in self.partitions['training']:
            data_to_normalize = self.load_partitions_fields(partition=partition, fields=fields)
            nb_samples.append(data_to_normalize['id'][-1])
            for field in fields:
                data = array(data_to_normalize[field])
                means[field].append(data.mean())
        # 2.2. Compute the global mean
        for field in fields:
            normalization[field][0] = sum([(n / sum(nb_samples)) * m
                                           for n, m in zip(nb_samples, means[field])])

        # 3. Compute the standard deviation of samples for each field
        stds = {field: [] for field in fields}
        # 3.1. Compute the standard deviation for each partition
        for partition in self.partitions['training']:
            data_to_normalize = self.load_partitions_fields(partition=partition,
                                                            fields=fields)
            for field in fields:
                data = array(data_to_normalize[field])
                stds[field].append(mean(abs(data - normalization[field][0]) ** 2))
        # 3.2. Compute the global standard deviation
        for field in fields:
            normalization[field][1] = sqrt(sum([(n / sum(nb_samples)) * std
                                                for n, std in zip(nb_samples, stds[field])]))
        return normalization

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

        # Check non-empty last partition
        self.remove_empty_partitions()

        # Compute final normalization if required
        if self.normalize and self.pipeline == 'data_generation':
            self.json_content['normalization'] = self.compute_normalization()
            self.update_json()

        # Close Database partitions
        for mode in self.modes:
            for database in self.partitions[mode]:
                database.close()
        self.exchange.close(erase_file=True)

    def __str__(self):

        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Dataset Repository: {self.database_dir}\n"
        size = f"No limits" if self.max_file_size is None else f"{self.max_file_size * 1e-9} Gb"
        description += f"    Partitions size: {size}\n"
        return description
