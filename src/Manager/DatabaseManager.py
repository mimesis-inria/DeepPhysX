from typing import Any, Dict, List, Optional
from os.path import isfile, isdir, join
from os import listdir, symlink, sep, remove, rename
from json import dump as json_dump
from json import load as json_load
from numpy import arange, ndarray, array, abs, mean, sqrt, empty, concatenate
from numpy.random import shuffle

from SSD.Core.Storage.Database import Database

from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Database.DatabaseHandler import DatabaseHandler
from DeepPhysX.Core.Utils.path import create_dir, copy_dir, get_first_caller
from DeepPhysX.Core.Utils.jsonUtils import CustomJSONEncoder


class DatabaseManager:

    def __init__(self,
                 database_config: Optional[BaseDatabaseConfig] = None,
                 data_manager: Optional[Any] = None,
                 pipeline: str = '',
                 session: str = 'sessions/default',
                 new_session: bool = True,
                 produce_data: bool = True):
        """
        DatabaseManager handle all operations with input / output files. Allows saving and read tensors from files.

        :param database_config: Specialisation containing the parameters of the dataset manager
        :param data_manager: DataManager that handles the DatabaseManager
        :param new_session: Define the creation of new directories to store data
        :param produce_data: True if this session is a network training
        """

        self.name: str = self.__class__.__name__

        # Manager variables
        self.pipeline: str = pipeline
        self.data_manager: Optional[Any] = data_manager
        self.database_dir: str = join(session, 'dataset')
        self.database_handlers: List[DatabaseHandler] = []
        root = get_first_caller()
        database_config = BaseDatabaseConfig() if database_config is None else database_config

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
                    create_dir(session_dir=session, session_name='dataset')
                    self.create_partition()
                # Complete a Database in a new session --> copy and load the existing directory
                else:
                    copy_dir(src_dir=join(root, database_config.existing_dir), dest_dir=session,
                             sub_folders='dataset')
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
                        create_dir(session_dir=session, session_name='dataset')
                        self.create_partition()
                    # Complete a Database in a new session --> copy and load the existing directory
                    else:
                        copy_dir(src_dir=join(root, database_config.existing_dir), dest_dir=session,
                                 sub_folders='dataset')
                        self.load_directory()
                # Complete a Database in the same directory --> load the directory
                else:
                    self.load_directory()

            # Load data
            else:
                # Load data in a new session  --> link and load the existing directory
                if new_session:
                    symlink(src=join(root, database_config.existing_dir, 'dataset'),
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
                                 database_name='Exchange').new()
        self.exchange.create_table(table_name='Exchange')

    #########################
    # Partitions Management #
    #########################

    def load_directory(self,
                       rename_partitions: bool = False):
        """

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
        if self.recompute_normalization or (
                self.normalize and self.json_content['normalization'] == self.json_default['normalization']):
            self.update_json(update_normalization=True)

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
        elif self.max_file_size is not None and self.partitions[self.mode][-1].memory_size > self.max_file_size:
            self.create_partition()

        # 6. Index partitions
        self.index_samples()

    def create_partition(self):
        """

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

        # 4. Update the partitions in handlers
        for handler in self.database_handlers:
            handler.update_list_partitions(self.partitions[self.mode][-1])
        self.partition_index[self.mode] += 1
        self.update_json(update_partitions_lists=True, update_nb_samples=True)

    def get_partition_objects(self) -> List[Database]:
        return self.partitions[self.mode]

    def get_partition_names(self) -> List[List[str]]:
        return [db.get_path() for db in self.partitions[self.mode]]

    def remove_empty_partitions(self):

        for mode in self.modes:
            if len(self.partitions[mode]) > 0 and self.partitions[mode][-1].nb_lines(table_name='Training') == 0:
                database = self.partitions[mode].pop(-1)
                path = database.get_path()
                remove(join(path[0], f'{path[1]}.db'))
                self.partition_names[mode].pop(-1)
                self.partition_index[mode] -= 1
                self.json_content['nb_samples'][mode].pop(-1)
                self.update_json(update_partitions_lists=True)

    ##################
    # JSON info file #
    ##################

    def search_partitions_info(self):
        """

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

    def get_database_architecture(self):
        if len(self.json_content['partitions']['training']) != 0:
            db = Database(database_dir=self.database_dir,
                          database_name=self.json_content['partitions']['training'][0]).load()
        elif len(self.json_content['partitions']['validation']) != 0:
            db = Database(database_dir=self.database_dir,
                          database_name=self.json_content['partitions']['validation'][0]).load()
        else:
            return {}
        architecture = db.get_architecture()
        if 'Prediction' in architecture.keys():
            del architecture['Prediction']
        for fields in architecture.values():
            for field in fields.copy():
                if field.split(' ')[0] in ['id', '_dt_']:
                    fields.remove(field)
        db.close()
        return architecture

    def get_data_shapes(self):
        if len(self.json_content['partitions']['training']) != 0:
            db = Database(database_dir=self.database_dir,
                          database_name=self.json_content['partitions']['training'][0]).load()
        elif len(self.json_content['partitions']['validation']) != 0:
            db = Database(database_dir=self.database_dir,
                          database_name=self.json_content['partitions']['validation'][0]).load()
        else:
            return {}
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
            self.json_content['partitions'] = self.partition_names

        # Update number of samples
        if update_nb_samples:
            if len(self.json_content['nb_samples'][self.mode]) == self.partition_index[self.mode]:
                self.json_content['nb_samples'][self.mode][-1] = self.nb_samples
            else:
                self.json_content['nb_samples'][self.mode].append(self.nb_samples)

        # Update DB architecture
        if update_architecture:
            self.json_content['architecture'] = self.get_database_architecture()

        # Update data shapes
        if update_shapes:
            self.json_content['data_shape'] = self.get_data_shapes()

        # Update normalization coefficients
        if update_normalization:
            self.json_content['normalization'] = self.compute_normalization()

        # Overwrite json file
        with open(join(self.database_dir, 'dataset.json'), 'w') as json_file:
            json_dump(self.json_content, json_file, indent=3, cls=CustomJSONEncoder)

    #########################
    # Database read / write #
    #########################

    def connect_handler(self,
                        handler: DatabaseHandler) -> None:

        handler.init(storing_partitions=self.get_partition_objects(),
                     exchange_db=self.exchange)
        self.database_handlers.append(handler)

    def index_samples(self):

        for i, nb_sample in enumerate(self.json_content['nb_samples'][self.mode]):
            partition_indices = empty((nb_sample, 2), dtype=int)
            partition_indices[:, 0] = i
            partition_indices[:, 1] = arange(1, nb_sample + 1)
            self.sample_indices = concatenate((self.sample_indices, partition_indices))
        self.sample_id = 0
        if self.shuffle:
            shuffle(self.sample_indices)

    @property
    def nb_samples(self) -> int:
        return self.partitions[self.mode][-1].nb_lines(table_name='Training')

    def add_data(self,
                 data_lines: Optional[List[int]] = None):
        """

        """

        # 1. Update the json file
        self.update_json(update_nb_samples=True)
        if self.first_add:
            for handler in self.database_handlers:
                handler.load()
            self.update_json(update_partitions_lists=True, update_shapes=True, update_architecture=True)
            self.first_add = False
        if self.normalize and self.mode == 'training' and self.pipeline == 'training' and data_lines is not None:
            self.json_content['normalization'] = self.update_normalization(data_lines=data_lines)
            self.update_json()

        # 2. Check the size of the partition
        if self.max_file_size is not None:
            if self.partitions[self.mode][-1].memory_size > self.max_file_size:
                self.create_partition()

    def get_data(self,
                 batch_size: int) -> List[List[int]]:
        """

        """

        # 1. Check if dataset is loaded and if the current sample is not the last
        if self.sample_id >= len(self.sample_indices):
            self.index_samples()

        # 2. Update dataset index
        idx = self.sample_id
        self.sample_id += batch_size

        # 3. Get a batch of data
        lines = self.sample_indices[idx:self.sample_id].tolist()

        # 4. Ensure the batch has th good size
        if len(lines) < batch_size:
            lines += self.get_data(batch_size=batch_size - len(lines))
        return lines

    ############
    # Behavior #
    ############

    def close(self):
        """

        """

        # Check non-empty last partition
        self.remove_empty_partitions()

        if self.normalize and self.pipeline == 'data_generation':
            self.update_json(update_normalization=True)

        # Cose databases
        for mode in self.modes:
            for database in self.partitions[mode]:
                database.close()
        self.exchange.close(erase_file=True)

    def change_mode(self, mode: int) -> None:
        """

        """

        pass

    #################
    # Normalization #
    #################

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
        data_to_normalize = self.partitions[self.mode][-1].get_lines(table_name='Training',
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
        return None if self.json_content['normalization'] == {} or not self.normalize else self.json_content['normalization']

    def load_partitions_fields(self,
                               partition: Database,
                               fields: List[str]):
        partition.load()
        return partition.get_lines(table_name='Training',
                                   fields=fields,
                                   batched=True)

    def __str__(self):

        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Dataset Repository: {self.database_dir}\n"
        size = f"No limits" if self.max_file_size is None else f"{self.max_file_size * 1e-9} Go"
        description += f"    Partitions size: {size}\n"
        return description
