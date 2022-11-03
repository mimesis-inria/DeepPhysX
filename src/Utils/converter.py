from typing import Dict, List, Optional
from os import listdir
from os.path import join, exists, isfile, dirname, sep
from numpy import ndarray, load
from vedo import ProgressBar

from DeepPhysX.Core.Utils.path import get_first_caller, create_dir
from DeepPhysX.Core.Manager.DatabaseManager import DatabaseManager
from DeepPhysX.Core.Database.DatabaseHandler import DatabaseHandler
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig


class DatasetConverter:

    def __init__(self,
                 session_path: str):
        """
        Convert a Dataset between the previous Numpy partitions and the new SQL Database.

        :param session_path: Relative path to the session to convert.
        """

        root = get_first_caller()
        self.dataset_dir = join(root, session_path, 'dataset')
        if not exists(self.dataset_dir):
            raise ValueError(f"The given path does not exist: {self.dataset_dir}")

    def numpy_to_database(self,
                          batch_size: int,
                          max_file_size: Optional[float] = None,
                          normalize: bool = False):
        """
        Convert a Database from the previous Numpy partitions to the new SQL Database.
        """

        # 1. Get the partitions files
        partitions = self.load_numpy_partitions()

        # 2. Create a new repository
        session_name = f'{self.dataset_dir.split(sep)[-2]}_converted'
        session = create_dir(session_dir=dirname(dirname(self.dataset_dir)),
                             session_name=session_name)
        database_manager = DatabaseManager(pipeline='data_generation',
                                           session=session)
        database_manager.close()

        # 3. Create the Database for each mode
        for mode in partitions.keys():
            print(f"\nConverting {mode} partitions...")

            # 3.1. Create a DatabaseManager with a DatabaseHandler
            database_config = BaseDatabaseConfig(mode=mode,
                                                 max_file_size=max_file_size,
                                                 normalize=normalize)
            database_manager = DatabaseManager(database_config=database_config,
                                               pipeline='data_generation',
                                               session=session,
                                               new_session=False)
            database_handler = DatabaseHandler()
            database_manager.connect_handler(database_handler)

            # 3.2. Create Fields in Tables
            training_fields = []
            additional_fields = []
            nb_partition = 0
            for field in partitions[mode].keys():
                if field in ['input', 'ground_truth']:
                    training_fields.append((field, ndarray))

                else:
                    additional_fields.append((field.split('_')[-1], ndarray))
                if nb_partition == 0:
                    nb_partition = len(partitions[mode][field])
                elif len(partitions[mode][field]) != nb_partition:
                    raise ValueError(f"The number of partition is not consistent in {mode} mode.")
            database_handler.create_fields(table_name='Training', fields=training_fields)
            database_handler.create_fields(table_name='Additional', fields=additional_fields)

            # 3.3. Add each partition to the Database
            if nb_partition == 0:
                print("   No partition.")
            for i in range(nb_partition):
                data_training = {field: load(join(self.dataset_dir, f'{partitions[mode][field][i]}.npy'))
                                 for field, _ in training_fields}
                data_additional = {field: load(join(self.dataset_dir, f'{partitions[mode][field][i]}.npy'))
                                   for field, _ in additional_fields}
                nb_sample = 0
                for data in [data_training, data_additional]:
                    for field in data.keys():
                        if nb_sample == 0:
                            nb_sample = data[field].shape[0]
                        elif data[field].shape[0] != nb_sample:
                            raise ValueError(f"The number of sample is not consistent in {mode} mode.")
                id_sample = 0
                pb = ProgressBar(0, nb_sample, c='orange', title=f"  Loading partition {i + 1}/{nb_partition}")
                while id_sample < nb_sample:
                    last_sample = min(nb_sample, id_sample + batch_size)
                    if len(data_training.keys()) > 0:
                        sample_training = {field: data_training[field][id_sample:last_sample]
                                           for field in data_training.keys()}
                        database_handler.add_batch(table_name='Training', batch=sample_training)
                    if len(data_additional.keys()) > 0:
                        sample_additional = {field: data_additional[field][id_sample:last_sample]
                                             for field in data_additional.keys()}
                        database_handler.add_batch(table_name='Additional', batch=sample_additional)
                    database_manager.add_data()
                    id_sample += batch_size
                    pb.print(counts=id_sample)

            # 3. Close DatabaseManager
            database_manager.close()

        print("\nConversion done.")

    def load_numpy_partitions(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get all the partition files in the repository. Do not use the JSON file to prevent bugs.
        """

        # 1. Get the partition files for each mode
        modes = ['training', 'validation', 'prediction']
        partitions = {mode: [f.split('.')[0] for f in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, f))
                             and f.endswith('.npy') and f.__contains__(mode)] for mode in modes}

        # 2. Sort partitions by field (IN, OUT, ADD) and by name
        sorted_partitions = {mode: {field[1:-1]: sorted([f for f in partitions[mode] if f.__contains__(field)])
                                    for field in ['_IN_', '_OUT_', '_ADD_']}
                             for mode in modes}
        all_partitions = {}
        for mode in modes:
            all_partitions[mode] = {}
            for field, name in zip(['IN', 'OUT', 'ADD'], ['input', 'ground_truth', '']):
                for partition in sorted_partitions[mode][field]:
                    # Extract information from the filename
                    partition_name = partition.split('_')
                    partition_name = partition_name[partition_name.index(field):]
                    # Additional data: <session_name>_<field>_<name>_<id>
                    if len(partition_name) == 3:
                        name = partition_name[-2]
                    # Add partition
                    if name not in all_partitions[mode]:
                        all_partitions[mode][name] = []
                    all_partitions[mode][name].append(partition)

        return all_partitions

