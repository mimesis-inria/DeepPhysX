from typing import Union, List, Dict, Any, Callable, Optional, Type, Tuple
from numpy import array, where
from itertools import chain

from SSD.Core.Storage.Database import Database


class DatabaseHandler:

    def __init__(self,
                 remote: bool = False,
                 on_init_handler: Optional[Callable] = None):

        self.remote = remote
        self.__storing_partitions: List[Database] = []
        self.__exchange_db: Optional[Database] = None
        self.__on_init_handler = self.__default_handler if on_init_handler is None else on_init_handler

    def __default_handler(self):
        pass

    ###
    # Manage partitions
    ###

    def init(self,
             storing_partitions: List[Database],
             exchange_db: Database) -> None:
        self.__storing_partitions = storing_partitions.copy()
        self.__exchange_db = exchange_db
        self.__on_init_handler()

    def init_remote(self,
                    storing_partitions: List[List[str]],
                    exchange_db: List[str]) -> None:
        self.__storing_partitions = [Database(database_dir=partition[0],
                                              database_name=partition[1]).load() for partition in storing_partitions]
        self.__exchange_db = Database(database_dir=exchange_db[0],
                                      database_name=exchange_db[1]).load()
        self.__on_init_handler()

    def update_list_partitions(self,
                               partition: Database) -> None:
        self.__storing_partitions.append(partition)

    def get_database_dir(self):
        return self.__storing_partitions[0].get_path()[0]

    def load(self):
        for db in self.__storing_partitions:
            db.load()
        self.__exchange_db.load()

    def get_partitions(self):
        return self.__storing_partitions

    ###
    # Database Architecture
    ###

    def create_fields(self,
                      table_name: str,
                      fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:

        if table_name == 'Exchange':
            self.__exchange_db.load()
            if len(self.__exchange_db.get_fields(table_name=table_name)) == 1:
                self.__exchange_db.create_fields(table_name=table_name,
                                                 fields=fields)
        else:
            if len(self.__storing_partitions) == 1:
                self.__storing_partitions[0].load()
                if len(self.__storing_partitions[0].get_fields(table_name=table_name)) <= 2:
                    self.__storing_partitions[0].create_fields(table_name=table_name,
                                                               fields=fields)

    def get_fields(self,
                   table_name: str) -> List[str]:

        database = self.__exchange_db if table_name == 'Exchange' else self.__storing_partitions[0]
        return database.get_fields(table_name=table_name)

    ###
    # Data Read / Write
    ###

    def add_data(self,
                 table_name: str,
                 data: Dict[str, Any]) -> Union[int, List[int]]:

        if table_name == 'Exchange':
            return self.__exchange_db.add_data(table_name=table_name, data=data)
        else:
            return [len(self.__storing_partitions) - 1,
                    self.__storing_partitions[-1].add_data(table_name=table_name, data=data)]

    def update(self,
               table_name: str,
               data: Dict[str, Any],
               line_id: Union[int, List[int]]) -> None:

        database = self.__exchange_db if table_name == 'Exchange' else self.__storing_partitions[line_id[0]]
        line_id = line_id[1] if type(line_id) == list else line_id
        database.update(table_name=table_name, data=data, line_id=line_id)

    def get_line(self,
                 table_name: str,
                 line_id: Union[int, List[int]],
                 fields: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:

        database = self.__exchange_db if table_name == 'Exchange' else self.__storing_partitions[line_id[0]]
        line_id = line_id[1] if type(line_id) == list else line_id
        return database.get_line(table_name=table_name,
                                 line_id=line_id,
                                 fields=fields)

    def get_lines(self,
                  table_name: str,
                  lines_id: List[List[int]],
                  fields: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:

        # Transform list of lines to batch of lines per partition
        batch_indices = array(lines_id)
        partition_batch_indices = []
        for i in range(len(self.__storing_partitions)):
            partition_indices = where(batch_indices[:, 0] == i)[0]
            if len(partition_indices) > 0:
                partition_batch_indices.append([i, batch_indices[partition_indices, 1].tolist()])

        # Get lines of data
        partition_batches = []
        for partition_indices in partition_batch_indices:
            data = self.__storing_partitions[partition_indices[0]].get_lines(table_name=table_name,
                                                                             lines_id=partition_indices[1],
                                                                             fields=fields,
                                                                             batched=True)
            del data['id']
            partition_batches.append(data)

        # Merge batches
        if len(partition_batches) == 1:
            return partition_batches[0]
        else:
            return dict(zip(partition_batches[0].keys(),
                            [list(chain.from_iterable([partition_batches[i][key]
                                                       for i in range(len(partition_batches))]))
                             for key in partition_batches[0].keys()]))
