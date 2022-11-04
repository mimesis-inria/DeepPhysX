from typing import Union, List, Dict, Any, Callable, Optional, Type, Tuple
from numpy import array, where
from itertools import chain

from SSD.Core.Storage.Database import Database


class DatabaseHandler:

    def __init__(self,
                 on_init_handler: Optional[Callable] = None,
                 on_partitions_handler: Optional[Callable] = None):
        """
        DatabaseHandler allows components to be synchronized with the Database partitions and to read / write data.

        :param on_init_handler: Event to trigger when the DatabaseHandler is initialized.
        :param on_partitions_handler: Event to trigger when the list of partitions is updated.
        """

        # Databases variables
        self.__storing_partitions: List[Database] = []
        self.__exchange_db: Optional[Database] = None

        # Event handlers
        self.__on_init_handler = self.default_handler if on_init_handler is None else on_init_handler
        self.__on_partitions_handler = self.default_handler if on_partitions_handler is None else on_partitions_handler

    def default_handler(self):
        pass

    ##########################################################################################
    ##########################################################################################
    #                                 Partitions management                                  #
    ##########################################################################################
    ##########################################################################################

    def init(self,
             storing_partitions: List[Database],
             exchange_db: Database) -> None:
        """
        Initialize the list of the partitions.

        :param storing_partitions: List of the storing Database partitions.
        :param exchange_db: Exchange Database.
        """

        self.__storing_partitions = storing_partitions.copy()
        self.__exchange_db = exchange_db
        self.__on_init_handler()

    def init_remote(self,
                    storing_partitions: List[List[str]],
                    exchange_db: List[str]) -> None:
        """
        Initialize the list of partitions in remote DatabaseHandlers.

        :param storing_partitions: List of paths to the storing Database partitions.
        :param exchange_db: Path to the exchange Database.
        """

        self.__storing_partitions = [Database(database_dir=partition[0],
                                              database_name=partition[1]).load() for partition in storing_partitions]
        self.__exchange_db = Database(database_dir=exchange_db[0],
                                      database_name=exchange_db[1]).load()
        self.__on_init_handler()

    def update_list_partitions(self,
                               partition: Database) -> None:
        """
        Add a new storing partition to the list.

        :param partition: New storing partition to add.
        """

        self.__storing_partitions.append(partition)
        self.__on_partitions_handler()

    def update_list_partitions_remote(self,
                                      partition: List[str]) -> None:
        """
        Add a new storing partition to the list in remote DatabaseHandler.

        :param partition: Path to the new storing partition.
        """

        self.__storing_partitions.append(Database(database_dir=partition[0],
                                                  database_name=partition[1]).load())
        self.__on_partitions_handler()

    def load(self) -> None:
        """
        Load the Database partitions stored by the component.
        """

        for db in self.__storing_partitions:
            db.load()
        if self.__exchange_db is not None:
            self.__exchange_db.load()

    def get_database_dir(self) -> str:
        """
        Get the database repository of the session.
        """

        return self.__storing_partitions[0].get_path()[0]

    def get_partitions(self) -> List[Database]:
        """
        Get the storing Database partitions.
        """

        return self.__storing_partitions

    def get_exchange(self) -> Database:
        """
        Get the exchange Database.
        """

        return self.__exchange_db

    ##########################################################################################
    ##########################################################################################
    #                                 Databases architecture                                 #
    ##########################################################################################
    ##########################################################################################

    def create_fields(self,
                      table_name: str,
                      fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Create new Fields in a Table from one of the Databases.

        :param table_name: Name of the Table.
        :param fields: Field or list of Fields names and types.
        """

        # Create the Field(s) in the exchange Database
        if table_name == 'Exchange':
            self.__exchange_db.load()
            if len(self.__exchange_db.get_fields(table_name=table_name)) == 1:
                self.__exchange_db.create_fields(table_name=table_name,
                                                 fields=fields)

        # Create the Field(s) in the storing Database
        else:
            if len(self.__storing_partitions) == 1:
                self.__storing_partitions[0].load()
            if len(self.__storing_partitions[0].get_fields(table_name=table_name)) <= 2:
                self.__storing_partitions[0].create_fields(table_name=table_name,
                                                           fields=fields)

    def get_fields(self,
                   table_name: str) -> List[str]:
        """
        Get the list of Fields in a Table.

        :param table_name: Name of the Table.
        """

        if self.__exchange_db is None and len(self.__storing_partitions) == 0:
            return []
        database = self.__exchange_db if table_name == 'Exchange' else self.__storing_partitions[0]
        return database.get_fields(table_name=table_name)

    ##########################################################################################
    ##########################################################################################
    #                                    Databases editing                                   #
    ##########################################################################################
    ##########################################################################################

    def add_data(self,
                 table_name: str,
                 data: Dict[str, Any]) -> Union[int, List[int]]:
        """
        Add a new line of data in a Database.

        :param table_name: Name of the Table.
        :param data: New line of the Table.
        """

        # Add data in the exchange Database
        if table_name == 'Exchange':
            return self.__exchange_db.add_data(table_name=table_name, data=data)

        # Add data in the storing Database
        else:
            return [len(self.__storing_partitions) - 1,
                    self.__storing_partitions[-1].add_data(table_name=table_name, data=data)]

    def add_batch(self,
                  table_name: str,
                  batch: Dict[str, List[Any]]) -> None:
        """
        Add a batch of data in a Database.

        :param table_name: Name of the Table.
        :param batch: New lines of the Table.
        """

        # Only available in the storing Database
        if table_name == 'Exchange':
            raise ValueError(f"Cannot add a batch in the Exchange Database.")
        self.__storing_partitions[-1].add_batch(table_name=table_name,
                                                batch=batch)

    def update(self,
               table_name: str,
               data: Dict[str, Any],
               line_id: Union[int, List[int]]) -> None:
        """
        Update a line in a Database.

        :param table_name: Name of the Table.
        :param data: Updated line of the Table.
        :param line_id: Index of the line to edit.
        """

        database = self.__exchange_db if table_name == 'Exchange' else self.__storing_partitions[line_id[0]]
        line_id = line_id[1] if type(line_id) == list else line_id
        database.update(table_name=table_name, data=data, line_id=line_id)

    def get_line(self,
                 table_name: str,
                 line_id: Union[int, List[int]],
                 fields: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Get a line of data from a Database.

        :param table_name: Name of the Table.
        :param line_id: Index of the line to get.
        :param fields: Data fields to extract.
        """

        database = self.__exchange_db if table_name == 'Exchange' else self.__storing_partitions[line_id[0]]
        line_id = line_id[1] if type(line_id) == list else line_id
        return database.get_line(table_name=table_name,
                                 line_id=line_id,
                                 fields=fields)

    def get_lines(self,
                  table_name: str,
                  lines_id: List[List[int]],
                  fields: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Get lines of data from a Database.

        :param table_name: Name of the Table.
        :param lines_id: Indices of the lines to get.
        :param fields: Data fields to extract.
        """

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
