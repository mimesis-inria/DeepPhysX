from typing import Union, List, Dict, Any, Callable, Optional, Type, Tuple
from numpy import array, where
from itertools import chain

from SSD.Core.Storage.database import Database


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
        self.__db: Optional[Database] = None
        self.__current_table: str = 'train'
        self.__exchange_db: Optional[Database] = None

        # Event handlers
        # self.__on_init_handler = self.default_handler if on_init_handler is None else on_init_handler
        # self.__on_partitions_handler = self.default_handler if on_partitions_handler is None else on_partitions_handler

    # def default_handler(self):
    #     pass
    #
    # def set_init_handler(self, on_init_handler: Callable):
    #     self.__on_init_handler = on_init_handler
    #
    # def set_partitions_handler(self, on_partitions_handler: Callable):
    #     self.__on_partitions_handler = on_partitions_handler

    ##########################################################################################
    ##########################################################################################
    #                                 Partitions management                                  #
    ##########################################################################################
    ##########################################################################################

    def init(self,
             database: Tuple[str, str],
             exchange_db: Tuple[str, str]) -> None:
        """
        Initialize the list of the partitions.

        :param database: Storing Database.
        :param exchange_db: Exchange Database.
        """

        self.__db = Database(database_dir=database[0],
                             database_name=database[1]).load()
        self.__exchange_db = Database(database_dir=exchange_db[0],
                                      database_name=exchange_db[1]).load()
        # self.__on_init_handler()

    def init_remote(self,
                    database: List[str],
                    exchange_db: List[str]) -> None:
        """
        Initialize the list of partitions in remote DatabaseHandlers.

        :param database: List of paths to the storing Database partitions.
        :param exchange_db: Path to the exchange Database.
        """

        self.__db = Database(database_dir=database[0],
                             database_name=database[1]).load()
        self.__exchange_db = Database(database_dir=exchange_db[0],
                                      database_name=exchange_db[1]).load()
        # self.__on_init_handler()

    # def update_list_partitions(self,
    #                            partition: Database) -> None:
    #     """
    #     Add a new storing partition to the list.
    #
    #     :param partition: New storing partition to add.
    #     """
    #
    #     self.__storing_partitions.append(partition)
    #     self.__on_partitions_handler()

    # def update_list_partitions_remote(self,
    #                                   partition: List[str]) -> None:
    #     """
    #     Add a new storing partition to the list in remote DatabaseHandler.
    #
    #     :param partition: Path to the new storing partition.
    #     """
    #
    #     self.__storing_partitions.append(Database(database_dir=partition[0],
    #                                               database_name=partition[1]).load())
    #     self.__on_partitions_handler()

    def load(self) -> None:
        """
        Load the Database partitions stored by the component.
        """

        self.__db.load()
        self.__exchange_db.load()

    def get_database_dir(self) -> str:
        """
        Get the database repository of the session.
        """

        return self.__db.get_path()[0]

    def get_database(self) -> Database:
        """
        Get the storing Database partitions.
        """

        return self.__db

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

    def create_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        """
        Create new Fields in a Table from one of the Databases.

        :param fields: Field or list of Fields names and types.
        """

        # Create the Field(s) in the exchange Database
        self.__exchange_db.load()
        self.__exchange_db.create_fields(table_name='data',
                                         fields=fields)

        # Create the Field(s) in the storing Database
        self.__db.load()
        for mode in ['train', 'test', 'run']:
            self.__db.create_fields(table_name=mode,
                                    fields=fields)

    def get_fields(self) -> List[str]:
        """
        Get the list of Fields in a Table.
        """

        return self.__db.get_fields(table_name=self.__current_table)

    ##########################################################################################
    ##########################################################################################
    #                                    Databases editing                                   #
    ##########################################################################################
    ##########################################################################################

    def add_data(self,
                 data: Dict[str, Any]) -> Union[int, List[int]]:
        """
        Add a new line of data in a Database.

        :param data: New line of the Table.
        """

        # Add data in the exchange Database
        # if table_name == 'Exchange':
        #     return self.__exchange_db.add_data(table_name=table_name, data=data)

        # Add data in the storing Database
        return self.__db.add_data(table_name=self.__current_table, data=data)

    def add_batch(self,
                  batch: Dict[str, List[Any]]) -> None:
        """
        Add a batch of data in a Database.

        :param batch: New lines of the Table.
        """

        # Only available in the storing Database
        # if table_name == 'Exchange':
        #     raise ValueError(f"Cannot add a batch in the Exchange Database.")
        self.__db.add_batch(table_name='train', batch=batch)

    def update(self,
               data: Dict[str, Any],
               line_id: Union[int, List[int]]) -> None:
        """
        Update a line in a Database.

        :param data: Updated line of the Table.
        :param line_id: Index of the line to edit.
        """

        # database = self.__exchange_db if table_name == 'Exchange' else self.__storing_partitions[line_id[0]]
        line_id = line_id[1] if type(line_id) == list else line_id
        self.__db.update(table_name='train', data=data, line_id=line_id)

    def get_line(self,
                 line_id: Union[int, List[int]],
                 fields: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Get a line of data from a Database.

        :param line_id: Index of the line to get.
        :param fields: Data fields to extract.
        """

        # database = self.__exchange_db if table_name == 'Exchange' else self.__storing_partitions[line_id[0]]
        line_id = line_id[1] if type(line_id) == list else line_id
        if self.__db.nb_lines(table_name='train') == 0:
            return {}
        return self.__db.get_line(table_name='train',
                                 line_id=line_id,
                                 fields=fields)

    # def get_lines(self,
    #               lines_id: List[List[int]],
    #               fields: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    #     """
    #     Get lines of data from a Database.
    #
    #     :param lines_id: Indices of the lines to get.
    #     :param fields: Data fields to extract.
    #     """
    #
    #     # Transform list of lines to batch of lines per partition
    #     batch_indices = array(lines_id)
    #     partition_batch_indices = []
    #     for i in range(len(self.__storing_partitions)):
    #         partition_indices = where(batch_indices[:, 0] == i)[0]
    #         if len(partition_indices) > 0:
    #             partition_batch_indices.append([i, batch_indices[partition_indices, 1].tolist()])
    #
    #     # Get lines of data
    #     partition_batches = []
    #     for partition_indices in partition_batch_indices:
    #         data = self.__storing_partitions[partition_indices[0]].get_lines(table_name=table_name,
    #                                                                          lines_id=partition_indices[1],
    #                                                                          fields=fields,
    #                                                                          batched=True)
    #         del data['id']
    #         partition_batches.append(data)
    #
    #     # Merge batches
    #     if len(partition_batches) == 1:
    #         return partition_batches[0]
    #     else:
    #         return dict(zip(partition_batches[0].keys(),
    #                         [list(chain.from_iterable([partition_batches[i][key]
    #                                                    for i in range(len(partition_batches))]))
    #                          for key in partition_batches[0].keys()]))
