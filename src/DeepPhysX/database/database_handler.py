from typing import Union, List, Dict, Any, Optional, Type, Tuple
from numpy import array

from SSD.core import Database


class DatabaseHandler:

    def __init__(self):
        """
        DatabaseHandler allows components to be synchronized with the Database partitions and to read / write data.
        """

        # Databases variables
        self.__db: Optional[Database] = None
        self.__current_table: str = 'train'
        self.__exchange_db: Optional[Database] = None

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

    def create_fields(self,
                      fields: Union[List[Tuple[str, Type]], Tuple[str, Type]],
                      exchange: bool = False) -> None:
        """
        Create new Fields in a Table from one of the Databases.

        :param fields: Field or list of Fields names and types.
        :param exchange: If True, add data to the exchange Table.
        """

        # Create the Field(s) in the exchange Database
        if exchange:
            self.__exchange_db.load()
            self.__exchange_db.create_fields(table_name='data',
                                             fields=fields)

        # Create the Field(s) in the storing Database
        else:
            self.__db.load()
            for mode in ['train', 'test', 'run']:
                self.__db.create_fields(table_name=mode,
                                        fields=fields)

    def get_fields(self, exchange: bool = False) -> List[str]:
        """
        Get the list of Fields in a Table.
        """

        if not exchange:
            return self.__db.get_fields(table_name=self.__current_table)
        return self.__exchange_db.get_fields(table_name='data')

    ##########################################################################################
    ##########################################################################################
    #                                    Databases editing                                   #
    ##########################################################################################
    ##########################################################################################

    def add_data(self,
                 data: Dict[str, Any],
                 exchange: bool = False) -> Union[int, List[int]]:
        """
        Add a new line of data in a Database.

        :param data: New line of the Table.
        :param exchange: If True, add data to the exchange Table.
        """

        # Add data in the exchange Database
        if exchange:
            return self.__exchange_db.add_data(table_name='data', data=data)

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
               line_id: Union[int, List[int]],
               exchange: bool = False) -> None:
        """
        Update a line in a Database.

        :param data: Updated line of the Table.
        :param line_id: Index of the line to edit.
        """

        # database = self.__exchange_db if table_name == 'Exchange' else self.__storing_partitions[line_id[0]]
        line_id = line_id[1] if type(line_id) == list else line_id
        if not exchange:
            self.__db.update(table_name='train', data=data, line_id=line_id)
        else:
            self.__exchange_db.update(table_name='data', data=data, line_id=line_id)

    def get_line(self,
                 line_id: Union[int, List[int]],
                 fields: Optional[Union[str, List[str]]] = None,
                 exchange: bool = False) -> Dict[str, Any]:
        """
        Get a line of data from a Database.

        :param line_id: Index of the line to get.
        :param fields: Data fields to extract.
        """

        line_id = line_id[1] if type(line_id) == list else line_id
        if not exchange:
            if self.__db.nb_lines(table_name='train') == 0:
                return {}
            return self.__db.get_line(table_name='train',
                                     line_id=line_id,
                                     fields=fields)
        if self.__exchange_db.nb_lines(table_name='data') == 0:
            return {}
        return self.__exchange_db.get_line(table_name='data',
                                  line_id=line_id,
                                  fields=fields)

    def get_lines(self,
                  lines_id: List[int],
                  fields: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Get lines of data from a Database.

        :param lines_id: Indices of the lines to get.
        :param fields: Data fields to extract.
        """

        # Get lines of data
        data = self.__db.get_lines(table_name='train',
                                   lines_id=lines_id,
                                   fields=fields,
                                   batched=True)
        del data['id']
        return data
