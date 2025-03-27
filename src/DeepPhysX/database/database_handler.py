from typing import Union, List, Dict, Any, Optional, Type, Tuple
from os.path import join
import json

from SSD.core import Database


class DatabaseHandler:

    def __init__(self, ):
        """
        DatabaseHandler allows components to be synchronized with the Database partitions and to  have a read & write
        access to data.
        """

        # Databases variables
        self.__db: Optional[Database] = None
        self.__current_table: str = 'train'
        self.__exchange_db: Optional[Database] = None

        self.do_normalize = False
        self.__normalize: bool = False
        self.__json_file: str = ''

    #######################
    # Database management #
    #######################

    def init(self,
             database_path: Tuple[str, str],
             normalize_data: bool) -> None:
        """
        Initialize the Database access.

        :param database_path: Storing Database path.
        :param normalize_data: If True, data will be normalized.
        """

        # Load the Database that was created in the DatabaseManager
        self.__db = Database(database_dir=database_path[0], database_name=database_path[1]).load()
        self.__exchange_db = Database(database_dir=database_path[0], database_name='temp').load()

        # Load the json file that contains data fields information
        self.__json_file = join(database_path[0], 'dataset.json')
        with open(self.__json_file) as json_file:
            fields = json.load(json_file)['fields']

        # Load the normalization coefficients
        if normalize_data:
            self.do_normalize = True
            self.__normalize = {field: fields[field]['normalize'] for field in fields}
        else:
            self.__normalize = {field: [0, 1] for field in fields}

    def reload_normalization(self) -> None:
        """
        Load the normalization coefficients from the database json file.
        """

        with open(self.__json_file) as json_file:
            fields = json.load(json_file)['fields']
            self.__normalize = {field: fields[field]['normalize'] for field in fields}

    def load(self) -> None:
        """
        Load the Database.
        """

        self.__db.load()
        self.__exchange_db.load()

    ###########################
    # Databases architectures #
    ###########################

    def create_fields(self,
                      fields: List[Tuple[str, Type]],
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

    def get_fields(self, exchange: bool = False, only_names: bool = True) -> List[str]:
        """
        Get the list of Fields in a Table.
        """

        if not exchange:
            return self.__db.get_fields(table_name=self.__current_table, only_names=only_names)
        return self.__exchange_db.get_fields(table_name='data', only_names=only_names)

    @property
    def normalization(self):

        return self.__normalize

    #####################
    # Databases editing #
    #####################

    def add_data(self,
                 data: Dict[str, Any],
                 exchange: bool = False) -> Union[int, List[int]]:
        """
        Add a new line of data in a Database.

        :param data: New line of the Table.
        :param exchange: If True, add data to the exchange Table.
        """

        if exchange:
            return self.__exchange_db.add_data(table_name='data', data=data)
        return self.__db.add_data(table_name=self.__current_table, data=data)

    def add_batch(self, batch: Dict[str, List[Any]]) -> None:
        """
        Add a batch of data in a Database.

        :param batch: New lines of the Table.
        """

        self.__db.add_batch(table_name=self.__current_table, batch=batch)

    def update(self,
               data: Dict[str, Any],
               line_id: Union[int, List[int]],
               exchange: bool = False) -> None:
        """
        Update a line in a Database.

        :param data: Updated line of the Table.
        :param line_id: Index of the line to edit.
        :param exchange: If True, add data to the exchange Table.
        """

        line_id = line_id[1] if type(line_id) == list else line_id
        if not exchange:
            self.__db.update(table_name=self.__current_table, data=data, line_id=line_id)
        else:
            self.__exchange_db.update(table_name='data', data=data, line_id=line_id)

    def get_data(self,
                 line_id: Union[int, List[int]],
                 fields: Optional[Union[str, List[str]]] = None,
                 exchange: bool = False) -> Dict[str, Any]:
        """
        Get a line of data from a Database.

        :param line_id: Index of the line to get.
        :param fields: Data fields to extract.
        :param exchange: If True, add data to the exchange Table.
        """

        line_id = line_id[1] if type(line_id) == list else line_id
        if not exchange:
            if self.__db.nb_lines(table_name=self.__current_table) == 0:
                return {}
            return self.__db.get_line(table_name=self.__current_table, line_id=line_id, fields=fields)
        if self.__exchange_db.nb_lines(table_name='data') == 0:
            return {}
        return self.__exchange_db.get_line(table_name='data', line_id=line_id, fields=fields)

    def get_batch(self,
                  lines_id: List[int],
                  fields: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Get lines of data from a Database.

        :param lines_id: Indices of the lines to get.
        :param fields: Data fields to extract.
        """

        data = self.__db.get_lines(table_name=self.__current_table, lines_id=lines_id, fields=fields, batched=True)
        del data['id']
        return data
