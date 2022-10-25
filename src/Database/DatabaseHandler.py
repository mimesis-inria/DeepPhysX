from typing import Union, List, Dict, Any, Callable, Optional, Type, Tuple
from numpy import ndarray

from SSD.Core.Storage.Database import Database


class DatabaseHandler:

    def __init__(self,
                 remote: bool = False,
                 on_init_handler: Optional[Callable] = None):

        self.remote = remote
        self.__partitions: List[Database] = []
        self.__on_init_handler = self.__default_handler if on_init_handler is None else on_init_handler

    def __default_handler(self):
        pass

    ###
    # Manage partitions
    ###

    def init(self, partitions: List[Database]) -> None:
        self.__partitions = partitions.copy()
        self.__on_init_handler()

    def init_remote(self, partitions: List[List[str]]) -> None:
        self.__partitions = [Database(database_dir=partition[0],
                                      database_name=partition[1]).load() for partition in partitions]
        self.__on_init_handler()

    def update_list_partitions(self,
                               partition: Database) -> None:
        self.__partitions.append(partition)

    def get_database_dir(self):
        return self.__partitions[0].get_path()[0]

    def load(self):
        for db in self.__partitions:
            db.load()

    ###
    # Database Architecture
    ###

    def create_fields(self,
                      table_name: str,
                      fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        if len(self.__partitions) == 1:
            self.__partitions[0].create_fields(table_name=table_name,
                                               fields=fields)

    def define_fields(self,
                      table_name: str,
                      fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None:
        if len(self.__partitions) == 1:
            self.__partitions[0].load()
            if len(self.__partitions[0].get_fields(table_name=table_name)) == 2:
                self.__partitions[0].create_fields(table_name=table_name,
                                                   fields=fields)

    def get_fields(self,
                   table_name: str) -> List[str]:
        return self.__partitions[0].get_fields(table_name=table_name)

    ###
    # Data Read / Write
    ###

    def get_line(self,
                 table_name: str,
                 line_id: List[int]) -> Dict[str, ndarray]:
        return self.__partitions[line_id[0]].get_line(table_name=table_name,
                                                      line_id=line_id[1])

    def add_data(self,
                 table_name: str,
                 data: Dict[str, Any]) -> int:
        return self.__partitions[-1].add_data(table_name=table_name,
                                              data=data)

    def update(self,
               table_name: str,
               data: Dict[str, Any],
               line_id: int) -> None:
        self.__partitions[-1].update(table_name=table_name,
                                     data=data,
                                     line_id=line_id)
