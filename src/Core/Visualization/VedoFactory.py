from typing import Optional, Tuple, List, Dict

from SSD.Core.Rendering.VedoFactory import VedoFactory as _VedoFactory
from SSD.Core.Rendering.VedoFactory import Database, VedoTable


class VedoFactory(_VedoFactory):

    def __init__(self,
                 database: Optional[Database] = None,
                 database_path: Optional[Tuple[str, str]] = None,
                 database_dir: str = '',
                 database_name: Optional[str] = None,
                 remove_existing: bool = False,
                 idx_instance: int = 0,
                 remote: bool = False):
        """
        A Factory to manage objects to render and save in the Database.
        User interface to create and update Vedo objects.

        :param database: Database to connect to.
        :param database_path: Path to the Database to connect to.
        :param database_dir: Directory which contains the Database file (used if 'database' is not defined).
        :param database_name: Name of the Database to connect to (used if 'database' is not defined).
        :param remove_existing: If True, overwrite a Database with the same path.
        :param idx_instance: If several Factories must be created, specify the index of the Factory.
        :param remote: If True, the Visualizer will treat the Factories as remote.
        """

        # Define Database
        if database is not None:
            self.__database: Database = database
        elif database_path is not None:
            self.__database: Database = Database(database_dir=database_path[0],
                                                 database_name=database_path[1]).load()
        elif database_name is not None:
            self.__database: Database = Database(database_dir=database_dir,
                                                 database_name=database_name).new(remove_existing=remove_existing)
        else:
            raise ValueError("Both 'database' and 'database_name' are not defined.")

        # Information about all Tables
        self.__tables: List[VedoTable] = []
        self.__current_id: int = 0
        self.__idx: int = idx_instance
        self.__update: Dict[int, bool] = {}
        self.__path = database_path

        # ExchangeTable to synchronize Factory and Visualizer
        if not remote:
            self.__database.register_pre_save_signal(table_name='Sync',
                                                     handler=self.__sync_visualizer,
                                                     name=f'Factory_{self.__idx}')
        self.remote = remote

    def render(self):
        """
        Render the current state of Actors in the Plotter.
        """

        if not self.remote:
            _VedoFactory.render(self)

        else:
            # Reset al the update flags
            for i, updated in self.__update.items():
                self.__update[i] = False
