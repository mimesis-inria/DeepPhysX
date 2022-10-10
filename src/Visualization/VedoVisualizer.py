from typing import Optional, Dict, Tuple, Union

from SSD.Core.Rendering.VedoVisualizer import VedoVisualizer as _VedoVisualizer
from SSD.Core.Rendering.VedoVisualizer import Database, VedoActor, Plotter


class VedoVisualizer(_VedoVisualizer):

    def __init__(self,
                 database: Optional[Database] = None,
                 database_dir: str = '',
                 database_name: Optional[str] = None,
                 remove_existing: bool = False,
                 offscreen: bool = False,
                 remote: bool = False):
        """
        Manage the creation, update and rendering of Vedo Actors.

        :param database: Database to connect to.
        :param database_dir: Directory which contains the Database file (used if 'database' is not defined).
        :param database_name: Name of the Database (used if 'database' is not defined).
        :param remove_existing: If True, overwrite a Database with the same path.
        :param offscreen: If True, visual data will be saved but not rendered.
        :param remote: If True, the Visualizer will treat the Factories as remote.
        """

        # Define Database
        if database is not None:
            self.__database: Database = database
        elif database_name is not None:
            self.__database: Database = Database(database_dir=database_dir,
                                                 database_name=database_name).new(remove_existing=remove_existing)
        else:
            raise ValueError("Both 'database' and 'database_name' are not defined.")

        # Information about all Factories / Actors
        self.__actors: Dict[int, Dict[Tuple[int, int], VedoActor]] = {}
        self.__all_actors: Dict[Tuple[int, int], VedoActor] = {}
        self.__plotter: Optional[Plotter] = None
        self.__offscreen: bool = offscreen
        self.step: Union[int, Dict[int, int]] = {} if remote else 0

        self.__database.create_table(table_name='Sync',
                                     storing_table=False,
                                     fields=('step', str))

        if not remote:
            self.__database.register_post_save_signal(table_name='Sync',
                                                      handler=self.__sync_visualizer)
        self.remote = remote

    def render_instance(self, instance: int):
        """
        Render the current state of Actors managed by a certain Factory in the Plotter.

        :param instance: Index of the Environment render to update.
        """

        if not self.remote:
            self.render()

        else:
            # 1. Update Factories steps
            if instance not in self.step:
                self.step[instance] = 0
            self.step[instance] += 1

            # 2. Retrieve visual data and update Actors (one Table per Actor)
            table_names = self.__database.get_tables()
            table_names.remove('Sync')
            table_names = [table_name for table_name in table_names if table_name.split('_')[1] == str(instance)]
            for table_name in table_names:
                # Get the current step line in the Table
                data_dict = self.__database.get_line(table_name=table_name,
                                                     line_id=self.step[instance])
                # If the id of line is correct, the Actor was updated
                if data_dict.pop('id') == self.step[instance]:
                    self.update_instance(table_name=table_name,
                                         data_dict=data_dict)
                # Otherwise, the actor was not updated, then add an empty line
                else:
                    self.__database.add_data(table_name=table_name,
                                             data={})

            # 3. Render Plotter if offscreen is False
            if not self.__offscreen:
                self.__plotter.render()
