from typing import Optional
from os.path import join, isfile, exists, sep
from datetime import datetime
from vedo import ProgressBar

from DeepPhysX.pipelines.core.base_pipeline import BasePipeline
from DeepPhysX.database.database_manager import DatabaseManager, DatabaseConfig
from DeepPhysX.networks.core.network_manager import NetworkManager
from DeepPhysX.networks.core.stats_manager import StatsManager
from DeepPhysX.networks.core.network_config import NetworkConfig
from DeepPhysX.simulation.core.simulation_manager import SimulationManager, SimulationConfig
from DeepPhysX.utils.path import create_dir


class Training(BasePipeline):

    def __init__(self,
                 network_config: NetworkConfig,
                 database_config: DatabaseConfig,
                 environment_config: Optional[SimulationConfig] = None,
                 new_session: bool = True,
                 session_dir: str = 'sessions',
                 session_name: str = 'training',
                 epoch_nb: int = 0,
                 batch_nb: int = 0,
                 batch_size: int = 0,
                 debug: bool = False):
        """
        BaseTraining implements the main loop that defines the training process of an artificial neural networks.
        Training can be launched with several data sources (from a Dataset, from an Environment, from combined sources).
        It provides a highly tunable learning process that can be used with any machine learning library.

        :param network_config: Configuration object with the parameters of the networks.
        :param database_config: Configuration object with the parameters of the Database.
        :param environment_config: Configuration object with the parameters of the Environment.
        :param session_dir: Relative path to the directory which contains sessions repositories.
        :param session_name: Name of the new the session repository.
        :param new_session: If True, a new repository will be created for this session.
        :param epoch_nb: Number of epochs to perform.
        :param batch_nb: Number of batches to use.
        :param batch_size: Number of samples in a single batch.
        :param debug: If True, main training features will not be launched.
        """

        BasePipeline.__init__(self,
                              network_config=network_config,
                              database_config=database_config,
                              environment_config=environment_config,
                              session_dir=session_dir,
                              session_name=session_name,
                              new_session=new_session,
                              pipeline='training')

        # Create a new session if required
        if not self.new_session:
            self.new_session = not exists(join(self.session_dir, self.session_name))
        if self.new_session:
            self.session_name = create_dir(session_dir=self.session_dir,
                                           session_name=self.session_name).split(sep)[-1]

        # Configure 'produce_data' flag
        if environment_config is None and database_config.existing_dir is None:
            raise ValueError(f"[{self.name}] No data source provided.")
        self.produce_data = database_config.existing_dir is None

        # Create a DataManager
        # self.data_manager = DataManager(pipeline=self,
        #                                 database_config=database_config,
        #                                 environment_config=environment_config,
        #                                 session=join(self.session_dir, self.session_name),
        #                                 new_session=new_session,
        #                                 produce_data=produce_data,
        #                                 batch_size=batch_size)
        self.database_manager = DatabaseManager(database_config=database_config,
                                                pipeline=self.type,
                                                session=join(self.session_dir, self.session_name),
                                                new_session=new_session,
                                                produce_data=self.produce_data)

        if environment_config is not None:
            self.simulation_manager = SimulationManager(config=environment_config,
                                                        pipeline=self.type,
                                                        session=join(self.session_dir, self.session_name),
                                                        produce_data=self.produce_data,
                                                        batch_size=batch_size)
            self.simulation_manager.connect_to_database(**self.database_manager.get_database_paths())

        # Create a NetworkManager
        self.network_manager = NetworkManager(network_config=network_config,
                                              pipeline=self.type,
                                              session=join(self.session_dir, self.session_name),
                                              new_session=new_session)
        self.network_manager.connect_to_database(**self.database_manager.get_database_paths())

        if self.simulation_manager is not None:
            self.network_manager.link_clients(1 if self.simulation_manager.server is None
                                              else self.simulation_manager.nb_parallel_env)

        # Create a StatsManager
        self.stats_manager = StatsManager(session=join(self.session_dir, self.session_name)) if not debug else None

        # Training variables
        self.epoch_nb = epoch_nb
        self.epoch_id = 0
        self.batch_nb = batch_nb
        self.batch_size = batch_size
        self.batch_id = 0
        self.nb_samples = batch_nb * batch_size * epoch_nb
        self.loss_dict = None
        self.debug = debug

        # Progressbar
        self.digits = ['{' + f':0{len(str(self.epoch_nb))}d' + '}',
                       '{' + f':0{len(str(self.batch_nb))}d' + '}']
        epoch_id, epoch_nb = self.digits[0].format(0), self.digits[0].format(self.epoch_nb)
        batch_id, batch_nb = self.digits[1].format(0), self.digits[1].format(self.batch_nb)
        self.progress_bar = ProgressBar(start=0, stop=self.batch_nb * self.epoch_nb, c='orange',
                                        title=f'Epoch n°{epoch_id}/{epoch_nb} - Batch n°{batch_id}/{batch_nb}')

        self.save_info_file()

    def execute(self) -> None:
        """
        Launch the training Pipeline.
        Each event is already implemented for a basic pipeline but can also be rewritten via inheritance to describe a
        more complex Pipeline.
        """

        self.train_begin()
        while self.epoch_condition():
            self.epoch_begin()
            while self.batch_condition():
                self.batch_begin()
                self.optimize()
                self.batch_count()
                self.batch_end()
            self.epoch_count()
            self.epoch_end()
        self.train_end()

    def train_begin(self) -> None:
        """
        Called once at the beginning of the training Pipeline.
        """

        pass

    def epoch_condition(self) -> bool:
        """
        Check the epoch number condition.
        """

        return self.epoch_id < self.epoch_nb

    def epoch_begin(self) -> None:
        """
        Called one at the beginning of each epoch.
        """

        self.batch_id = 0

    def batch_condition(self) -> bool:
        """
        Check the batch number condition.
        """

        return self.batch_id < self.batch_nb

    def batch_begin(self) -> None:
        """
        Called one at the beginning of a batch production.
        """

        id_epoch, nb_epoch = self.digits[0].format(self.epoch_id + 1), self.digits[0].format(self.epoch_nb)
        id_batch, nb_batch = self.digits[1].format(self.batch_id + 1), self.digits[1].format(self.batch_nb)
        self.progress_bar.title = f'Epoch n°{id_epoch}/{nb_epoch} - Batch n°{id_batch}/{nb_batch} '
        self.progress_bar.print()

    def optimize(self) -> None:
        """
        Pulls data, run a prediction and an optimizer step.
        """

        # Get data from Environment(s) if used and if the data should be created at this epoch
        if self.simulation_manager is not None and self.produce_data and \
                (self.epoch_id == 0 or self.simulation_manager.always_produce):
            self.data_lines = self.simulation_manager.get_data(animate=True)
            self.database_manager.add_data(self.data_lines)

        # Get data from Dataset
        else:
            self.data_lines = self.database_manager.get_data(batch_size=self.batch_size)
            # Dispatch a batch to clients
            if self.simulation_manager is not None:
                if self.simulation_manager.load_samples and \
                        (self.epoch_id == 0 or not self.simulation_manager.only_first_epoch):
                    self.simulation_manager.dispatch_batch(data_lines=self.data_lines,
                                                           animate=True)
                # # Environment is no longer used
                # else:
                #     self.simulation_manager.close()
                #     self.simulation_manager = None

        self.loss_dict = self.network_manager.compute_prediction_and_loss(
            data_lines=self.data_lines,
            normalization=self.database_manager.normalization,
            optimize=True)

    def batch_count(self) -> None:
        """
        Increment the batch counter.
        """

        self.batch_id += 1

    def batch_end(self) -> None:
        """
        Called one at the end of a batch production.
        """

        if self.stats_manager is not None:
            self.stats_manager.add_train_batch_loss(self.loss_dict['loss'],
                                                    self.epoch_id * self.batch_nb + self.batch_id)
            for key in self.loss_dict.keys():
                if key != 'loss':
                    self.stats_manager.add_custom_scalar(tag=key,
                                                         value=self.loss_dict[key],
                                                         count=self.epoch_id * self.batch_nb + self.batch_id)

    def epoch_count(self) -> None:
        """
        Increment the epoch counter.
        """

        self.epoch_id += 1

    def epoch_end(self) -> None:
        """
        Called one at the end of each epoch.
        """

        if self.stats_manager is not None:
            self.stats_manager.add_train_epoch_loss(self.loss_dict['loss'], self.epoch_id)
        self.network_manager.save_network()

    def train_end(self) -> None:
        """
        Called once at the end of the training Pipeline.
        """

        for manager in (self.database_manager, self.network_manager, self.stats_manager, self.simulation_manager):
            if manager is not None:
                manager.close()

    def save_info_file(self) -> None:
        """
        Save a .txt file that provides a template for user notes and the description of all the components.
        """

        filename = join(join(self.session_dir, self.session_name), 'info.txt')
        date_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        if not isfile(filename):
            f = open(filename, "w+")
            # Session description template for user
            f.write("## DeepPhysX Training Session ##\n")
            f.write(date_time + "\n\n")
            f.write("Personal notes on the training session:\nnetworks Input:\nnetworks Output:\nComments:\n\n")
            # Listing every component descriptions
            f.write("## List of Components Parameters ##\n")
            f.write(str(self))
            f.write(str(self.network_manager))

            if self.simulation_manager is not None:
                f.write(str(self.simulation_manager))
            if self.database_manager is not None:
                f.write(str(self.database_manager))

            if self.stats_manager is not None:
                f.write(str(self.stats_manager))
            f.close()

    def __str__(self):

        description = BasePipeline.__str__(self)
        description += f"    Number of epochs: {self.epoch_nb}\n"
        description += f"    Number of batches per epoch: {self.batch_nb}\n"
        description += f"    Number of samples per batch: {self.batch_size}\n"
        description += f"    Number of samples per epoch: {self.batch_nb * self.batch_size}\n"
        description += f"    Total: Number of batches : {self.batch_nb * self.epoch_nb}\n"
        description += f"           Number of samples : {self.nb_samples}\n"
        return description
