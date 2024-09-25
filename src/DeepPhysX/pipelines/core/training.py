from typing import Optional
from os.path import join, isfile, exists, sep
from datetime import datetime
from vedo import ProgressBar

from DeepPhysX.pipelines.core.base_pipeline import BasePipeline
from DeepPhysX.pipelines.core.data_manager import DataManager
from DeepPhysX.networks.core.network_manager import NetworkManager
from DeepPhysX.networks.core.stats_manager import StatsManager
from DeepPhysX.networks.core.dpx_network_config import BaseNetworkConfig
from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.simulation.core.simulation_config import SimulationConfig
from DeepPhysX.utils.path import create_dir


class BaseTraining(BasePipeline):

    def __init__(self,
                 network_config: BaseNetworkConfig,
                 database_config: DatabaseConfig,
                 environment_config: Optional[SimulationConfig] = None,
                 session_dir: str = 'sessions',
                 session_name: str = 'training',
                 new_session: bool = True,
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
        produce_data = database_config.existing_dir is None

        # Create a DataManager
        self.data_manager = DataManager(pipeline=self,
                                        database_config=database_config,
                                        environment_config=environment_config,
                                        session=join(self.session_dir, self.session_name),
                                        new_session=new_session,
                                        produce_data=produce_data,
                                        batch_size=batch_size)
        self.batch_size = batch_size

        # Create a NetworkManager
        self.network_manager = NetworkManager(network_config=network_config,
                                              pipeline=self.type,
                                              session=join(self.session_dir, self.session_name),
                                              new_session=new_session)
        self.data_manager.connect_handler(self.network_manager.get_database_handler())
        self.network_manager.link_clients(self.data_manager.nb_environment)

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

        self.data_manager.get_data(epoch=self.epoch_id,
                                   animate=True)
        self.loss_dict = self.network_manager.compute_prediction_and_loss(
            data_lines=self.data_manager.data_lines,
            normalization=self.data_manager.normalization,
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

        self.data_manager.close()
        self.network_manager.close()
        if self.stats_manager is not None:
            self.stats_manager.close()

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
            f.write(str(self.data_manager))
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
