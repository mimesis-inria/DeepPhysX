from typing import Optional
from sys import stdout
from os.path import join, isfile
from datetime import datetime

from DeepPhysX.Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Core.Manager.Manager import Manager
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Database.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Utils.progressbar import Progressbar


class BaseTrainer(BasePipeline):

    def __init__(self,
                 network_config: BaseNetworkConfig,
                 dataset_config: BaseDatasetConfig,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 session_dir: str = 'sessions',
                 session_name: str = 'training',
                 epoch_nb: int = 0,
                 batch_nb: int = 0,
                 batch_size: int = 0,
                 new_session: bool = True,
                 debug_session: bool = False):
        """
        BaseTrainer implements the main loop that defines the training process of an artificial neural network.
        Training can be launched with several data sources (from a Dataset, from an Environment, from combined sources).
        It provides a highly tunable learning process that can be used with any machine learning library.

        :param network_config: Specialisation containing the parameters of the network manager.
        :param dataset_config: Specialisation containing the parameters of the dataset manager.
        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param session_dir: Relative path to the directory which contains sessions directories.
        :param session_name: Name of the new the session directory.
        :param epoch_nb: Number of epochs to run.
        :param batch_nb: Number of batches to use.
        :param batch_size: Number of samples in a single batch.
        :param new_session: Define the creation of new directories to store data.
        :param debug_session: If True, main training features will not be launched.
        """

        BasePipeline.__init__(self,
                              network_config=network_config,
                              dataset_config=dataset_config,
                              environment_config=environment_config,
                              session_dir=session_dir,
                              session_name=session_name,
                              pipeline='training')

        # Training variables
        self.epoch_nb = epoch_nb
        self.epoch_id = 0
        self.batch_nb = batch_nb
        self.batch_size = batch_size
        self.batch_id = 0
        self.nb_samples = batch_nb * batch_size * epoch_nb
        self.loss_dict = None
        self.debug = debug_session

        # Configure 'produce_data' flag
        if environment_config is None and dataset_config.existing_dir is None:
            raise ValueError(f"[{self.name}] No data source provided.")
        produce_data = dataset_config.existing_dir is None

        # Progressbar
        if not self.debug:
            self.progress_counter = 0
            self.digits = ['{' + f':0{len(str(self.epoch_nb))}d' + '}',
                           '{' + f':0{len(str(self.batch_nb))}d' + '}']
            epoch_id, epoch_nb = self.digits[0].format(0), self.digits[0].format(self.epoch_nb)
            batch_id, batch_nb = self.digits[1].format(0), self.digits[1].format(self.batch_nb)
            self.progress_bar = Progressbar(start=0, stop=self.batch_nb * self.epoch_nb, c='orange',
                                            title=f'Epoch n°{epoch_id}/{epoch_nb} - Batch n°{batch_id}/{batch_nb}')

        self.manager = Manager(network_config=self.network_config,
                               dataset_config=self.dataset_config,
                               environment_config=self.environment_config,
                               session_dir=session_dir,
                               session_name=session_name,
                               new_session=new_session,
                               pipeline='training',
                               produce_data=produce_data,
                               batch_size=batch_size,
                               debug_session=debug_session)
        self.save_info_file(self.manager.session)

    def execute(self) -> None:
        """
        Launch the training Pipeline.
        Each event is already implemented for a basic pipeline but can also be rewritten via inheritance to describe a
        more complex pipeline.
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
            self.save_network()
        self.train_end()

    def train_begin(self) -> None:
        """
        Called once at the beginning of the training pipeline.
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

        if not self.debug:
            stdout.write("\033[K")
            self.progress_counter += 1
            id_epoch, nb_epoch = self.digits[0].format(self.epoch_id + 1), self.digits[0].format(self.epoch_nb)
            id_batch, nb_batch = self.digits[1].format(self.batch_id + 1), self.digits[1].format(self.batch_nb)
            self.progress_bar.title = f'Epoch n°{id_epoch}/{nb_epoch} - Batch n°{id_batch}/{nb_batch} '
            self.progress_bar.print(counts=self.progress_counter)

    def optimize(self) -> None:
        """
        Pulls data, run a prediction and an optimizer step.
        """

        self.manager.get_data(self.epoch_id)
        _, self.loss_dict = self.manager.optimize_network()

    def batch_count(self) -> None:
        """
        Increment the batch counter.
        """

        self.batch_id += 1

    def batch_end(self) -> None:
        """
        Called one at the end of a batch production.
        """

        self.manager.stats_manager.add_train_batch_loss(self.loss_dict['loss'],
                                                        self.epoch_id * self.batch_nb + self.batch_id)
        for key in self.loss_dict.keys():
            if key != 'loss':
                self.manager.stats_manager.add_custom_scalar(tag=key,
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

        self.manager.stats_manager.add_train_epoch_loss(self.loss_dict['loss'], self.epoch_id)

    def save_network(self) -> None:
        """
        Store the network parameters in the corresponding directory.
        """

        self.manager.save_network()

    def train_end(self) -> None:
        """
        Called once at the end of the training pipeline.
        """

        self.manager.close()

    def save_info_file(self,
                       directory: str) -> None:
        """
        Save a .txt file that provides a template for user notes and the description of all the components.
        """

        filename = join(directory, 'info.txt')
        date_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        if not isfile(filename):
            f = open(filename, "w+")
            # Session description template for user
            f.write("## DeepPhysX Training Session ##\n")
            f.write(date_time + "\n\n")
            f.write("Personal notes on the training session:\nNetwork Input:\nNetwork Output:\nComments:\n\n")
            # Listing every component descriptions
            f.write("## List of Components Parameters ##\n")
            f.write(str(self))
            f.write(str(self.manager))
            f.close()

    def __str__(self) -> str:

        description = "\n"
        description += f"# {self.__class__.__name__}\n"
        description += f"    Session directory: {self.manager.session}\n"
        description += f"    Number of epochs: {self.epoch_nb}\n"
        description += f"    Number of batches per epoch: {self.batch_nb}\n"
        description += f"    Number of samples per batch: {self.batch_size}\n"
        description += f"    Number of samples per epoch: {self.batch_nb * self.batch_size}\n"
        description += f"    Total: Number of batches : {self.batch_nb * self.epoch_nb}\n"
        description += f"           Number of samples : {self.nb_samples}\n"
        return description
