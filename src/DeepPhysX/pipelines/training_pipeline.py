from typing import Optional, Type, Dict, Any, Callable
from os.path import join, isfile, exists, sep
from datetime import datetime
from vedo import ProgressBar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.networks.network_manager import NetworkManager
from DeepPhysX.networks.stats_manager import StatsManager
from DeepPhysX.simulation.simulation_manager import SimulationManager
from DeepPhysX.utils.path import create_dir, get_session_dir


class TrainingPipeline:

    def __init__(self,
                 network_manager: NetworkManager,
                 database_manager: DatabaseManager,
                 loss_fnc: Type[_Loss],
                 optimizer: Type[Optimizer],
                 optimizer_kwargs: Dict[str, Any],
                 simulation_manager: Optional[SimulationManager] = None,
                 new_session: bool = True,
                 session_dir: str = 'sessions',
                 session_name: str = 'training',
                 epoch_nb: int = 0,
                 batch_nb: int = 0,
                 batch_size: int = 0,
                 debug: bool = False,
                 save_intermediate_state_every: int = 0):
        """
        """

        # Create a new session if required
        self.session_dir = get_session_dir(session_dir, new_session)
        if not new_session:
            new_session = not exists(join(self.session_dir, session_name))
        if new_session:
            session_name = create_dir(session_dir=self.session_dir,
                                      session_name=session_name).split(sep)[-1]

        # Configure 'produce_data' flag
        # if simulation_config is None and database_config.existing_dir is None:
        #     raise ValueError(f"[{self.__class__.__name__}] No data source provided.")
        self.produce_data = database_manager.existing_dir is None

        # Create a DatabaseManager
        self.database_manager = database_manager
        # self.database_manager = DatabaseManager(config=database_config,
        #                                         session=join(self.session_dir, session_name))
        self.database_manager.init_training_pipeline(session=join(self.session_dir, session_name),
                                                     new_session=new_session,
                                                     produce_data=self.produce_data)

        # Create a SimulationManager
        self.simulation_manager = None
        if simulation_manager is not None:
            self.simulation_manager = simulation_manager
            self.simulation_manager.init_training_pipeline(batch_size=batch_size)
            # self.simulation_manager = SimulationManager(config=simulation_config,
            #                                             pipeline='training',
            #                                             session=join(self.session_dir, session_name),
            #                                             produce_data=self.produce_data,
            #                                             batch_size=batch_size)
            self.simulation_manager.connect_to_database(database_path=self.database_manager.get_database_path(),
                                                        normalize_data=self.database_manager.normalize)

        # Create a NetworkManager
        self.network_manager = network_manager
        self.network_manager.init_training(loss_fnc=loss_fnc,
                                           optimizer=optimizer,
                                           optimizer_kwargs=optimizer_kwargs,
                                           new_session=new_session,
                                           session=join(self.session_dir, session_name),
                                           save_intermediate_state_every=save_intermediate_state_every)
        self.network_manager.connect_to_database(database_path=self.database_manager.get_database_path(),
                                                 normalize_data=self.database_manager.normalize)
        if self.simulation_manager is not None:
            self.network_manager.link_clients(1 if self.simulation_manager.server is None
                                              else self.simulation_manager.nb_parallel_env)

        # Create a StatsManager
        self.stats_manager = StatsManager(session=join(self.session_dir, session_name)) if not debug else None

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

        # Save info file
        filename = join(join(self.session_dir, session_name), 'info.txt')
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

    def execute(self, user_training_loop: Optional[Callable] = None) -> None:
        """
        Launch the training Pipeline.
        Each event is already implemented for a basic pipeline but can also be rewritten via inheritance to describe a
        more complex Pipeline.
        """

        self.__default_training_loop() if user_training_loop is None else user_training_loop()
        # Training end
        for manager in (self.database_manager, self.network_manager, self.stats_manager, self.simulation_manager):
            if manager is not None:
                manager.close()


    def __default_training_loop(self):

        # Epoch condition
        while self.epoch_id < self.epoch_nb:

            # Epoch begin
            self.batch_id = 0

            # Batch condition
            while self.batch_id < self.batch_nb:

                # Batch begin
                id_epoch, nb_epoch = self.digits[0].format(self.epoch_id + 1), self.digits[0].format(self.epoch_nb)
                id_batch, nb_batch = self.digits[1].format(self.batch_id + 1), self.digits[1].format(self.batch_nb)
                self.progress_bar.title = f'Epoch n°{id_epoch}/{nb_epoch} - Batch n°{id_batch}/{nb_batch} '
                self.progress_bar.print()

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

                # Optimize
                batch_fwd, batch_bwd = self.network_manager.get_data(lines_id=self.data_lines)
                net_predict = self.network_manager.get_predict(batch_fwd=batch_fwd)
                loss = self.network_manager.get_loss(net_predict=net_predict, batch_bwd=batch_bwd)
                self.network_manager.optimize()

                # self.loss_dict = self.network_manager.get_prediction_and_loss(data_lines=self.data_lines,
                #                                                                   optimize=True)

                # Batch end
                self.batch_id += 1
                if self.stats_manager is not None:
                    self.stats_manager.add_train_batch_loss(loss,
                                                            self.epoch_id * self.batch_nb + self.batch_id)
                    # for key in self.loss_dict.keys():
                    #     if key != 'loss':
                    #         self.stats_manager.add_custom_scalar(tag=key,
                    #                                              value=self.loss_dict[key],
                    #                                              count=self.epoch_id * self.batch_nb + self.batch_id)

            # Epoch end
            self.epoch_id += 1
            if self.simulation_manager is not None and self.produce_data and \
                    (self.epoch_id == 0 or self.simulation_manager.always_produce):
                self.database_manager.compute_normalization()
                self.network_manager.db_handler.reload_normalization()
            if self.stats_manager is not None:
                self.stats_manager.add_train_epoch_loss(loss, self.epoch_id)
            self.network_manager.save_network()

    def __str__(self):

        description = "\n"
        description += f"# {self.__class__.__name__}\n"
        description += f"    Session repository: {self.session_dir}\n"
        description += f"    Number of epochs: {self.epoch_nb}\n"
        description += f"    Number of batches per epoch: {self.batch_nb}\n"
        description += f"    Number of samples per batch: {self.batch_size}\n"
        description += f"    Number of samples per epoch: {self.batch_nb * self.batch_size}\n"
        description += f"    Total: Number of batches : {self.batch_nb * self.epoch_nb}\n"
        description += f"           Number of samples : {self.nb_samples}\n"
        return description
