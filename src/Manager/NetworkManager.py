from typing import Any, Dict, Tuple, Optional, List
from os import listdir
from os.path import join, isdir, isfile
from numpy import ndarray, array

from DeepPhysX.Core.Database.DatabaseHandler import DatabaseHandler
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Utils.path import copy_dir, create_dir


class NetworkManager:

    def __init__(self,
                 network_config: BaseNetworkConfig,
                 pipeline: str = '',
                 session: str = 'sessions/default',
                 new_session: bool = True):
        """
        Deals with all the interactions with the neural network: predictions, saves, initialisation, loading,
        back-propagation, etc.

        :param network_config: Specialisation containing the parameters of the network manager.
        :param pipeline: Type of the Pipeline.
        :param session: Path to the session directory.
        :param new_session: Define the creation of new directories to store data.
        """

        self.name: str = self.__class__.__name__

        # Storage variables
        self.database_handler: DatabaseHandler = DatabaseHandler()
        self.batch: Optional[Any] = None
        self.session: str = session
        self.new_session: bool = new_session
        self.network_dir: Optional[str] = None
        self.saved_counter: int = 0
        self.save_each_epoch: bool = network_config.save_each_epoch

        # Init Network
        self.network = network_config.create_network()
        self.network.set_device()
        if pipeline == 'training' and not network_config.training_stuff:
            raise ValueError(f"[{self.name}] Training requires a loss and an optimizer in your NetworkConfig")
        self.is_training: bool = pipeline == 'training'

        # Init Optimization
        self.optimization = network_config.create_optimization()
        if self.optimization.loss_class is not None:
            self.optimization.set_loss()

        # Init DataTransformation
        self.data_transformation = network_config.create_data_transformation()

        # Training configuration
        if self.is_training:
            self.network.set_train()
            self.optimization.set_optimizer(self.network)
            # Setting network directory
            if new_session and network_config.network_dir is not None and isdir(network_config.network_dir):
                self.network_dir = copy_dir(src_dir=network_config.network_dir,
                                            dest_dir=session,
                                            sub_folders='network')
                self.load_network(which_network=network_config.which_network)
            else:
                self.network_dir = create_dir(session_dir=session, session_name='network')

        # Prediction configuration
        else:
            self.network.set_eval()
            self.network_dir = join(session, 'network/')
            self.load_network(which_network=network_config.which_network)

    def get_database_handler(self):
        return self.database_handler

    def link_clients(self,
                     nb_clients: Optional[int] = None):
        if nb_clients is not None:
            fields = [(field_name, ndarray) for field_name in self.network.net_fields + self.network.pred_fields]
            self.database_handler.create_fields(table_name='Exchange', fields=fields)
            for _ in range(nb_clients):
                self.database_handler.add_data(table_name='Exchange', data={})

    def load_network(self,
                     which_network: int = -1) -> None:
        """
        Load an existing set of parameters to the network.
        """

        # Get eventual epoch saved networks
        networks_list = [join(self.network_dir, f) for f in listdir(self.network_dir) if
                         isfile(join(self.network_dir, f)) and f.__contains__('network_')]
        networks_list = sorted(networks_list)
        # Add the final saved network
        last_saved_network = [join(self.network_dir, f) for f in listdir(self.network_dir) if
                              isfile(join(self.network_dir, f)) and f.__contains__('network.')]
        networks_list = networks_list + last_saved_network
        if len(networks_list) == 0:
            raise FileNotFoundError(f"[{self.name}]: There is no network in {self.network_dir}.")
        elif len(networks_list) == 1:
            which_network = 0
        elif which_network > len(networks_list):
            print(f"[{self.name}] The network 'network_{self.saved_counter} doesn't exist, loading the most trained "
                  f"by default.")
            which_network = -1
        print(f"[{self.name}]: Loading network from {networks_list[which_network]}.")
        self.network.load_parameters(networks_list[which_network])

    def compute_prediction_and_loss(self,
                                    optimize: bool,
                                    data_lines: List[List[int]],
                                    normalization: Optional[Dict[str, List[float]]] = None) -> Tuple[ndarray, Dict[str, float]]:
        """
        Make a prediction with the data passed as argument, optimize or not the network

        :param batch_indices: Indices of the line of the Database that correspond to the current bach
        :param optimize: If true run a back propagation

        :return: The prediction and the associated loss value
        """

        # Define in and out batches
        batches = {}
        normalization = {} if normalization is None else normalization
        for side, fields in zip(['net', 'opt'], [self.network.net_fields, self.network.opt_fields]):

            # Get the batch from the Database
            batch = self.database_handler.get_lines(table_name='Training',
                                                    fields=fields,
                                                    lines_id=data_lines)

            # Apply normalization and convert to tensor
            for field in batch.keys():
                batch[field] = array(batch[field])
                if field in normalization:
                    batch[field] = self.normalize_data(data=batch[field],
                                                       normalization=normalization[field])
                batch[field] = self.network.numpy_to_tensor(data=batch[field],
                                                            grad=optimize)
            batches[side] = batch
        data_net, data_opt = batches.values()

        # Compute prediction
        data_net = self.data_transformation.transform_before_prediction(data_net)
        data_pred = self.network.predict(data_net)

        # Compute loss
        data_pred, data_opt = self.data_transformation.transform_before_loss(data_pred, data_opt)
        data_loss = self.optimization.compute_loss(data_pred, data_opt)

        # Optimizing network if training
        if optimize:
            self.optimization.optimize()

        # Transform prediction to be compatible with environment
        data_pred = self.data_transformation.transform_before_apply(data_pred)
        for field in data_pred:
            data_pred[field] = self.network.tensor_to_numpy(data_pred[field])
            if field in normalization.keys():
                data_pred[field] = self.normalize_data(data=data_pred[field],
                                                       normalization=normalization[field],
                                                       reverse=True)
        return data_pred, data_loss

    def compute_online_prediction(self,
                                  instance_id: int,
                                  normalization: Optional[Dict[str, List[float]]] = None) -> None:
        """
        Make a prediction with the data passed as argument.

        :param network_input: Input of the network
        :return: The prediction
        """

        # Get Network data
        normalization = {} if normalization is None else normalization
        sample = self.data_db.get_line(table_name='Prediction',
                                       fields=self.network.net_fields,
                                       line_id=instance_id)
        del sample['id']

        # Apply normalization and convert to tensor
        for field in sample.keys():
            sample[field] = array(sample[field])
            if field in normalization.keys():
                sample[field] = self.normalize_data(data=sample[field],
                                                    normalization=normalization[field])
            sample[field] = self.network.numpy_to_tensor(data=sample[field])

        # Compute prediction
        data_net = self.data_transformation.transform_before_prediction(sample)
        data_pred = self.network.predict(data_net)
        data_pred, _ = self.data_transformation.transform_before_loss(data_pred)
        data_pred = self.data_transformation.transform_before_apply(data_pred)

        # Return the prediction
        for field in data_pred.keys():
            data_pred[field] = self.network.tensor_to_numpy(data=data_pred[field])
            if field in normalization.keys():
                data_pred[field] = self.normalize_data(data=data_pred[field],
                                                       normalization=normalization[field],
                                                       reverse=True)
            data_pred[field].reshape(-1)
        self.data_db.update(table_name='Prediction',
                            data=data_pred,
                            line_id=instance_id)

    @classmethod
    def normalize_data(cls,
                       data: ndarray,
                       normalization: List[float],
                       reverse: bool = False) -> ndarray:
        """
        Apply or unapply normalization following current standard score.

        :param data: Data to normalize.
        :param field: Specify if data is an 'input' or an 'output'.
        :param reverse: If False, apply normalization; if False, unapply normalization.
        :return: Data with applied or misapplied normalization.
        """

        if reverse:
            # Unapply normalization
            return (data * normalization[1]) + normalization[0]
        # Apply normalization
        return (data - normalization[0]) / normalization[1]

    def save_network(self, last_save: bool = False) -> None:
        """
        | Save the network with the corresponding suffix, so they do not erase the last save.

        :param bool last_save: Do not add suffix if it's the last save
        """

        # Final session saving
        if last_save:
            path = join(self.network_dir, 'network')
            print(f"[{self.name}] Saving final network at {self.network_dir}.")
            self.network.save_parameters(path)

        # Intermediate states saving
        elif self.save_each_epoch:
            path = self.network_dir + self.network_template_name.format(self.saved_counter)
            self.saved_counter += 1
            print(f"[{self.name}] Saving intermediate network at {path}.")
            self.network.save_parameters(path)

    def close(self) -> None:
        """
        Closing procedure.
        """

        if self.is_training:
            self.save_network(last_save=True)
        del self.network

    def __str__(self) -> str:

        description = "\n"
        description += f"# {self.__class__.__name__}\n"
        description += f"    Network Directory: {self.network_dir}\n"
        description += f"    Save each Epoch: {self.save_each_epoch}\n"
        description += f"    Managed objects: Network: {self.network.__class__.__name__}\n"
        description += f"                     Optimization: {self.optimization.__class__.__name__}\n"
        description += f"                     Data Transformation: {self.data_transformation.__class__.__name__}\n"
        description += str(self.network)
        description += str(self.optimization)
        description += str(self.data_transformation)
        return description
