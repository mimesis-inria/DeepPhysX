from typing import Optional, Dict, Any, Type, Union, List, Tuple
from os import sep, listdir, remove
from os.path import isfile, isdir, join
from numpy import ndarray, array
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch import Tensor

from DeepPhysX.networks.network_wrapper import NetworkWrapper
from DeepPhysX.database.database_handler import DatabaseHandler
from DeepPhysX.utils.path import create_dir, copy_dir


class NetworkManager:

    def __init__(self,
                 network_architecture: Type[Module],
                 network_kwargs: Dict[str, Any],
                 data_forward_fields: Union[str, List[str]],
                 data_backward_fields: Union[str, List[str]],
                 network_dir: Optional[str] = None,
                 network_load_id: int = -1,
                 data_type: str = 'float32'):
        """
        NetworkManager handles the neural network instance for inference and training, load and save.

        :param network_architecture: The neural network architecture.
        :param network_kwargs: Dict of kwargs to create an instance of the neural network architecture.
        :param data_forward_fields:
        :param data_backward_fields:
        :param network_dir:
        :param network_load_id:
        :param data_type:
        """

        # Network repository variables
        self.network_dir = network_dir
        self.network_load_id = network_load_id
        self.saved_counter: int = 0
        self.save_every: int = 0
        self.network_template_name: str = ''

        # Network instance
        self.__network = NetworkWrapper(network_architecture=network_architecture,
                                        network_kwargs=network_kwargs,
                                        data_type=data_type)
        self.__network.set_device()

        # Training materials variables
        self.__loss_fnc: Optional[_Loss] = None
        self.__loss_value: Optional[Any] = None
        self.__optimizer: Optional[Optimizer] = None

        # Database access variables
        self.__db_handler: DatabaseHandler = DatabaseHandler()
        self.data_forward_fields = data_forward_fields if isinstance(data_forward_fields, list) else [data_forward_fields]
        self.data_backward_fields = data_backward_fields if isinstance(data_backward_fields, list) else [data_backward_fields]

    ################
    # Init methods #
    ################

    def init_training_pipeline(self,
                               loss_fnc: Type[_Loss],
                               optimizer: Type[Optimizer],
                               optimizer_kwargs: Dict[str, Any],
                               new_session: bool,
                               session: str = 'sessions/default',
                               save_intermediate_state_every: int = 0):
        """
        Init the NetworkManager for the training pipeline.

        :param loss_fnc:
        :param optimizer:
        :param optimizer_kwargs:
        :param new_session:
        :param session:
        :param save_intermediate_state_every:
        """

        # Configure the Network for the current pipeline
        self.__network.train()
        self.network_template_name = session.split(sep)[-1] + '_network_{}'

        # Create the training materials
        self.__loss_fnc = loss_fnc()
        self.__optimizer = optimizer(params=self.__network.parameters(), **optimizer_kwargs)
        self.save_every = save_intermediate_state_every

        # Case 1: Training from an existing Network state of parameters
        if new_session and self.network_dir is not None and isdir(self.network_dir):

            # TODO
            self.network_dir = copy_dir(src_dir=self.network_dir,
                                        dest_dir=session,
                                        sub_folders='networks')
            self.load_network(network_id=self.network_load_id)

        # Case 2: Training from scratch
        else:

            # Create a new directory
            self.network_dir = create_dir(session_dir=session, session_name='networks')

    def init_prediction_pipeline(self, session: str = 'sessions/default'):
        """
        Init method for the prediction pipeline.

        :param session:
        """

        # Configure the Network for the current pipeline
        self.__network.eval()

        # Load the Network state of parameters
        self.network_dir = join(session, 'networks')
        self.load_network(network_id=self.network_load_id)

    @staticmethod
    def __check_init(foo):
        """
        Wrapper to check that an 'init_*_pipeline' method was called before to use the NetworkManager.
        """

        def wrapper(self, *args, **kwargs):
            if not self.__network.is_ready:
                raise ValueError(f"[NetworkManager] The manager is not completely initialized; please use one of the "
                                 f"'init_*_pipeline' methods.")
            return foo(self, *args, **kwargs)

        return wrapper

    ##############################
    # Database access management #
    ##############################

    def connect_to_database(self,
                            database_path: Tuple[str, str],
                            normalize_data: bool):
        """
        Connect the NetworkManager to the Database.

        :param database_path: Path of the Database to connect to.
        :param normalize_data:
        """

        self.__db_handler.init(database_path=database_path, normalize_data=normalize_data)

    def reload_normalization(self):

        self.__db_handler.reload_normalization()

    def link_clients(self, nb_clients: Optional[int] = None) -> None:
        """
        Update the data Exchange Database with a new line for each TcpIpClient.

        :param nb_clients: Number of Clients to connect.
        """

        # TODO: move this to the Simulation manager
        if nb_clients is not None:
            # Create the networks fields in the Exchange Database
            fields = [(field_name, ndarray) for field_name in self.__db_handler.get_fields() if field_name not in ['id', 'env_id']]
            self.__db_handler.create_fields(fields=fields, exchange=True)
            # Add an empty line for each Client
            for _ in range(nb_clients):
                self.__db_handler.add_data(exchange=True, data={})

    ##############################
    # Network storage management #
    ##############################

    def load_network(self, network_id: int) -> None:
        """
        Load a Network state of parameters.
        """

        # Get the list of the saved weights files
        files = sorted([join(self.network_dir, f) for f in listdir(self.network_dir)
                        if isfile(join(self.network_dir, f)) and f.endswith('.pth')])

        # Check the network id
        if len(files) == 0:
            raise FileNotFoundError(f"[NetworkManager] No saved weights in {self.network_dir}.")
        elif len(files) == 1:
            network_id = 0
        elif network_id > len(files):
            network_id = -1
            print(f"[NetworkManager] The networks 'network_{self.saved_counter} does not exist, loading the most "
                  f"trained by default.")

        # Load the set of parameters
        self.__network.load(files[network_id])
        print(f"[NetworkManager] Load weights from {files[network_id]}.")

    def save_network(self, final_save: bool = False) -> None:
        """
        Save a Network state of parameters.
        """

        # Case 1: Final save
        if final_save:

            path = join(self.network_dir, 'network')
            self.__network.save(path=path)
            print(f"[NetworkManager] Save final set of weights at {path}.")

        # Case 2: Intermediate save
        else:

            self.saved_counter += 1

            # Remove previous temporary backup file(s)
            for temp in [f for f in listdir(self.network_dir) if 'temp' in f]:
                remove(join(self.network_dir, temp))

            # Case 2.1: Save intermediate state
            if self.save_every > 0 and self.saved_counter % self.save_every == 0:
                path = join(self.network_dir, self.network_template_name.format(self.saved_counter))
                self.__network.save(path=path)
                print(f"[NetworkManager] Save intermediate set of weights at {path}.")

            # Case 2.2: Save backup file
            else:
                path = join(self.network_dir, f'temp_{self.saved_counter}')
                self.__network.save(path=path)

    #####################################
    # Network optimization & prediction #
    #####################################

    @__check_init
    def get_data(self, lines_id: List[int]):

        # 1. Get data from the Database
        batch_fwd = self.__db_handler.get_batch(lines_id=lines_id, fields=self.data_forward_fields)
        batch_bwd = self.__db_handler.get_batch(lines_id=lines_id, fields=self.data_backward_fields)

        # 2. Convert data to PyTorch & normalize if required
        for batch in (batch_fwd, batch_bwd):
            for field_name in batch.keys():
                batch[field_name] = array(batch[field_name])
                if self.__db_handler.do_normalize and field_name in self.__db_handler.normalization:
                    batch[field_name] = self.normalize_data(data=batch[field_name],
                                                            normalization=self.__db_handler.normalization[field_name])
                batch[field_name] = self.__network.to_torch(tensor=batch[field_name],
                                                          grad=self.__network.is_training)

        return batch_fwd, batch_bwd

    @__check_init
    def get_predict(self, batch_fwd: Dict[str, Tensor]):
        """
        """

        return self.__network.predict(*batch_fwd.values())

    @__check_init
    def predict_to_db(self, batch_fwd: Dict[str, Tensor], instance_id: int):
        """
        """

        net_predict = self.__network.predict(*batch_fwd.values())
        data = {}
        for field, value in zip(self.data_backward_fields, net_predict):
            data[field] = self.__network.to_numpy(tensor=net_predict[0])
            if field in self.__db_handler.normalization.keys():
                data[field] = self.normalize_data(data=data[field], normalization=self.__db_handler.normalization[field],
                                                  reverse=True)
            data[field].reshape(-1)
        self.__db_handler.update(exchange=True, data=data, line_id=instance_id)

    @__check_init
    def get_loss(self, batch_bwd: Dict[str, Tensor], net_predict):
        """
        """

        net_predict = net_predict if isinstance(net_predict, tuple) else (net_predict,)
        self.__loss_value = self.__loss_fnc(*net_predict, *batch_bwd.values())
        return self.__loss_value.item()

    @__check_init
    def optimize(self):
        """
        """

        self.__optimizer.zero_grad()
        self.__loss_value.backward()
        self.__optimizer.step()

    @__check_init
    def get_prediction(self, instance_id: int) -> None:
        """

        """

        # 1. Normalize the batch of data
        normalization = self.__db_handler.normalization
        sample = self.__db_handler.get_data(exchange=True, line_id=instance_id, fields=self.data_forward_fields)
        del sample['id']
        for field in sample.keys():
            sample[field] = array([sample[field]])
            if field in normalization:
                sample[field] = self.normalize_data(data=sample[field], normalization=normalization[field])
            sample[field] = self.__network.to_torch(tensor=sample[field], grad=False)

        # 2. Compute prediction
        net_predict = self.__network.predict(*(sample[field_name] for field_name in self.data_forward_fields))
        net_predict = net_predict if isinstance(net_predict, tuple) else (net_predict,)

        # 3. Return the prediction
        data = {}
        for field, value in zip(self.data_backward_fields, net_predict):
            data[field] = self.__network.to_numpy(tensor=net_predict[0])
            if field in normalization.keys():
                data[field] = self.normalize_data(data=data[field], normalization=normalization[field],
                                                  reverse=True)
            data[field].reshape(-1)
        self.__db_handler.update(exchange=True, data=data, line_id=instance_id)

    @classmethod
    def normalize_data(cls,
                       data: ndarray,
                       normalization: List[float],
                       reverse: bool = False) -> ndarray:
        """
        Apply or unapply normalization following current standard score.

        :param data: Data to normalize.
        :param normalization: Normalization coefficients.
        :param reverse: If True, apply normalization; if False, unapply normalization.
        :return: Data with applied or misapplied normalization.
        """

        # Unapply normalization
        if reverse:
            return (data * normalization[1]) + normalization[0]

        # Apply normalization
        return (data - normalization[0]) / normalization[1]

    ###################
    # Manager methods #
    ###################

    def close(self) -> None:
        """
        Launch the closing procedure of the NetworkManager.
        """

        if self.__network.is_training:
            self.save_network(final_save=True)
        del self.__network

    # def __str__(self) -> str:
    #
    #     description = "\n"
    #     description += f"# {self.__class__.__name__}\n"
    #     description += f"    networks Directory: {self.network_dir}\n"
    #     description += f"    Managed objects: networks: {self.__network.__class__.__name__}\n"
    #     description += str(self.__network)
    #     return description
