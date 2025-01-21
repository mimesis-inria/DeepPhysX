from typing import Optional, Dict, Any, Type, Union, List, Tuple
from os import sep, listdir, remove
from os.path import isfile, isdir, join
from numpy import ndarray, array
from numpy.random import normal
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch import Tensor

from DeepPhysX.net.network_wrapper import NetworkWrapper
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

        """

        # Define Network Module
        self.network = NetworkWrapper(network_architecture=network_architecture,
                                      network_kwargs=network_kwargs,
                                      data_type=data_type)
        self.network.set_device()

        # Define Database access
        self.db_handler: DatabaseHandler = DatabaseHandler()
        self.data_forward_fields = data_forward_fields if isinstance(data_forward_fields, list) else [data_forward_fields]
        self.data_backward_fields = data_backward_fields if isinstance(data_backward_fields, list) else [data_backward_fields]

        # Define session storage
        self.network_dir = network_dir
        self.network_load_id = network_load_id
        # self.network_template_name: str = session.split(sep)[-1] + '_network_{}'
        self.saved_counter: int = 0
        self.save_every: int = 0

        # Training materials
        self.__is_training: bool = False
        self.loss_fnc: Optional[_Loss] = None
        self.__loss_value: Optional[Any] = None
        self.optimizer: Optional[Optimizer] = None

    ################
    # INIT METHODS #
    ################

    def init_training(self,
                      loss_fnc: Type[_Loss],
                      optimizer: Type[Optimizer],
                      optimizer_kwargs: Dict[str, Any],
                      new_session: bool,
                      session: str = 'sessions/default',
                      save_intermediate_state_every: int = 0,):
        """
        """

        self.__is_training = True
        self.network.train()
        self.loss_fnc = loss_fnc()
        self.optimizer = optimizer(params=self.network.parameters(), **optimizer_kwargs)

        if new_session and self.network_dir is not None and isdir(self.network_dir):
            self.network_dir = copy_dir(src_dir=self.network_dir,
                                        dest_dir=session,
                                        sub_folders='networks')
            self.load_network(network_id=self.network_load_id)
        else:
            self.network_dir = create_dir(session_dir=session, session_name='networks')

        self.save_every = save_intermediate_state_every

    def init_prediction(self,
                        session: str = 'sessions/default',):
        """
        """

        self.network.eval()
        self.network_dir = join(session, 'networks')
        self.load_network(network_id=self.network_load_id)

    #######################
    # DATABASE MANAGEMENT #
    #######################

    def connect_to_database(self,
                            database_path: Tuple[str, str],
                            normalize_data: bool):

        self.db_handler.init(database_path=database_path, normalize_data=normalize_data)

    def get_database_handler(self) -> DatabaseHandler:
        """
        Get the DatabaseHandler of the NetworkManager.
        """

        return self.db_handler

    def link_clients(self, nb_clients: Optional[int] = None) -> None:
        """
        Update the data Exchange Database with a new line for each TcpIpClient.

        :param nb_clients: Number of Clients to connect.
        """

        if nb_clients is not None:
            # Create the networks fields in the Exchange Database
            fields = [(field_name, ndarray) for field_name in self.db_handler.get_fields() if field_name not in ['id', 'env_id']]
            self.db_handler.create_fields(fields=fields, exchange=True)
            # Add an empty line for each Client
            for _ in range(nb_clients):
                self.db_handler.add_data(exchange=True, data={})

    ##############################
    # NETWORK WEIGHTS MANAGEMENT #
    ##############################

    def load_network(self, network_id: int) -> None:
        """

        """

        # 1. Get the list of the saved weights files
        files = sorted([join(self.network_dir, f) for f in listdir(self.network_dir)
                        if isfile(join(self.network_dir, f)) and f.endswith('.pth')])

        # 2. Check the network id
        if len(files) == 0:
            raise FileNotFoundError(f"[{self.__class__.__name__}] No saved weights in {self.network_dir}.")
        elif len(files) == 1:
            network_id = 0
        elif network_id > len(files):
            print(f"[{self.__class__.__name__}] The networks 'network_{self.saved_counter} doesn't exist, loading the most trained "
                  f"by default.")
            network_id = -1

        # 3. Load the set of parameters
        print(f"[{self.__class__.__name__}] Loading saved weights from {files[network_id]}.")
        self.network.load(files[network_id])

    def save_network(self, final_save: bool = False) -> None:
        """

        """

        # 1. Final save case
        if final_save:
            path = join(self.network_dir, 'network')
            print(f"[{self.__class__.__name__}] Saving final set of weights at {path}.")
            self.network.save(path=path)

        # 2. Intermediate save case
        else:
            # Remove previous temporary backup file(s)
            for temp in [f for f in listdir(self.network_dir) if 'temp' in f]:
                remove(join(self.network_dir, temp))
            # Save intermediate state
            self.saved_counter += 1
            if self.save_every > 0 and self.saved_counter % self.save_every == 0:
                path = join(self.network_dir, self.network_template_name.format(self.saved_counter))
                print(f"[{self.__class__.__name__}] Saving intermediate set of weights at {path}.")
                self.network.save(path=path)
            # Save backup file
            else:
                path = join(self.network_dir, f'temp_{self.saved_counter}')
                self.network.save(path=path)

    #####################################
    # NETWORK OPTIMIZATION & PREDICTION #
    #####################################

    def get_data(self, lines_id: List[int]):

        # 1. Get data from the Database
        batch_fwd = self.db_handler.get_batch(lines_id=lines_id,
                                              fields=self.data_forward_fields)
        batch_bwd = self.db_handler.get_batch(lines_id,
                                              fields=self.data_backward_fields)


        # 2. Convert data to PyTorch & normalize if required
        for batch in (batch_fwd, batch_bwd):
            for field_name in batch.keys():
                batch[field_name] = array(batch[field_name])
                if self.db_handler.do_normalize and field_name in self.db_handler.normalization:
                    batch[field_name] = self.normalize_data(data=batch[field_name],
                                                            normalization=self.db_handler.normalization[field_name])
                batch[field_name] = self.network.to_torch(tensor=batch[field_name],
                                                          grad=self.__is_training)

        return batch_fwd, batch_bwd

    def get_predict(self, batch_fwd: Dict[str, Tensor]):
        """
        """

        return self.network.predict(*batch_fwd.values())

    def predict_to_db(self, batch_fwd: Dict[str, Tensor], instance_id: int):
        """
        """

        net_predict = self.network.predict(*batch_fwd.values())
        data = {}
        for field, value in zip(self.data_backward_fields, net_predict):
            data[field] = self.network.to_numpy(tensor=net_predict[0])
            if field in self.db_handler.normalization.keys():
                data[field] = self.normalize_data(data=data[field], normalization=self.db_handler.normalization[field],
                                                  reverse=True)
            data[field].reshape(-1)
        self.db_handler.update(exchange=True, data=data, line_id=instance_id)

    def get_loss(self, batch_bwd: Dict[str, Tensor], net_predict):
        """
        """

        net_predict = net_predict if isinstance(net_predict, tuple) else (net_predict,)
        self.__loss_value = self.loss_fnc(*net_predict, *batch_bwd.values())
        return self.__loss_value.item()

    def optimize(self):
        """
        """

        self.optimizer.zero_grad()
        self.__loss_value.backward()
        self.optimizer.step()

    def get_prediction(self, instance_id: int) -> None:
        """

        """

        # 1. Normalize the batch of data
        normalization = self.db_handler.normalization
        sample = self.db_handler.get_data(exchange=True, line_id=instance_id)
        del sample['id']
        for field in sample.keys():
            sample[field] = array([sample[field]])
            if field in normalization:
                sample[field] = self.normalize_data(data=sample[field], normalization=normalization[field])
            sample[field] = self.network.to_torch(tensor=sample[field], grad=False)

        # 2. Compute prediction
        inp = (sample[field_name] for field_name in self.data_forward_fields)
        net_predict = self.network.predict(*inp)
        net_predict = net_predict if isinstance(net_predict, tuple) else (net_predict,)

        # 3. Return the prediction
        data = {}
        for field, value in zip(self.data_backward_fields, net_predict):
            data[field] = self.network.to_numpy(tensor=net_predict[0])
            if field in normalization.keys():
                data[field] = self.normalize_data(data=data[field], normalization=normalization[field],
                                                  reverse=True)
            data[field].reshape(-1)
        self.db_handler.update(exchange=True, data=data, line_id=instance_id)

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
    # MANAGER METHODS #
    ####################

    def close(self) -> None:
        """
        Launch the closing procedure of the NetworkManager.
        """

        if self.__is_training:
            self.save_network(final_save=True)
        del self.network

    def __str__(self) -> str:

        description = "\n"
        description += f"# {self.__class__.__name__}\n"
        description += f"    networks Directory: {self.network_dir}\n"
        description += f"    Managed objects: networks: {self.network.__class__.__name__}\n"
        description += str(self.network)
        return description
