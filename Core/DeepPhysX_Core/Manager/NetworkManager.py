from typing import Any, Dict, Tuple, Optional
from os import listdir
from os.path import join as osPathJoin
from os.path import isdir, isfile
from numpy import copy, array, ndarray

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Utils.pathUtils import copy_dir, create_dir, get_first_caller


class NetworkManager:
    """
    Deals with all the interactions with the neural network. Predictions, saves, initialisation, loading,
    back-propagation, etc...

    :param Optional[BaseNetworkConfig] network_config: Specialisation containing the parameters of the network manager
    :param Manager manager: Manager that handle the network manager
    :param str session_name: Name of the newly created directory if session_dir is not defined
    :param Optional[str] session_dir: Name of the directory in which to write all the necessary data
    :param bool new_session: Define the creation of new directories to store data
    :param bool train: If True prediction will cause tensors gradient creation
    """

    def __init__(self,
                 network_config: Optional[BaseNetworkConfig] = None,
                 manager: Optional[Any] = None,
                 session_name: str = 'default',
                 session_dir: Optional[str] = None,
                 new_session: bool = True,
                 train: bool = True):

        self.name: str = self.__class__.__name__

        # Check network_config type
        if not isinstance(network_config, BaseNetworkConfig):
            raise TypeError(f"[{self.name}] Wrong 'network_config' type: BaseNetworkConfig required, "
                            f"get {type(network_config)}")
        # Check session_name type
        if type(session_name) != str:
            raise TypeError(f"[{self.name}] Wrong 'session_name' type: str required, get {type(session_name)}")
        # Check session_dir type and existence
        if session_dir is not None:
            if type(session_dir) != str:
                raise TypeError(f"[{self.name}] Wrong 'session_dir' type: str required, get {type(session_dir)}")
            if not isdir(session_dir):
                raise ValueError(f"[{self.name}] Given 'session_dir' does not exists: {session_dir}")
        # Check new_session type
        if type(new_session) != bool:
            raise TypeError(f"[{self.name}] Wrong 'new_session' type: bool required, get {type(new_session)}")
        # Check train type
        if type(train) != bool:
            raise TypeError(f"[{self.name}] Wrong 'train' type: bool required, get {type(train)}")

        # Storage management
        self.session_dir: str = session_dir if session_dir is not None else osPathJoin(get_first_caller(), session_name)
        self.new_session: bool = new_session
        self.network_dir: Optional[str] = None
        self.network_template_name: str = session_name + '_network_{}'

        # Network management
        self.manager: Any = manager
        if train and not network_config.training_stuff:
            raise ValueError(f"[{self.name}] Training requires a loss and an optimizer in your NetworkConfig")
        self.training: bool = train
        self.save_each_epoch: bool = network_config.save_each_epoch
        self.saved_counter: int = 0

        # Init network objects: Network, Optimization, DataTransformation
        self.network: Any = None
        self.optimization: Any = None
        self.data_transformation: Any = None
        self.network_config: BaseNetworkConfig = network_config
        self.set_network()

    def get_manager(self) -> Any:
        """
        | Return the Manager of the NetworkManager.

        :return: Manager that handles the NetworkManager
        """

        return self.manager

    def set_network(self) -> None:
        """
        | Set the network to the corresponding weight from a given file.
        """

        # Init network
        self.network = self.network_config.create_network()
        self.network.set_device()
        # Init optimization
        self.optimization = self.network_config.create_optimization()
        self.optimization.manager = self
        if self.optimization.loss_class:
            self.optimization.set_loss()

        # Init DataTransformation
        self.data_transformation = self.network_config.create_data_transformation()

        # Training
        if self.training:
            # Configure as training
            self.network.set_train()
            self.optimization.set_optimizer(self.network)
            # Setting network directory
            if self.new_session and self.network_config.network_dir and isdir(self.network_config.network_dir):
                self.network_dir = self.network_config.network_dir
                self.network_dir = copy_dir(self.network_dir, self.session_dir, dest_dir='network')
                self.load_network()
            else:
                self.network_dir = osPathJoin(self.session_dir, 'network/')
                self.network_dir = create_dir(self.network_dir, dir_name='network')

        # Prediction
        else:
            # Configure as prediction
            self.network.set_eval()
            # Need an existing network
            self.network_dir = osPathJoin(self.session_dir, 'network/')
            # Load parameters
            self.load_network()

    def load_network(self) -> None:
        """
        | Load an existing set of parameters to the network.
        """

        # Get eventual epoch saved networks
        networks_list = [osPathJoin(self.network_dir, f) for f in listdir(self.network_dir) if
                         isfile(osPathJoin(self.network_dir, f)) and f.__contains__('_network_.')]
        networks_list = sorted(networks_list)
        # Add the final saved network
        last_saved_network = [osPathJoin(self.network_dir, f) for f in listdir(self.network_dir) if
                              isfile(osPathJoin(self.network_dir, f)) and f.__contains__('network.')]
        networks_list = networks_list + last_saved_network
        which_network = self.network_config.which_network
        if len(networks_list) == 0:
            print(f"[{self.name}]: There is no network in {self.network_dir}. Shutting down.")
            quit(0)
        elif len(networks_list) == 1:
            which_network = 0
        elif len(networks_list) > 1 and which_network is None:
            print(f"[{self.name}] There is more than one network in this directory, loading the most trained by "
                  f"default. If you want to load another network please use the 'which_network' variable.")
            which_network = -1
        elif which_network > len(networks_list) > 1:
            print(f"[{self.name}] The selected network doesn't exist (index is too big), loading the most trained "
                  f"by default.")
            which_network = -1
        print(f"[{self.name}]: Loading network from {networks_list[which_network]}.")
        self.network.load_parameters(networks_list[which_network])

    def compute_prediction_and_loss(self, batch: Dict[str, ndarray],
                                    optimize: bool) -> Tuple[ndarray, Dict[str, float]]:
        """
        | Make a prediction with the data passed as argument, optimize or not the network

        :param Dict[str, ndarray] batch: Format {'input': numpy.ndarray, 'output': numpy.ndarray}.
                                         Contains the input value and ground truth to compare against
        :param bool optimize: If true run a back propagation

        :return: The prediction and the associated loss value
        """

        # Getting data from the data manager
        data_in = self.network.transform_from_numpy(batch['input'], grad=optimize)
        data_gt = self.network.transform_from_numpy(batch['output'], grad=optimize)
        loss_data = self.network.transform_from_numpy(batch['loss'], grad=False) if 'loss' in batch.keys() else None

        # Compute prediction
        data_in = self.data_transformation.transform_before_prediction(data_in)
        data_out = self.network.predict(data_in)

        # Compute loss
        data_out, data_gt = self.data_transformation.transform_before_loss(data_out, data_gt)
        loss_dict = self.optimization.compute_loss(data_out.reshape(data_gt.shape), data_gt, loss_data)
        # Optimizing network if training
        if optimize:
            self.optimization.optimize()
        # Transform prediction to be compatible with environment
        data_out = self.data_transformation.transform_before_apply(data_out)
        prediction = self.network.transform_to_numpy(data_out)
        return prediction, loss_dict

    def compute_online_prediction(self, network_input: ndarray) -> ndarray:
        """
        | Make a prediction with the data passed as argument.

        :param ndarray network_input: Input of the network=
        :return: The prediction
        """

        # Getting data from the data manager
        data_in = self.network.transform_from_numpy(copy(network_input), grad=False)

        # Compute prediction
        data_in = self.data_transformation.transform_before_prediction(data_in)
        pred = self.network.predict(data_in)
        pred, _ = self.data_transformation.transform_before_loss(pred)
        pred = self.data_transformation.transform_before_apply(pred)
        pred = self.network.transform_to_numpy(pred)
        return pred.reshape(-1)

    def save_network(self, last_save: bool = False) -> None:
        """
        | Save the network with the corresponding suffix, so they do not erase the last save.

        :param bool last_save: Do not add suffix if it's the last save
        """

        # Final session saving
        if last_save:
            path = self.network_dir + "network"
            print(f"[{self.name}] Saving final network at {path}.")
            self.network.save_parameters(path)

        # Intermediate states saving
        elif self.save_each_epoch:
            path = self.network_dir + self.network_template_name.format(self.saved_counter)
            self.saved_counter += 1
            print(f"[{self.name}] Saving intermediate network at {path}.")
            self.network.save_parameters(path)

    def close(self) -> None:
        """
        | Closing procedure.
        """

        if self.training:
            self.save_network(last_save=True)
        del self.network
        del self.network_config

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseNetwork object
        """

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
