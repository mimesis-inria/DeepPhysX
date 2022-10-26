from typing import Dict, Union, Any, Iterable, Optional
from tensorboardX import SummaryWriter
from tensorboard import program
from webbrowser import open as w_open
from numpy import full, inf, array, ndarray, append, concatenate
from os.path import join


def generate_default_scene():
    return {'camera': {'cls': 'PerspectiveCamera', 'fov': 75},
            'light': {'cls': 'AmbientLight', 'color': '#ffffff', 'intensity': 0.75},
            }


def generate_default_material():
    return {'material': {'cls': 'MeshStandardMaterial', 'roughness': 1, 'metalness': 0, 'color': '#8888ff'}}


class StatsManager:

    def __init__(self,
                 session: str,
                 keep_losses: bool = False):
        """
        Record all given values using the tensorboard framework. Open a tab in the navigator to inspect these values
        during the training.

        :param str log_dir: Path of the created directory
        :param bool keep_losses: If True Allow saving loss to .csv file
        """

        self.name: str = self.__class__.__name__

        # Init writer
        self.log_dir: str = join(session, 'stats/')
        self.writer: SummaryWriter = SummaryWriter(self.log_dir)

        # Open Tensorboard
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir])
        url = tb.launch()
        w_open(url)

        # Values
        self.mean: ndarray = full(4, inf)  # Contains in the 1st dimension the mean, and 2nd the variance of the mean
        self.train_loss: ndarray = array([])
        self.keep_losses: bool = keep_losses
        self.tag_dict: Dict[str, int] = {}

    def add_train_batch_loss(self, value: float, count: int) -> None:
        """
        | Add batch loss to tensorboard framework. Also compute mean and variance.

        :param float value: Value to store
        :param int count: ID of the value
        """

        var = self.update_mean_get_var(0, value, count + 1)
        self.writer.add_scalar("Train/Batch/Loss", value, count)
        self.writer.add_scalar("Train/Batch/Mean", self.mean[0], count)
        if var is not None:
            self.writer.add_scalar("Train/Batch/Variance", var, count)
        if self.keep_losses is True:
            self.train_loss = append(self.train_loss, value)

    def add_train_epoch_loss(self, value: float, count: int) -> None:
        """
        | Add epoch loss to tensorboard framework. Also compute mean and variance.

        :param float value: Value to store
        :param int count: ID of the value
        """

        var = self.update_mean_get_var(1, value, count + 1)
        self.writer.add_scalar("Train/Epoch/Loss", value, count)
        self.writer.add_scalar("Train/Epoch/Mean", self.mean[1], count)
        if var is not None:
            self.writer.add_scalar("Train/Epoch/Variance", var, count)

    def add_train_test_batch_loss(self, train_value: float, test_value: float, count: int) -> None:
        """
        | Add train and test batch loss to tensorboard framework.

        :param float train_value: Value of the training batch
        :param float test_value: Value of the testing batch
        :param int count: ID of the value
        """

        if train_value is not None:
            self.writer.add_scalars("Combined/Batch/Loss", {'Train': train_value}, count)
        if test_value is not None:
            self.writer.add_scalars("Combined/Batch/Loss", {'Test': test_value}, count)

    def add_values_multi_plot(self, graph_name: str, tags: Iterable, values: Iterable, counts: Iterable) -> None:
        """
        | Plot multiples value on the same graph

        :param str graph_name: Name of the graph
        :param Iterable tags: Iterable containing the names of the values
        :param Iterable values: Iterable containing the value
        :param Iterable counts: ID of the plots
        """

        for t, v, c in zip(tags, values, counts):
            self.writer.add_scalars(graph_name, {t: v}, c)

    def add_test_loss(self, value: float, count: int) -> None:
        """
        | Add test loss to tensorboard framework. Also compute mean and variance.

        :param float value: Value to store
        :param int count: ID of the value
        """

        var = self.update_mean_get_var(2, value, count + 1)
        self.writer.add_scalar("Test/Valid/Loss", value, count)
        self.writer.add_scalar("Test/Valid/Mean", self.mean[2], count)
        if var is not None:
            self.writer.add_scalar("Test/Valid/Variance", var, count)

    def add_test_loss_OOB(self, value: float, count: int) -> None:
        """
        | Add out of bound test loss to tensorboard framework. Also compute mean and variance.

        :param float value: Value to store
        :param int count: ID of the value
        """

        var = self.update_mean_get_var(3, value, count + 1)
        self.writer.add_scalar("Test/Out-of-boundaries/Loss", value, count)
        self.writer.add_scalar("Test/Out-of-boundaries/Mean", self.mean[3], count)
        if var is not None:
            self.writer.add_scalar("Test/Out-of-boundaries/Variance", var, count)

    def add_custom_scalar(self, tag: str, value: float, count: int) -> None:
        """
        | Add a custom scalar to tensorboard framework.

        :param str tag: Graph name
        :param float value: Value to store
        :param int count: ID of the value
        """

        self.writer.add_scalar(tag, value, count)

    def add_custom_scalar_full(self, tag: str, value: float, count: int) -> None:
        """
        | Add a custom scalar to tensorboard framework. Also compute mean and variance.

        :param str tag: Graph name
        :param float value: Value to store
        :param int count: ID of the value
        """

        try:
            self.tag_dict[tag]
        except KeyError:
            self.tag_dict[tag] = len(self.tag_dict) + 4  # Size of self.mean at the initialization
        var = self.update_mean_get_var(self.tag_dict[tag], value, count + 1)
        self.writer.add_scalar(tag + "/Value", value, count)
        self.writer.add_scalar(tag + "/Mean", self.mean[self.tag_dict[tag]], count)
        if var is not None:
            self.writer.add_scalar(tag + "/Variance", var, count)

    def update_mean_get_var(self, index: int, value: float, count: int) -> Optional[ndarray]:
        """
        | Update mean and return the variance of the selected value

        :param float value: Value to add in the computation of the mean
        :param int index: Target that is updated by the value
        :param int count: ID of the value

        :return: The mean and var values over the last 50 elements
        """

        if index > self.mean.shape[0] - 1:
            self.mean = concatenate((self.mean, full(index - (self.mean.shape[0] - 1), inf)))
        if self.mean[index] == inf:
            self.mean[index] = value
            return None
        else:
            # Mean value over the last 50 elements
            n = count if count < 50 else 50
            variance = self.mean[index]
            self.mean[index] = self.mean[index] + (value - self.mean[index]) / n
            return variance - self.mean[index]

    def add_3D_point_cloud(self, tag: str, vertices: ndarray, colors: ndarray = None, b_n_3: bool = False,
                           config_dict: Optional[Dict[Any, Any]] = None) -> None:
        """
        Add 3D point cloud to tensorboard framework

        :param str tag: Data identifier
        :param DataContainer vertices: List of the 3D coordinates of vertices.
        :param DataContainer colors: Colors for each vertex
        :param bool b_n_3: Data is in the format [batch_size, number_of_nodes, 3]
        :param Optional[Dict[Any, Any]] config_dict: Dictionary with ThreeJS classes names and configuration.
        """

        if config_dict is None:
            config_dict = {**generate_default_scene(), **generate_default_material()}
        # Information should be written using (Batch, number of vertex, 3) as shape. Hence, if not we emulate it
        if not b_n_3:
            v = vertices[None, :, :]
            if colors is not None:
                c = colors[None, :, :]
        else:
            v = vertices
            c = colors

        self.writer.add_mesh(tag=tag, vertices=v, colors=c, config_dict=config_dict)

    def add_3D_mesh(self, tag: str, vertices: ndarray, colors: Optional[ndarray] = None,
                    faces: Optional[ndarray] = None, b_n_3: bool = False,
                    config_dict: Optional[Dict[Any, Any]] = None) -> None:
        """
        | Add 3D Mesh cloud to tensorboard framework.

        :param str tag: Data identifier
        :param ndarray vertices: List of the 3D coordinates of vertices.
        :param Optional[ndarray] colors: Colors for each vertex
        :param Optional[ndarray] faces: Indices of vertices within each triangle.
        :param bool b_n_3: Data is in the format [batch_size, number_of_nodes, 3]
        :param Optional[Dict[Any, Any]] config_dict: Dictionary with ThreeJS classes names and configuration.
        """

        if config_dict is None:
            config_dict = {**generate_default_scene(), **generate_default_material()}
        # Information should be written using (Batch, number of vertex, 3) as shape. Hence, if not we emulate it
        if not b_n_3:
            v = vertices[None, :, :]
            if colors is not None:
                c = colors[None, :, :]
            if faces is not None:
                f = faces[None, :, :]
        else:
            v = vertices
            c = colors
            f = faces
        self.writer.add_mesh(tag=tag, vertices=v, colors=c, faces=f, config_dict=config_dict)

    def add_network_weight_grad(self, network: Any, count: int, save_weights: bool = False,
                                save_gradients: bool = True) -> None:
        """
        | Add network weights and gradiant if specified to tensorboard framework.

        :param BaseNetwork network: Network you want to display
        :param int count: ID of the sample
        :param bool save_weights: If True will save weights to tensorboard
        :param bool save_gradients: If True will save gradient to tensorboard
        """

        for tag, value in network.named_parameters():
            tag = tag.replace('.', '/')
            if save_weights:
                self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), count)
            if save_gradients:
                self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), count)

    def close(self) -> None:
        """
        | Closing procedure

        :return:
        """

        self.writer.close()
        del self.train_loss

    def __str__(self) -> str:
        """
        :return: A string containing valuable information about the StatsManager
        """

        description = "\n"
        description += f"# {self.name}\n"
        description += f"    Stats repository: {self.log_dir}\n"
        description += f"    Store losses as CSV: {self.keep_losses}\n"
        if self.keep_losses:
            description += f"    CSV file path: {self.log_dir}\n"
        return description
