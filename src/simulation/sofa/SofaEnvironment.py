from typing import Dict, Any, Union, Tuple
from numpy import ndarray

from SSD.SOFA.Rendering.UserAPI import UserAPI, Database

from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment

import Sofa
import Sofa.Simulation


class SofaEnvironment(Sofa.Core.Controller, BaseEnvironment):

    def __init__(self,
                 as_tcp_ip_client: bool = True,
                 instance_id: int = 1,
                 instance_nb: int = 1,
                 *args, **kwargs):
        """
        SofaEnvironment computes simulated data with SOFA for the Network and its training process.

        :param as_tcp_ip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False.
        :param instance_id: ID of the instance.
        :param instance_nb: Number of simultaneously launched instances.
        :param visualization_db: The path to the visualization Database or the visualization Database object to connect.
        """

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        # Warning: Define root node before init Environment
        self.root = Sofa.Core.Node('root')
        BaseEnvironment.__init__(self,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id,
                                 instance_nb=instance_nb,
                                 **kwargs)
        self.root.addObject(self)

    ##########################################################################################
    ##########################################################################################
    #                                 Environment initialization                             #
    ##########################################################################################
    ##########################################################################################

    def create(self) -> None:
        """
        Create the Environment. Automatically called when Environment is launched.
        Must be implemented by user.
        """

        raise NotImplementedError

    def init(self) -> None:
        """
        Initialize the Environment. Automatically called when Environment is launched.
        """

        # Init the root node
        Sofa.Simulation.init(self.root)

    def init_database(self) -> None:
        """
        Define the fields of the training dataset. Automatically called when Environment is launched.
        Must be implemented by user.
        """

        raise NotImplementedError

    def init_visualization(self) -> None:
        """
        Define the visualization objects to send to the Visualizer. Automatically called when Environment is launched.
        Not mandatory.
        """

        pass

    def save_parameters(self, **kwargs) -> None:
        """
        Save a set of parameters in the Database.
        """

        BaseEnvironment.save_parameters(self, **kwargs)

    def load_parameters(self) -> Dict[str, Any]:
        """
        Load a set of parameters from the Database.
        """

        return BaseEnvironment.load_parameters(self)

    ##########################################################################################
    ##########################################################################################
    #                                 Environment behavior                                   #
    ##########################################################################################
    ##########################################################################################

    async def step(self):
        """
        Compute the number of steps in the Environment specified by simulations_per_step in EnvironmentConfig.
        """

        Sofa.Simulation.animate(self.root, self.root.dt.value)
        await self.on_step()

    async def on_step(self):
        """
        Executed after an animation step.
        No mandatory.
        """

        pass

    def check_sample(self) -> bool:
        """
        Check if the current produced sample is usable for training.
        Not mandatory.

        :return: Current data can be used or not
        """

        return True

    def apply_prediction(self,
                         prediction: Dict[str, ndarray]) -> None:
        """
        Apply network prediction in environment.
        Not mandatory.

        :param prediction: Prediction data.
        """

        pass

    def close(self) -> None:
        """
        Close the Environment. Automatically called when Environment is shut down.
        Not mandatory.
        """

        pass

    ##########################################################################################
    ##########################################################################################
    #                                   Available requests                                   #
    ##########################################################################################
    ##########################################################################################

    def get_prediction(self, **kwargs) -> Dict[str, ndarray]:
        """
        Request a prediction from Network.

        :return: Network prediction.
        """

        return BaseEnvironment.get_prediction(self, **kwargs)

    def update_visualisation(self) -> None:
        """
        Triggers the Visualizer update.
        """

        # The Sofa UserAPI does automatic updates
        pass

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """

        description = BaseEnvironment.__str__(self)
        return description

    def _create_visualization(self,
                              visualization_db: Union[Database, Tuple[str, str]],
                              produce_data: bool = True) -> None:
        """
        Create a Factory for the Environment.
        """

        if type(visualization_db) == list:
            self.factory = UserAPI(root=self.root,
                                   database_dir=visualization_db[0],
                                   database_name=visualization_db[1],
                                   idx_instance=self.instance_id - 1,
                                   non_storing=not produce_data)
        else:
            self.factory = UserAPI(root=self.root,
                                   database=visualization_db,
                                   idx_instance=self.instance_id - 1,
                                   non_storing=not produce_data)
        self.init_visualization()
