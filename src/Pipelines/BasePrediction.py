from typing import Optional
from numpy import ndarray

from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Core.Manager.Manager import Manager


class BasePrediction(BasePipeline):

    def __init__(self,
                 network_config: BaseNetworkConfig,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 database_config: Optional[BaseDatabaseConfig] = None,
                 session_dir: str = 'session',
                 session_name: str = 'training',
                 nb_steps: int = -1,
                 record: bool = False):
        """
        BasePrediction is a pipeline defining the running process of an artificial neural network.
        It provides a highly tunable learning process that can be used with any machine learning library.

        :param network_config: Specialisation containing the parameters of the network manager.
        :param environment_config: Specialisation containing the parameters of the environment manager.
        :param database_config: Specialisation containing the parameters of the dataset manager.
        :param session_name: Name of the newly created directory if session is not defined.
        :param session_dir: Name of the directory in which to write all the necessary data.
        :param nb_steps: Number of simulation step to play.
        :param record: Save or not the prediction data.
        """

        BasePipeline.__init__(self,
                              network_config=network_config,
                              database_config=database_config,
                              environment_config=environment_config,
                              session_dir=session_dir,
                              session_name=session_name,
                              pipeline='prediction')

        # Prediction variables
        self.nb_samples = nb_steps
        self.idx_step = 0

        # Tell if data is recording while predicting
        self.record_data = record
        self.is_environment = environment_config is not None

        self.manager = Manager(network_config=self.network_config,
                               database_config=database_config,
                               environment_config=self.environment_config,
                               session_dir=session_dir,
                               session_name=session_name,
                               new_session=False,
                               pipeline='prediction',
                               produce_data=environment_config is not None and record,
                               batch_size=1)

    def execute(self) -> None:
        """
        | Main function of the running process "execute" call the functions associated with the learning process.
        | Each of the called functions are already implemented so one can start a basic run session.
        | Each of the called function can also be rewritten via inheritance to provide more specific / complex running
          process.
        """

        self.run_begin()
        while self.running_condition():
            self.sample_begin()
            self.sample_end(self.predict())
        self.run_end()

    def predict(self, animate: bool = True) -> ndarray:
        """
        | Pull the data from the manager and return the prediction

        :param bool animate: True if getData fetch from the environment
        :return: Prediction from the Network
        """

        self.manager.get_data(animate=animate)
        data = self.manager.data_manager.data['input']
        data = self.manager.data_manager.normalize_data(data, 'input')
        return self.manager.network_manager.compute_online_prediction(data)

    def run_begin(self) -> None:
        """
        | Called once at the very beginning of the Run process.
        | Allows the user to run some pre-computations.
        """

        pass

    def run_end(self) -> None:
        """
        | Called once at the very end of the Run process.
        | Allows the user to run some post-computations.
        """

        pass

    def running_condition(self) -> bool:
        """
        | Condition that characterize the end of the running process.

        :return: False if the training needs to stop.
        """

        running = self.idx_step < self.nb_samples if self.nb_samples > 0 else True
        self.idx_step += 1
        return running

    def sample_begin(self) -> None:
        """
        | Called one at the start of each step.
        | Allows the user to run some pre-step computations.
        """

        pass

    def sample_end(self, prediction: ndarray) -> None:
        """
        | Called one at the end of each step.
        | Allows the user to run some post-step computations.

        :param ndarray prediction: Prediction of the Network.
        """

        if self.is_environment:
            self.manager.data_manager.apply_prediction(prediction)

    def close(self) -> None:
        """
        | End the running process and close all the managers
        """

        self.manager.close()

    def __str__(self) -> str:
        """
        :return: str Contains running information about the running process
        """

        description = ""
        description += f"Running statistics :\n"
        description += f"Number of simulation step: {self.nb_samples}\n"
        description += f"Record inputs : {self.record_data[0]}\n"
        description += f"Record outputs : {self.record_data[1]}\n"

        return description
