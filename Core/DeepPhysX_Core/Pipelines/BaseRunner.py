from typing import Optional
from numpy import ndarray

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX_Core.Manager.Manager import Manager


class BaseRunner(BasePipeline):
    """
    | BaseRunner is a pipeline defining the running process of an artificial neural network.
    | It provides a highly tunable learning process that can be used with any machine learning library.

    :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
    :param BaseEnvironmentConfig environment_config: Specialisation containing the parameters of the environment manager
    :param Optional[BaseDatasetConfig] dataset_config: Specialisation containing the parameters of the dataset manager
    :param str session_name: Name of the newly created directory if session_dir is not defined
    :param Optional[str] session_dir: Name of the directory in which to write all the necessary data
    :param int nb_steps: Number of simulation step to play
    :param bool record_inputs: Save or not the input in a numpy file
    :param bool record_outputs: Save or not the output in a numpy file
    """

    def __init__(self,
                 network_config: BaseNetworkConfig,
                 environment_config: Optional[BaseEnvironmentConfig] = None,
                 dataset_config: Optional[BaseDatasetConfig] = None,
                 session_name: str = 'default',
                 session_dir: Optional[str] = None,
                 nb_steps: int = 0,
                 record_inputs: bool = False,
                 record_outputs: bool = False):

        BasePipeline.__init__(self,
                              network_config=network_config,
                              dataset_config=dataset_config,
                              environment_config=environment_config,
                              session_name=session_name,
                              session_dir=session_dir,
                              pipeline='prediction')

        self.name = self.__class__.__name__

        if type(nb_steps) != int or nb_steps < 0:
            raise TypeError("[BaseRunner] The number of steps must be a positive int")

        self.nb_samples = nb_steps
        self.idx_step = 0

        # Tell if data is recording while predicting (output is recorded only if input too)
        self.record_data = {'input': False, 'output': False}
        if dataset_config is not None:
            self.record_data = {'input': record_inputs, 'output': record_outputs and record_inputs}
        self.is_environment = environment_config is not None

        self.manager = Manager(pipeline=self,
                               network_config=self.network_config,
                               dataset_config=dataset_config,
                               environment_config=self.environment_config,
                               session_name=session_name,
                               session_dir=session_dir,
                               new_session=True)

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
