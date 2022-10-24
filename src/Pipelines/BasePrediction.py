from typing import Optional

from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Pipelines.BasePipeline import BasePipeline
from DeepPhysX.Core.Manager.Manager import Manager


class BasePrediction(BasePipeline):

    def __init__(self,
                 network_config: BaseNetworkConfig,
                 environment_config: BaseEnvironmentConfig,
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

        self.manager = Manager(network_config=self.network_config,
                               database_config=database_config,
                               environment_config=self.environment_config,
                               session_dir=session_dir,
                               session_name=session_name,
                               new_session=False,
                               pipeline='prediction',
                               produce_data=record,
                               batch_size=1)

    def execute(self) -> None:
        """
        Launch the prediction Pipeline.
        Each event is already implemented for a basic pipeline but can also be rewritten via inheritance to describe a
        more complex pipeline.
        """

        self.prediction_begin()
        while self.prediction_condition():
            self.sample_begin()
            self.predict()
            self.sample_end()
        self.run_end()

    def prediction_begin(self) -> None:
        """
        Called once at the beginning of the prediction pipeline.
        """

        pass

    def prediction_condition(self) -> bool:
        """
        Condition that characterize the end of the prediction pipeline.
        """

        running = self.idx_step < self.nb_samples if self.nb_samples > 0 else True
        self.idx_step += 1
        return running

    def sample_begin(self) -> None:
        """
        Called one at the beginning of each sample.
        """

        pass

    def predict(self) -> None:
        """
        Pull the data from the manager and return the prediction.
        """

        self.manager.get_data(animate=True)

    def sample_end(self) -> None:
        """
        Called one at the end of each sample.
        """

        pass

    def run_end(self) -> None:
        """
        Called once at the end of the prediction pipeline.
        """

        self.manager.close()

    def __str__(self) -> str:

        description = ""
        description += f"Running statistics :\n"
        description += f"Number of simulation step: {self.nb_samples}\n"
        description += f"Record data : {self.record_data}\n"

        return description
