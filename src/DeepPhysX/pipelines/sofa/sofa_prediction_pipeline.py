from typing import Optional
import Sofa

from DeepPhysX.pipelines.prediction_pipeline import PredictionPipeline as _PredictionPipeline
from DeepPhysX.networks.network_config import NetworkConfig
from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.simulation.sofa.sofa_environment_config import SofaEnvironmentConfig


class PredictionPipeline(Sofa.Core.Controller, _PredictionPipeline):

    def __init__(self,
                 network_config: NetworkConfig,
                 environment_config: SofaEnvironmentConfig,
                 database_config: Optional[DatabaseConfig] = None,
                 session_dir: str = 'session',
                 session_name: str = 'training',
                 step_nb: int = -1,
                 record: bool = False,
                 *args, **kwargs):
        """
        SofaPrediction is a pipeline defining the running process of an artificial neural networks.
        It provides a highly tunable learning process that can be used with any machine learning library.

        :param network_config: Configuration object with the parameters of the Network.
        :param environment_config: Configuration object with the parameters of the Environment.
        :param database_config: Configuration object with the parameters of the Database.
        :param session_dir: Relative path to the directory which contains sessions repositories.
        :param session_name: Name of the new the session repository.
        :param step_nb: Number of simulation step to play.
        :param record: If True, prediction data will be saved in a dedicated Database.
        """

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        _PredictionPipeline.__init__(self,
                                network_config=network_config,
                                database_config=database_config,
                                environment_config=environment_config,
                                session_name=session_name,
                                session_dir=session_dir,
                                step_nb=step_nb,
                                record=record)

        self.prediction_begin()
        self.root = self.data_manager.environment_manager.environment.root
        self.root.addObject(self)
        self.load_samples = environment_config.load_samples

    def onAnimateBeginEvent(self, _):

        if self.load_samples:
            sample_id = self.data_manager.load_sample()
            self.data_manager.environment_manager.environment._get_training_data(sample_id)

    def onAnimateEndEvent(self, _):
        """
        Called within the Sofa pipeline at the end of the time step.
        """

        if self.prediction_condition():
            self.sample_begin()
            self.predict()
            self.sample_end()

    def predict(self) -> None:
        """
        Pull the data from the manager and return the prediction.
        """

        self.data_manager.get_data(epoch=0,
                                   animate=False,
                                   load_samples=not self.load_samples)

    def close(self) -> None:
        """
        Manually trigger the end of the Pipeline.
        """

        self.prediction_end()
