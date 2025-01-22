"""
prediction.py
Run the pipeline BaseRunner to check the predictions of the trained networks.
"""

# Python related imports
from os.path import exists

# DeepPhysX related imports
from DeepPhysX.pipelines.prediction_pipeline import PredictionPipeline
from DeepPhysX.simulation import SimulationConfig
from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.networks.architecture.mlp import MLP
from DeepPhysX.networks.network_manager import NetworkManager

# Session imports
from simulation import SpringEnvironmentPrediction


if __name__ == '__main__':

    session = 'sessions/offline_training'
    if not exists(session):
        session = 'sessions/online_training'
        if not exists(session):
            quit(print("Trained networks required. Run a training script first."))

    # Environment configuration
    environment_config = SimulationConfig(environment_class=SpringEnvironmentPrediction,
                                          use_viewer=True)

    # Dataset configuration with the path to the existing Dataset
    database_manager = DatabaseManager(normalize=True)

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_manager = NetworkManager(network_architecture=MLP,
                                     network_kwargs={'dim_layers': [5, 5, 5, 1],
                                                     'out_shape': (1,)},
                                     data_forward_fields='state',
                                     data_backward_fields='displacement')

    # Create DataGenerator
    pipeline = PredictionPipeline(simulation_config=environment_config,
                                  database_manager=database_manager,
                                  network_manager=network_manager,
                                  session_dir='sessions',
                                  session_name=session.split('/')[1],
                                  step_nb=-1)

    # Launch the training session
    pipeline.execute()
