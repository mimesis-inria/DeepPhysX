"""
prediction.py
Run the pipeline BaseRunner to check the predictions of the trained networks.
"""

# Python related imports
from os.path import exists

# DeepPhysX related imports
from DeepPhysX.pipelines.core import Prediction
from DeepPhysX.simulation.core import SimulationConfig
from DeepPhysX.database import DatabaseConfig
from DeepPhysX.networks.architectures.mlp import MLPConfig

# Session imports
from simulation import SpringEnvironment


if __name__ == '__main__':

    session = 'sessions/offline_training'
    if not exists(session):
        session = 'sessions/online_training'
        if not exists(session):
            quit(print("Trained networks required. Run a training script first."))

    # Environment configuration
    environment_config = SimulationConfig(environment_class=SpringEnvironment,
                                          use_viewer=True)

    # Dataset configuration with the path to the existing Dataset
    database_config = DatabaseConfig(normalize=True)

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = MLPConfig(dim_layers=[5, 5, 2],
                               dim_output=2)

    # Create DataGenerator
    pipeline = Prediction(simulation_config=environment_config,
                          database_config=database_config,
                          network_config=network_config,
                          session_dir='sessions',
                          session_name=session.split('/')[1])

    # Launch the training session
    pipeline.execute()
