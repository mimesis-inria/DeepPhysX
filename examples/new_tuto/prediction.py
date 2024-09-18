"""
prediction.py
Run the pipeline BaseRunner to check the predictions of the trained network.
"""

# Python related imports
from os.path import exists

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BasePrediction import BasePrediction
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig

# Session imports
from environment import SpringEnvironment


if __name__ == '__main__':

    if not exists('sessions/training'):
        quit(print("Trained Network required, 'sessions/training' not found. Run training.py script first."))

    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=SpringEnvironment,
                                               visualizer='vedo')

    # Dataset configuration with the path to the existing Dataset
    database_config = BaseDatabaseConfig(normalize=False)

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(dim_layers=[6, 6, 2],
                              dim_output=2)

    # Create DataGenerator
    trainer = BasePrediction(environment_config=environment_config,
                             database_config=database_config,
                             network_config=network_config,
                             session_dir='sessions',
                             session_name='training')

    # Launch the training session
    trainer.execute()
