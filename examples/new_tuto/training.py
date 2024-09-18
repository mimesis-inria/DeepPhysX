"""
training.py
Run the pipeline BaseTrainer to create a training session with an existing Dataset.
"""

# Python related imports
from os.path import exists
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTraining import BaseTraining
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig


if __name__ == '__main__':

    if not exists('sessions/data_generation'):
        quit(print("Existing Dataset required, 'sessions/data_generation' not found. Run dataset.py script first."))

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(lr=1e-4,
                              loss=MSELoss,
                              optimizer=Adam,
                              dim_layers=[6, 6, 2],
                              dim_output=2)

    # Dataset configuration with the path to the existing Dataset
    database_config = BaseDatabaseConfig(
        existing_dir='sessions/data_generation/',
        shuffle=True,
        normalize=True)

    # Create DataGenerator
    trainer = BaseTraining(network_config=network_config,
                           database_config=database_config,
                           session_dir='sessions',
                           session_name='training',
                           epoch_nb=100,
                           batch_nb=100,
                           batch_size=32)

    # Launch the training session
    trainer.execute()
