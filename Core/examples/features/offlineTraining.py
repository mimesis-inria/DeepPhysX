"""
offlineTraining.py
Run the pipeline BaseTrainer to create a training session with an existing Dataset.
"""

# Python related imports
import os
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Torch.FC.FCConfig import FCConfig


def launch_training():
    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3
    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = FCConfig(loss=MSELoss,
                              lr=1e-3,
                              optimizer=Adam,
                              dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                              dim_output=dimension)
    # Dataset configuration with the path to the existing Dataset
    dataset_config = BaseDatasetConfig(dataset_dir=os.path.join(os.getcwd(), 'sessions/data_generation'),
                                       shuffle_dataset=True,
                                       normalize=False)
    # Create DataGenerator
    trainer = BaseTrainer(session_dir='sessions',
                          session_name='offline_training',
                          dataset_config=dataset_config,
                          network_config=network_config,
                          nb_epochs=1,
                          nb_batches=500,
                          batch_size=10)
    # Launch the training session
    trainer.execute()


if __name__ == '__main__':
    if not os.path.exists(os.path.join(os.getcwd(), 'sessions/data_generation')):
        print("Existing Dataset required, 'sessions/data_generation' not found. "
              "Run dataGeneration_single.py script first.")
        from dataGeneration_single import launch_data_generation
        launch_data_generation()
    launch_training()
