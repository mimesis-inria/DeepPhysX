# Python related imports
from os.path import exists
from shutil import rmtree
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX.pipelines import DataPipeline, TrainingPipeline
from DeepPhysX.simulation.simulation_config import SimulationConfig
from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.networks.architectures.mlp.mlp_config import MLPConfig

# Session related imports
from simulation import SpringEnvironment

if __name__ == '__main__':

    if not exists('sessions/data_generation'):

        # Environment configuration
        simulation_config = SimulationConfig(environment_class=SpringEnvironment,
                                              use_viewer=False,
                                              nb_parallel_env=1,
                                              simulations_per_step=5)
        # Database configuration
        database_config = DatabaseConfig()

        # Create DataGenerator
        data_generator = DataPipeline(simulation_config=simulation_config,
                                        database_config=database_config,
                                        batch_nb=1000,
                                        batch_size=32)

        # Launch the training session
        data_generator.execute()

    # TODO: remove this line
    if exists('sessions/offline_training'):

        rmtree('sessions/offline_training')

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = MLPConfig(lr=1e-5,
                               loss=MSELoss,
                               optimizer=Adam,
                               dim_layers=[5, 5, 2],
                               dim_output=2)

    # Dataset configuration with the path to the existing Dataset
    database_config = DatabaseConfig(existing_dir='sessions/data_generation/',
                                     shuffle=True,
                                     normalize=True)

    # Create DataGenerator
    trainer = TrainingPipeline(network_config=network_config,
                       database_config=database_config,
                       session_dir='sessions',
                       session_name='offline_training',
                       epoch_nb=30,
                       batch_nb=1000,
                       batch_size=32)

    # Launch the training session
    trainer.execute()
